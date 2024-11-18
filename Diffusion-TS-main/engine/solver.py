import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])

        # Check first batch shape from the DataLoader
        first_batch = next(self.dl)
        print("First batch shape:", first_batch.shape if hasattr(first_batch, 'shape') else "Shape unknown")

        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                'Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0

        # Start training log
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        # Progress bar for training steps
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    try:
                        # Fetch data and print its shape
                        data = next(self.dl).to(device)
                        #print(f"Data shape at step {step}: {data.shape}")
                        loss = self.model(data, target=data)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                        total_loss += loss.item()
                    except StopIteration:
                        print("Data loader exhausted; ensure it loops correctly or resets.")
                        break
                    except Exception as e:
                        print(f"Error during gradient accumulation: {e}")
                        continue

                # Updating progress bar with loss information
                pbar.set_description(f'loss: {total_loss:.6f}')

                # Gradient clipping and optimizer step
                try:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                    self.opt.zero_grad()
                    self.step += 1
                    step += 1
                    self.ema.update()
                except Exception as e:
                    print(f"Error during optimizer step: {e}")

                # Checkpoint saving logic
                try:
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        # Debugging print statements to confirm save function call
                        print(f"Attempting to save checkpoint at step {self.step}, milestone {self.milestone}")
                        os.makedirs(self.results_folder, exist_ok=True)  # Ensure results folder exists
                        self.save(self.milestone)
                        print(f"Checkpoint saved: {str(self.results_folder / f'checkpoint-{self.milestone}.pt')}")
                except Exception as e:
                    print(f"Error saving checkpoint at step {self.step}: {e}")

                # Logger updates
                try:
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)
                except Exception as e:
                    print(f"Error logging training metrics: {e}")

                # Progress bar update
                pbar.update(1)

        # Training complete log
        print('Training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None, input_data=None):

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')

        if input_data is not None and input_data.ndim != 3:
            raise ValueError(f"Expected input_data to be 3D, but got shape {input_data.shape}")

        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        print("\n--- Starting sampling process ---")
        print(f"Initial empty samples shape: {samples.shape}")

        for cycle in range(num_cycle):
            # Debugging: input_data slice
            if input_data is not None:
                initial_state = input_data[cycle * size_every:(cycle + 1) * size_every, :, :]
                print(f"Cycle {cycle + 1}: Using input_data slice with shape={initial_state.shape}")
            else:
                initial_state = None
                print(f"Cycle {cycle + 1}: No input_data provided, initial_state=None")

            sample = self.ema.ema_model.generate_mts(batch_size=size_every, initial_state=initial_state)

            # Debugging: generated samples
            print(f"Cycle {cycle + 1}: First generated sequence starting point: {sample[0, 0, :]}")
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            print(f"Updated samples shape after stacking: {samples.shape}")

            torch.cuda.empty_cache()

        print("Final sampled data shape:", samples.shape)
        print("--- Sampling process complete ---\n")

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')

        model_kwargs = {
            'coef': coef,
            'learning_rate': stepsize
        }
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        print("\n--- Starting restore process ---")

        for idx, (x, t_m) in enumerate(raw_dataloader):
            # Debugging output
            print(f"\nRestore Iteration {idx + 1}")
            print("Input x shape:", x.shape)
            print("Input t_m shape:", t_m.shape)
            print("Input x values (first sequence, first 5 entries):",
                  x[0, :5].cpu().numpy() if x.shape[0] > 0 else "Empty")

            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                               model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            # Additional debugging output
            print("Generated sample shape:", sample.shape)
            print("Generated sample values (first sequence, first 5 entries):",
                  sample[0, :5].cpu().numpy() if sample.shape[0] > 0 else "Empty")

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

            # After stacking, check updated shapes
            print("Updated samples shape after stacking:", samples.shape)
            print("Updated reals shape after stacking:", reals.shape)
            print("Updated masks shape after stacking:", masks.shape)

        print("Final restored data shape:", samples.shape)
        print("Final reals data shape:", reals.shape)
        print("Final masks data shape:", masks.shape)
        print("--- Restore process complete ---\n")

        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
