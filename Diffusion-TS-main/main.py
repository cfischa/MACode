import os
import torch
import argparse
import numpy as np
import argparse
from pathlib import Path
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    # Get the absolute path to the directory where this script is located
    base_dir = Path(__file__).resolve().parent

    # Dynamically set the default config file path
    default_config_path = base_dir / 'Config' / 'weather_data.yaml'

    # Default output directory as a subfolder named 'OUTPUT' in the current directory
    default_output_path = base_dir / 'OUTPUT'

    # set default =None to enter from command line
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default='hellisheidi_weather_cleaned_final')

    # Update config_file argument to use cross-platform path
    parser.add_argument('--config_file', type=str, default=str(default_config_path), help='path of config file')

    # Update output argument to use cross-platform path
    parser.add_argument('--output', type=str, default=str(default_output_path), help='directory to save the results')


    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                             ' used, and ddp will be disabled')

    # args for training
    # true/false for training if you choose false sampling is running
    # While training, the script will save check points to the results folder after a fixed number of epochs. Once trained, please use the saved model for sampling by running
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    # if set to 1 is uses the checkpoints
    parser.add_argument('--sample', type=int, default=1,
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='Forecasting',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10) # default=10

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=1, help='Length of Predictions.')

    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    if args.gpu is not None:
        print(f"GPU setting ignored, using CPU instead.")

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    #model = instantiate_from_config(config['model'])  # Keep the model on CPU
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']

        # Restore samples using deterministic initialization
        samples, *_ = trainer.restore(
            dataloader,
            [dataset.window, dataset.var_num],
            coef,
            stepsize,
            sampling_steps
        )

        # Generate additional samples with deterministic initialization
        generated_samples = trainer.sample(
            num=len(dataset),
            size_every=2001,
            shape=[dataset.window, dataset.var_num],
        )

        # Save normalized restored samples
        np.save(os.path.join(args.save_dir, f'ddpm_restored_normalized{args.mode}_{args.name}.npy'), samples)

        if dataset.auto_norm:
            # Unnormalize restored samples
            samples = unnormalize_to_zero_to_one(samples)
            samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
            # Save denormalized restored samples
            np.save(os.path.join(args.save_dir, f'ddpm_restored_denormalized{args.mode}_{args.name}.npy'), samples)

        # Save normalized generated samples
        np.save(os.path.join(args.save_dir, f'ddpm_generated_normalized{args.mode}_{args.name}.npy'), generated_samples)

        if dataset.auto_norm:
            # Unnormalize generated samples
            generated_samples = unnormalize_to_zero_to_one(generated_samples)
            generated_samples = dataset.scaler.inverse_transform(
                generated_samples.reshape(-1, generated_samples.shape[-1])
            ).reshape(generated_samples.shape)
            # Save denormalized generated samples
            np.save(os.path.join(args.save_dir, f'ddpm_generated_denormalized{args.mode}_{args.name}.npy'),
                    generated_samples)

    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']


        # Generate samples with deterministic initialization
        samples = trainer.sample(
            num=len(dataset),
            size_every=2001,
            shape=[dataset.window, dataset.var_num],
        )

        # Save normalized samples
        np.save(os.path.join(args.save_dir, f'ddpm_fake_normalized{args.name}.npy'), samples)

        if dataset.auto_norm:
            # Unnormalize the samples if required
            samples = unnormalize_to_zero_to_one(samples)
            samples = dataset.scaler.inverse_transform(
                samples.reshape(-1, samples.shape[-1])
            ).reshape(samples.shape)

        # Save denormalized samples
        np.save(os.path.join(args.save_dir, f'ddpm_fake_denormalized{args.name}.npy'), samples)


if __name__ == '__main__':
    main()
