# modules/autoencoder_training.py
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.optim as optim
import random
from ModelCode.timeautoencoder import DeapStack, auto_loss
import ModelCode.process_edited as pce

def train_autoencoder(real_df, processed_data, params, device, progress_bar=None):
    parser = pce.DataFrameParser().fit(real_df, params['threshold'])
    data = parser.transform()
    data = torch.tensor(data.astype('float32')).unsqueeze(0)

    datatype_info = parser.datatype_info()
    n_bins, n_cats, n_nums, cards = datatype_info['n_bins'], datatype_info['n_cats'], datatype_info['n_nums'], datatype_info['cards']

    ae = DeapStack(
        params['channels'], params['batch_size'], processed_data.shape[1],
        n_bins, n_cats, n_nums, cards, processed_data.shape[2],
        params['hidden_size'], params['num_layers'], params['emb_dim'],
        params['time_dim'], params['lat_dim']
    ).to(device)

    optimizer = Adam(ae.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    inputs = processed_data.to(device)

    best_loss = float('inf')
    patience = 0
    beta = params['max_beta']
    losses = []

    if progress_bar is None:
        # Use rich.progress.Progress if no progress_bar is provided
        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("[red]Training Autoencoder...", total=params['n_epochs'])
            for epoch in range(params['n_epochs']):
                batch_indices = random.sample(range(len(inputs)), params['batch_size'])
                optimizer.zero_grad()
                outputs, _, mu, logvar = ae(inputs[batch_indices])
                disc_loss, num_loss = auto_loss(inputs[batch_indices], outputs, n_bins, n_nums, n_cats, beta, cards)
                loss_kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = num_loss + disc_loss + beta * loss_kld
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                progress.update(task, advance=1, description=f"Epoch {epoch+1}: Loss {loss.item():.4f}")

                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10 and beta > params['min_beta']:
                        beta *= 0.7
    else:
        # Use Streamlit progress bar
        for epoch in range(params['n_epochs']):
            batch_indices = random.sample(range(len(inputs)), params['batch_size'])
            optimizer.zero_grad()
            outputs, _, mu, logvar = ae(inputs[batch_indices])
            disc_loss, num_loss = auto_loss(inputs[batch_indices], outputs, n_bins, n_nums, n_cats, beta, cards)
            loss_kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = num_loss + disc_loss + beta * loss_kld
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update the Streamlit progress bar
            progress = (epoch + 1) / params['n_epochs']
            progress_bar.progress(progress)

            # Optionally, you can display the loss in Streamlit
            # st.write(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

            if loss < best_loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience >= 10 and beta > params['min_beta']:
                    beta *= 0.7

    return ae, losses
