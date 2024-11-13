import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import json
import pathlib
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sb

from torch.utils.tensorboard import SummaryWriter

import argparse

from ddpm import *  # Importing custom diffusion model classes
from data_make import *  # Importing data generation and processing functions

import warnings

warnings.filterwarnings('ignore')


def visualize(ori_data, fake_data, dataset_name, seq_len, save_path, epoch, writer):
    """
    Visualizes the real and synthetic data using PCA and t-SNE, and saves the visualizations.

    Inputs:
    - ori_data: Original data (real data).
    - fake_data: Generated synthetic data from the model.
    - dataset_name: Name of the dataset being used.
    - seq_len: Length of the sequence for time series data.
    - save_path: Directory path to save the visualizations.
    - epoch: Current epoch of training, used for labeling the outputs.
    - writer: TensorBoard writer to log the figures.

    Outputs:
    - Saves PCA and t-SNE visualizations comparing real and synthetic data.
    """
    ori_data = np.asarray(ori_data)
    fake_data = np.asarray(fake_data)

    # Truncate original data to match the size of the fake data
    ori_data = ori_data[:fake_data.shape[0]]

    # Randomly sample a subset of the data for visualization
    sample_size = 250
    idx = np.random.permutation(len(ori_data))[:sample_size]
    randn_num = np.random.permutation(sample_size)[:1]

    # Select random samples from the real and synthetic data
    real_sample = ori_data[idx]
    fake_sample = fake_data[idx]

    # Reshape the samples for PCA and t-SNE
    real_sample_2d = real_sample.reshape(-1, seq_len)
    fake_sample_2d = fake_sample.reshape(-1, seq_len)

    mode = 'visualization'

    # PCA: Principal Component Analysis for dimensionality reduction
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = pd.DataFrame(pca.transform(real_sample_2d)).assign(Data='Real')
    pca_synthetic = pd.DataFrame(pca.transform(fake_sample_2d)).assign(Data='Synthetic')
    pca_result = pd.concat([pca_real, pca_synthetic], ignore_index=True).rename(columns={0: '1st Component', 1: '2nd Component'})

    # t-SNE: t-Distributed Stochastic Neighbor Embedding for dimensionality reduction
    tsne_data = np.concatenate((real_sample_2d, fake_sample_2d), axis=0)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'

    # Create subplots for visualizations
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

    # Plot PCA results
    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result, hue='Data', style='Data', ax=axs[0, 0])
    sb.despine()
    axs[0, 0].set_title('PCA Result')

    # Plot t-SNE results
    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axs[0, 1])
    sb.despine()
    axs[0, 1].set_title('t-SNE Result')

    # Ensure index is within bounds for plotting
    randn_num[0] = randn_num[0] % real_sample.shape[0]
    axs[1, 0].plot(real_sample[randn_num[0], :, :])
    axs[1, 0].set_title('Original Data')

    # Plot synthetic data samples
    axs[1, 1].plot(fake_sample[randn_num[0], :, :])
    axs[1, 1].set_title('Synthetic Data')

    # Main title and layout adjustments
    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)

    # Save the main plot
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-tsne-result-{dataset_name}.png'))
    writer.add_figure(mode, fig, epoch)

    # Plot each feature separately for synthetic data
    num_features = fake_sample.shape[2]
    fig_features, axs_features = plt.subplots(ncols=1, nrows=num_features, figsize=(15, num_features * 4))

    # Ensure axs_features is always an array
    axs_features = np.atleast_1d(axs_features)

    # Plot each feature of the synthetic data
    for feature_idx in range(num_features):
        axs_features[feature_idx].plot(fake_sample[randn_num[0], :, feature_idx], label=f'Feature {feature_idx + 1}')
        axs_features[feature_idx].set_title(f'Synthetic Data - Feature {feature_idx + 1}')
        axs_features[feature_idx].set_xlabel('Time Step')
        axs_features[feature_idx].set_ylabel('Value')
        axs_features[feature_idx].legend()
        axs_features[feature_idx].grid(True)

    # Final layout adjustments and save
    fig_features.tight_layout()
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-synthetic-features-{dataset_name}.png'))
    writer.add_figure('synthetic_features', fig_features, epoch)
    plt.show()


def train_main(args):
    """
    Main training function for the diffusion model using transformer encoder architecture.

    Inputs:
    - args: Command-line arguments specifying training parameters like sequence length, number of epochs, etc.

    This function:
    - Loads and preprocesses data.
    - Initializes the model, optimizer, and training setup.
    - Trains the model over several epochs.
    - Logs performance metrics and saves generated synthetic samples periodically.
    """
    # Extract training parameters from args
    seq_len = args.seq_len
    epochs = args.training_epoch
    timesteps = args.timesteps
    batch_size = args.batch_size
    latent_dim = args.hidden_dim
    num_layers = args.num_of_layers
    n_heads = args.n_head
    dataset_name = args.dataset_name
    beta_schedule = args.beta_schedule
    objective = args.objective

    # Load and preprocess data
    train_data, test_data = LoadData(dataset_name, seq_len)
    train_data, test_data = np.asarray(train_data), np.asarray(test_data)
    features = train_data.shape[2] # Extracts the number of features per time step

    # Transpose data to fit model requirements (batch, features, sequence length)
    train_data, test_data = train_data.transpose(0, 2, 1), test_data.transpose(0, 2, 1)

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, len(test_data))

    # Get a batch of real data for comparison
    real_data = next(iter(test_loader))

    # Set the device for training (CPU in this case)
    device = 'cuda'
    mode = 'diffusion'
    architecture = 'custom-transformers'
    loss_mode = 'l1'

    # Define file paths for saving models and results
    file_name = f'{architecture}-{dataset_name}-{loss_mode}-{beta_schedule}-{seq_len}-{objective}'
    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    gan_fig_dir_path = f'{folder_name}/output/gan'
    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True)
    file_name_gan_fig = f'{file_name}-gan'

    # Save training parameters to a file
    with open(f'{folder_name}/params.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.close()

        # Initialize TensorBoard writer for logging
    writer = SummaryWriter(log_dir=folder_name, comment=f'{file_name}', flush_secs=45)

    # Initialize the transformer encoder model
    model = TransEncoder(
        features=features,
        latent_dim=latent_dim,
        num_heads=n_heads,
        num_layers=num_layers
    )

    # Initialize the diffusion model with the transformer encoder as the base
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_len,
        timesteps=timesteps,
        objective=objective,  # Prediction objective (e.g., predicting initial state or noise)
        loss_type='l2',  # Loss type used for training
        beta_schedule=beta_schedule  # Schedule for noise variance in the diffusion process
    )

    diffusion = diffusion.to(device)

    # Set up the optimizer
    lr = 1e-4
    betas = (0.9, 0.99)
    optim = torch.optim.Adam(diffusion.parameters(), lr=lr, betas=betas)

    # Training loop
    for running_epoch in tqdm(range(epochs)):
        for i, data in enumerate(train_loader):
            data = data.to(device)  # Move data to the device (CPU in this case)

            optim.zero_grad()  # Zero the gradients
            loss = diffusion(data)  # Forward pass through the diffusion model

            loss.backward()  # Backpropagate the loss
            optim.step()  # Update model parameters

            # Log the loss to TensorBoard
            if i % len(train_loader) == 0:
                writer.add_scalar('Loss', loss.item(), running_epoch)

            # Print loss every 100 epochs
            if i % len(train_loader) == 0 and running_epoch % 100 == 0:
                print(f'Epoch: {running_epoch + 1}, Loss: {loss.item()}')

            # Save model and visualize data every 500 epochs
            if i % len(train_loader) == 0 and running_epoch % 500 == 0:
                with torch.no_grad():
                    samples = diffusion.sample(len(test_data))

                    samples = samples.cpu().numpy()

                    np.save(f'{folder_name}/synth_normalized_untransposed-{dataset_name}-{seq_len}-{running_epoch}.npy', samples)

                    samples = samples.transpose(0, 2, 1)

                    np.save(f'{folder_name}/synth_normalized_transposed-{dataset_name}-{seq_len}-{running_epoch}.npy', samples)

                visualize(real_data.cpu().numpy().transpose(0,2,1), samples, dataset_name, seq_len, gan_fig_dir_path, running_epoch, writer)

    # Save the final model state
    torch.save({
        'epoch': running_epoch + 1,
        'diffusion_state_dict': diffusion.state_dict(),
        'diffusion_optim_state_dict': optim.state_dict()
    }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))