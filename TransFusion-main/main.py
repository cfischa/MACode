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

import warnings
warnings.filterwarnings('ignore')
import os
import platform
from pathlib import Path
import pandas as pd
import numpy as np

from ddpm import *
from data_make import *
from train import *
from long_predictive_score import *
from long_discriminative_score import *
from pathlib import Path

def check_file_path(file_path):
    """Check if the specified file path exists and is accessible."""
    path = Path(file_path)
    if path.exists():
        print(f"File exists and is accessible: {file_path}")
    else:
        print(f"File does not exist or is inaccessible: {file_path}")

def load_csv_data(file_path):
    """Function to load data from a CSV file."""
    return pd.read_csv(file_path)


def load_npy_data(npy_path):

    # Load data from a .npy file using NumPy
    data = np.load(npy_path)

    # Explicitly convert the NumPy array to a PyTorch tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Check and confirm the conversion
    print(f"Converted generated data type: {type(tensor_data)}")  # Should print <class 'torch.Tensor'>
    return tensor_data


def reshape_data_to_tensor(data, seq_len):
    # Remove any non-numeric columns to focus on feature data
    numeric_data = data.select_dtypes(include=[float, int])

    # Determine the number of features from the data
    num_features = numeric_data.shape[1]

    # Convert the numeric data to a tensor
    values = torch.tensor(numeric_data.values, dtype=torch.float32)

    # Calculate the number of full sequences
    num_sequences = len(values) // seq_len

    # Trim the data to fit the number of full sequences
    trimmed_values = values[:num_sequences * seq_len]

    # Reshape to 3D tensor: (num_sequences, seq_len, num_features)
    reshaped_tensor = trimmed_values.view(num_sequences, seq_len, num_features)

    return reshaped_tensor


def show_generated_data(generated_data, reshaped_ori_data_tensor):
    # Check if generated_data is a PyTorch tensor and convert it to a NumPy array if necessary
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()  # Convert to NumPy array, move to CPU if on GPU

    # Check if original_data is a PyTorch tensor and convert it to a NumPy array if necessary
    if isinstance(reshaped_ori_data_tensor, torch.Tensor):
        original_data = reshaped_ori_data_tensor.cpu().numpy()  # Convert to NumPy array, move to CPU if on GPU

    # Extract the shape of the data: (num_sequences, seq_len, num_features)
    num_sequences, seq_len, num_features = generated_data.shape

    # Set up the figure for plotting all sequences
    plt.figure(figsize=(15, num_sequences * 2))  # Adjust the figure size to accommodate all sequences

    # Plot each sequence in a separate subplot
    for seq_idx in range(num_sequences):
        plt.subplot(num_sequences, 1, seq_idx + 1)  # Create a subplot for each sequence

        for feature_idx in range(num_features):
            plt.plot(generated_data[seq_idx, :, feature_idx], label=f'Generated Feature {feature_idx + 1}', linestyle='--')
            plt.plot(original_data[seq_idx, :, feature_idx], label=f'Original Feature {feature_idx + 1}', linestyle='-')

        plt.title(f'Sequence {seq_idx + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()


# only for vizualization etc
def load_data(seq_len):
    """Load and reshape data based on sequence length."""
    # Define paths directly in the script
    #ori_data_path = '/home/sc.uni-leipzig.de/ys09emuw/MACode/TransFusion-main/data/weather_data.csv'
    #generated_data_path = '/home/sc.uni-leipzig.de/ys09emuw/MACode/TransFusion-main/saved_files/generated_data.npy'
    #save_reshaped_data_path = '/home/sc.uni-leipzig.de/ys09emuw/MACode/TransFusion-main/saved_files/reshaped_ori_data_tensor.npy'


    ori_data = load_csv_data(ori_data_path)  # Load original data from CSV
    reshaped_ori_data_tensor = reshape_data_to_tensor(ori_data, seq_len)  # Reshape the original data into sequences

    # Save the reshaped original data tensor to the specified path in .npy format
    np.save(save_reshaped_data_path, reshaped_ori_data_tensor.numpy())  # Convert tensor to NumPy and save as .npy

    # Load the generated data from .npy (seem like being set before model training completed
    generated_data = load_npy_data(generated_data_path)


    # Check and convert to ensure inputs are tensors
    assert isinstance(reshaped_ori_data_tensor, torch.Tensor), "reshaped_ori_data_tensor must be a PyTorch tensor"
    assert isinstance(generated_data, torch.Tensor), "generated_data must be a PyTorch tensor"

    return reshaped_ori_data_tensor, generated_data
    


def run_discriminative(seq_len):
    """Run discriminative score metrics."""
    reshaped_ori_data_tensor, generated_data = load_data(seq_len)
    discriminative_score = long_discriminative_score_metrics(reshaped_ori_data_tensor, generated_data)
    print(discriminative_score)

def run_predictive(seq_len):
    """Run predictive score metrics."""
    reshaped_ori_data_tensor, generated_data = load_data(seq_len)
    predictive_score = long_predictive_score_metrics(reshaped_ori_data_tensor, generated_data)
    print(predictive_score)

def run_training(args):
    """Run the training module."""
    train_main(args)

def main(args):
    """Main function to select and run appropriate method."""
    # Use seq_len from args
    seq_len = args.seq_len

    # Define and hardcode the choice of operation here
    #choice = 'show_generated_data'  # Change this to 'discriminative', 'predictive', 'train', or 'show_generated_data' as needed
    choice = 'train'
    # Dictionary of available methods
    methods = {
        'discriminative': run_discriminative,
        'predictive': run_predictive,
        'train': run_training,
        'show_generated_data': show_generated_data,
    }

    # Run the selected method if available, else show error
    if choice in methods:
        if choice == 'train':
            methods[choice](args)  # Pass args for the train method
        elif choice == 'show_generated_data':
            # Load both original and generated data and pass them to the show function
            reshaped_ori_data_tensor, generated_data = load_data(seq_len)  # Ensure both are loaded correctly
            methods[choice](reshaped_ori_data_tensor, generated_data)  # Pass both datasets to the function
        else:
            methods[choice](seq_len)
    else:
        print(f"Invalid choice '{choice}'. Available options are: {', '.join(methods.keys())}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--dataset_name',
        choices=['sine', 'stock', 'air', 'energy', 'sinecurve_and_number_seq'],
        default='hellisheidi_weather_cleaned_final',
        type=str)
    parser.add_argument(
        '--beta_schedule',
        choices=['cosine', 'linear', 'quadratic', 'sigmoid'],
        default='cosine',
        type=str)

    # test pred_x0
    parser.add_argument(
        '--objective',
        choices=['pred_x0', 'pred_v', 'pred_noise'],
        default='pred_v',
        type=str)

    # defines how to capture temporal dependencies
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        #default=100,
        default=24,
        type=int)

    parser.add_argument(
        '--batch_size',
        help='batch size for the network',
        default=256,
        type=int)

    parser.add_argument(
        '--n_head',
        help='number of heads for the attention',
        default=8,
        type=int)

    parser.add_argument(
        '--hidden_dim',
        help='number of hidden state',
        default=256,
        type=int)

    parser.add_argument(
        '--num_of_layers',
        help='Number of Layers',
        default=6,
        type=int)

    parser.add_argument(
        '--training_epoch',
        help='Diffusion Training Epoch',
        default=5000,
        #default=2500,
        type=int)

    parser.add_argument(
        '--timesteps',
        help='Timesteps for Diffusion',
        #default=1000,
        default=250,
        type=int)

    args = parser.parse_args()


    main(args)
