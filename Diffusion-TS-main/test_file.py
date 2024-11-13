import numpy as np
import pandas as pd
import torch
from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config
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


# File paths
csv_path = './Data/datasets/test_dataset_50016_0.01.csv'
config_path = './Config/test_dataset_50016.yaml'  # Assuming this is the config file path

# Load original data from CSV
original_data = pd.read_csv(csv_path).values  # Convert CSV to numpy array
original_data = original_data.reshape(-1, 24, original_data.shape[1])  # Shape to [num_sequences, sequence_length, num_features]

# Extract starting values from original data
original_starts = original_data[:, 0, :]  # Get the first timestep for each sequence
original_mean_start = np.mean(original_starts, axis=0)
original_std_start = np.std(original_starts, axis=0)

print("Original Data Starting Values")
print("Mean of starting values:", original_mean_start)
print("Std deviation of starting values:", original_std_start)

# Load config and initialize Trainer
config = load_yaml_config(config_path)
args = argparse.Namespace()  # Create a blank namespace for arguments if needed
args.name = 'test_dataset_50016_0.01'
args.output = './OUTPUT'  # Adjust if necessary
args.save_dir = args.output
args.sample = 1  # Set to sample mode
args.mode = 'Forecasting'
args.milestone = 10

model = instantiate_from_config(config['model']).cuda()  # Instantiate the model
dataloader_info = build_dataloader(config, args)  # Build dataloader

# Initialize Trainer
trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info)

# Generate synthetic data
num_samples = original_data.shape[0]
synthetic_data = trainer.sample(num=num_samples, size_every=128, shape=original_data.shape[1:])

# Extract starting values from synthetic data
synthetic_starts = synthetic_data[:, 0, :]
synthetic_mean_start = np.mean(synthetic_starts, axis=0)
synthetic_std_start = np.std(synthetic_starts, axis=0)

print("\nSynthetic Data Starting Values")
print("Mean of starting values:", synthetic_mean_start)
print("Std deviation of starting values:", synthetic_std_start)

# Calculate differences
diff_mean = synthetic_mean_start - original_mean_start
diff_std = synthetic_std_start - original_std_start

print("\nDifferences between Synthetic and Original Starting Values")
print("Mean difference:", diff_mean)
print("Standard deviation difference:", diff_std)
