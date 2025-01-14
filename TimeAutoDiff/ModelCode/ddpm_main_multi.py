import numpy as np
import timeautoencoder as tae
import timediffusion as tdf
import DP as dp
import pandas as pd
import torch
import os
import time
import process_edited as pce
import matplotlib.pyplot as plt

# Define the base directory dynamically using the current file's location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
dataset_folder = os.path.join(os.path.dirname(base_dir), 'Dataset', 'Single-Sequence')
output_folder = os.path.join(os.path.dirname(base_dir), 'output')


# Filename for the dataset
filename = os.path.join(dataset_folder, 'hellisheidi_weather_cleaned_date_subset.csv')

# Read dataframe
print(filename)
real_df = pd.read_csv(filename)
#real_df1 = real_df.drop('date', axis=1).iloc[0:2000,:]
#real_df2= real_df.iloc[0:2000,:]

real_df1 = real_df.drop('date', axis=1)
real_df2 = real_df


# Pre-processing Data
threshold = 1
#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

column_to_partition = 'Symbol'
processed_data = dp.splitData(real_df1, 24, threshold).float()  # Cast to float32
time_info = dp.splitTimeData(real_df2, processed_data.shape[1]).to(device).float()  # Cast to float32


##############################################################################################################################
# Auto-encoder Training
#n_epochs = 50000
n_epochs = 5000
#n_epochs = 50
eps = 1e-5
weight_decay = 1e-6
lr = 2e-4
hidden_size = 200
num_layers = 1
batch_size = 50
channels = 64
# original min_beta = 1e-5, max_beta = 0.1
# test 1 min_beta = 1e-6, max_beta = 0.2
# test 2 min_beta = 1e-4, max_beta = 0.02
min_beta = 1e-4
max_beta = 0.02
emb_dim = 128
time_dim = 8
lat_dim = 7
seq_col = 'Symbol'

ds = tae.train_autoencoder(real_df1, processed_data.float(), channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
                           batch_size, threshold,  min_beta, max_beta, emb_dim, time_dim, lat_dim, device)

##############################################################################################################################
# Diffusion Training
latent_features = ds[1].float()
time = time_info.to(device)
hidden_dim = 200
num_layers = 2
# org: 100
diffusion_steps = 200
#n_epochs = 50000
n_epochs = 5000
#n_epochs = 50

num_classes = len(latent_features)

diff = tdf.train_diffusion(latent_features, time, hidden_dim, num_layers, diffusion_steps, n_epochs)

#############################################################################################################################
# Sampling multiple new data files
num_samples = 20  # Number of samples to generate
latent_features = ds[1]
# Increase T to generate longer samples (temporal depth)
T = latent_features.shape[1]
time_duration = []
# Modify N to generate more samples than in latent features (what I want)
N, _, _ = latent_features.shape
# Define t_grid starting point for sampling
t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device)

# Placeholder for all generated samples
all_samples = []

# Generate multiple samples and save them
for sample_idx in range(1, num_samples + 1):
    print(f"Generating sample {sample_idx}/{num_samples}...")

    # Sample new data
    samples = tdf.sample(
        t_grid.repeat(N, 1, 1),
        latent_features.detach().float().to(device),
        diff,
        time,
        connect_sequences=True  # Enable sequence connection
    )

    # Post-process the generated data
    gen_output = ds[0].decoder(samples.float().to(device))
    data_size, seq_len, _ = latent_features.shape
    synth_data = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)
    _synth_data = pce.convert_to_table(real_df1, synth_data, threshold)

    ##############################################################################################################################
    # Draw the plots for marginals of features: Real vs. Synthetic
    _real_data = pce.convert_to_table(real_df1, processed_data, threshold)

    B, L, K = _synth_data.shape
    sd_reshaped = _synth_data.reshape(B * L, K)
    pd_reshaped = _real_data.reshape(B * L, K)

    real_df = pd.DataFrame(pd_reshaped.numpy())
    synth_df = pd.DataFrame(sd_reshaped.numpy())

    ##############################################################################################################################
    # Save in/output tensors
    # Define file paths for current sample
    sample_output_folder = os.path.join(output_folder, f"sample_{sample_idx}")
    os.makedirs(sample_output_folder, exist_ok=True)  # Create subfolder for each sample

    processed_data_filepath = os.path.join(sample_output_folder, f"origi_hellisheidi_weather_cleaned_date_subset_sample{sample_idx}.npy")
    synth_data_filepath = os.path.join(sample_output_folder, f"synth_hellisheidi_weather_cleaned_date_subset_sample{sample_idx}.npy")

    real_csv_filepath = os.path.join(sample_output_folder, f"weather_real_date_sample{sample_idx}.csv")
    synth_csv_filepath = os.path.join(sample_output_folder, f"weather_synthetic_date_sample{sample_idx}.csv")

    # Save the real and synthetic DataFrames as separate CSV files
    real_df.to_csv(real_csv_filepath, index=False)
    synth_df.to_csv(synth_csv_filepath, index=False)

    # Save processed_data as a .npy file
    np.save(processed_data_filepath, processed_data.numpy())

    # Save synth_data as a .npy file
    np.save(synth_data_filepath, synth_data.cpu().numpy())

    print(f"Sample {sample_idx} saved successfully!")

##############################################################################################################################
print(f"All {num_samples} samples have been generated and saved.")

