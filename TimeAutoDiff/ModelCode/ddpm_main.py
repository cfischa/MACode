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
n_epochs = 50000
#n_epochs = 5000
#n_epochs = 500
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
n_epochs = 50000
#n_epochs = 5000
#n_epochs = 500

num_classes = len(latent_features)

diff = tdf.train_diffusion(latent_features, time, hidden_dim, num_layers, diffusion_steps, n_epochs)

##############################################################################################################################

# Sampling new data
latent_features = ds[1]
# increase T to generate longer samples (temporal depth)
T = latent_features.shape[1]
time_duration = []
# modify N to generate more samples than in latent features (what I want)
N, _, _ = latent_features.shape
# which t_grid startingpoint for sampling is defined
t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device)

#######idea to handle staring point t and length of generated samples
# Define the start date
'''start_date = datetime(2024, 1, 1, 0)  # Example: January 1, 2024, 00:00
hours_since_start = 0  # Relative to this start date

# Define the period (24 hours for hourly data, for example)
period = 24

# Encode the start date using sine and cosine
sin_encoding = np.sin(hours_since_start / (period * 2 * np.pi))
cos_encoding = np.cos(hours_since_start / (period * 2 * np.pi))

print(f"Sine: {sin_encoding}, Cosine: {cos_encoding}")

+ integration into sample()
'''

#samples = tdf.sample(t_grid.repeat(N, 1, 1), latent_features.detach().float().to(device), diff, time)
# Sampling new data
samples = tdf.sample(
    t_grid.repeat(N, 1, 1),
    latent_features.detach().float().to(device),
    diff,
    time,
    connect_sequences=True  # Enable sequence connection
)



# Placeholder for all generated samples
all_samples = []


#to sample faster
'''
batch_size = 500
for i in range(0, N, batch_size):
    # Generate samples for the current batch
    batch_samples = tdf.sample(
        t_grid.repeat(batch_size, 1, 1),  # Time grid for the batch
        latent_features[i:i + batch_size].detach().float().to(device),  # Latent features for this batch
        diff,  # Diffusion model
        time,
        #timestamp_encoding=(sin_encoding, cos_encoding)  # Pass encoded timestamp# Time information
    )
    # Append the batch samples to the list
    all_samples.append(batch_samples)

# Concatenate all batch samples into one tensor
samples = torch.cat(all_samples, dim=0)  # Combine along the batch dimension
'''


##############################################################################################################################
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
# Define file paths
processed_data_filepath = os.path.join(output_folder, 'origi_hellisheidi_weather_cleaned_date_subset.npy')
synth_data_filepath = os.path.join(output_folder, 'synth_hellisheidi_weather_cleaned_date_subset.npy')

# Save the real and synthetic DataFrames as separate CSV files
real_df.to_csv("weather_real_date.csv", index=False)
synth_df.to_csv("weather_synthetic_date.csv", index=False)

# Save processed_data as a .npy file
np.save(processed_data_filepath, processed_data.numpy())

# Save synth_data as a .npy file
np.save(synth_data_filepath, synth_data.cpu().numpy())

##############################################################################################################################
# Plotting
fig, axes = plt.subplots(nrows=1, ncols=18, figsize=(33.1, 23.4 / 5))

for k in range(K):
    axes[k].hist(pd_reshaped[:, k].cpu().detach(), bins=50, color='blue', alpha=0.5, label='Real')
    axes[k].hist(sd_reshaped[:, k].cpu().detach(), bins=50, color='red', alpha=0.5, label='Synthetic')
    axes[k].legend()
    axes[k].set_title(f'Feature {k}', fontsize=15)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(os.path.join(output_folder, 'energy_data_date_subset.png'), dpi=500)
plt.show()
