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
torch.cuda.empty_cache()

# Define the base directory dynamically using the current file's location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
dataset_folder = os.path.join(os.path.dirname(base_dir), 'Dataset', 'Single-Sequence')
output_folder = os.path.join(os.path.dirname(base_dir), 'output')


# Filename for the dataset
filename = os.path.join(dataset_folder, 'test_challenging_patterns_6a_date_final.csv')

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
#processed_data = dp.splitData(real_df1, 24, threshold, step_size=10).float()  # Cast to float32
#time_info = dp.splitTimeData(real_df2, processed_data.shape[1], step_size=10).to(device).float()  # Cast to float32

processed_data = dp.splitData(real_df1, 48, threshold, step_size=48).float()  # Cast to float32
time_info = dp.splitTimeData(real_df2, processed_data.shape[1], step_size=48).to(device).float()  # Cast to float32


##############################################################################################################################
# Auto-encoder Training
#n_epochs = 50000
#n_epochs = 5000
n_epochs = 500
eps = 1e-5
weight_decay = 1e-6
lr = 2e-4
hidden_size = 200
#hidden_size = 100
num_layers = 1
batch_size = 32
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
diffusion_steps = 100
#n_epochs = 50000
#n_epochs = 5000
n_epochs = 500

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

############################################
# 1) Day-by-Day infinite generation
############################################

# Let’s define how many days we want:
num_days_to_generate = 1000  # or 100, or 365, etc.

# We'll store each "day" in a list:
list_of_days = []

# We'll pick the "initial day" from your trained latents as a starting point
# For example, day 0 is the first window in latent_features:
prev_day_latent = latent_features[0:1].to(device)  # shape: (1, seq_len, lat_dim)
day_length = prev_day_latent.shape[1]

# If you have time_info shaped (N, L, 8), then we can slice day by day:
# e.g. time_info[0:1], time_info[1:2], etc.
# But if you only have as many windows as you do days, we need to handle carefully.

for day_idx in range(num_days_to_generate):
    print(f"Generating day {day_idx} ...")

    # 1a) Pick time_info for this day
    # Suppose you have enough rows in time_info, one row per day:
    # shape = (N, L, 8). Then day_idx is the slice [day_idx : day_idx+1]
    # If your dataset has fewer days than num_days_to_generate,
    # you can cycle or clamp. For now let's assume day_idx < time_info.shape[0].
    current_time_info = time_info[day_idx : day_idx+1]  # shape (1, L, 8)

    # 1b) Build t_grid for 'day_length' steps
    t_grid_day = torch.linspace(0, 1, day_length).view(1, -1, 1).to(device)

    # 1c) Sample new day from your diffusion code,
    #     "conditioned" on the final latent from the previous day:
    # If your code has no explicit "init_latent" param,
    # we can inline-edit the existing sample function to skip random init
    # and start from prev_day_latent.
    # For demonstration, let's assume you do:
    new_day_latent = tdf.sample(
        t_grid_day,
        prev_day_latent[:, -1:, :],  # (1, 1, lat_dim) to "seed"
        diff,
        current_time_info,
        connect_sequences=False,  # we'll handle continuity ourselves
        noise_scale=0.01,
        blend_steps=0
    )
    # Now new_day_latent has shape (1, day_length, lat_dim)

    # 1d) Decode the new day
    new_day_output = ds[0].decoder(new_day_latent.float().to(device))
    # ds[0] is your AE model
    # new_day_output might be a dict of 'bins','cats','nums' if that's how your AE works.
    # So we convert it to a final numeric tensor:
    new_day_tensor = pce.convert_to_tensor(
        real_df1,
        new_day_output,
        threshold,
        data_size=1,      # 1 = we have just 1 sequence in the batch
        seq_len=day_length
    )
    # shape: (1, day_length, num_features)

    # 1e) We only want a 2D array, remove the batch dimension:
    new_day_tensor_2d = new_day_tensor.squeeze(0)  # shape (day_length, num_features)

    # Append to the list
    list_of_days.append(new_day_tensor_2d)

    # 1f) Update prev_day_latent for next iteration
    #     either use the entire day, or the final step.
    # Let's use the entire day to carry continuity:
    prev_day_latent = new_day_latent  # shape (1, day_length, lat_dim)

all_synthetic_2d = torch.cat(list_of_days, dim=0)  # shape: (num_days_to_generate * day_length, num_features)

# Convert synthetic data back to the original scale
synth_original_scale = pce.convert_to_table(
    org_df=real_df1,
    synth_data=all_synthetic_2d.unsqueeze(0),  # Add batch dimension: shape (1, total_days * day_length, num_features)
    threshold=threshold
)

# Reshape to 2D and convert to DataFrame
synth_original_scale_2d = synth_original_scale.squeeze(0).cpu().numpy()  # shape: (total_days * day_length, num_features)
df_synth_original = pd.DataFrame(synth_original_scale_2d, columns=real_df1.columns)

# Save to CSV
output_file = "infinite_synth_series_original_scale.csv"
df_synth_original.to_csv(output_file, index=False)
print(f"Saved scaled data to {output_file}")

'''
# Now we have num_days_to_generate days in list_of_days,
# each is shape (day_length, num_features). Let's stitch them:
all_synthetic_2d = torch.cat(list_of_days, dim=0)  # shape => (num_days_to_generate*day_length, num_features)

# Convert to DataFrame, save, etc.
df_synth = pd.DataFrame(all_synthetic_2d.cpu().numpy())
df_synth.to_csv("infinite_synth_series_10_challenge.csv", index=False)
print("Saved infinite time series as 'infinite_synth_series_10.csv'")
'''


