import streamlit as st
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# Import your modules
import modules.data_processing as dp
import modules.autoencoder_training as at
import modules.diffusion_training as dt
import modules.sampling as sp
import modules.postprocessing as pp
import ModelCode.process_edited as pce

# Set device
device_option = st.sidebar.selectbox("Select Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
device = torch.device(device_option)
st.sidebar.write(f"Using device: {device}")

# Define session states for each step
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "time_info" not in st.session_state:
    st.session_state.time_info = None
if "autoencoder_model" not in st.session_state:
    st.session_state.autoencoder_model = None
if "diffusion_model" not in st.session_state:
    st.session_state.diffusion_model = None
if "latent_features" not in st.session_state:
    st.session_state.latent_features = None
if "samples" not in st.session_state:
    st.session_state.samples = None
if "synth_data_list" not in st.session_state:
    st.session_state.synth_data_list = []  # Store synthetic data from each iteration

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the base directory
base_dir = os.path.dirname(current_dir)
# Define the output folder relative to the base directory
output_folder = os.path.join(base_dir, 'TimeAutoDiff', 'output')


# Step 1: Data Upload and Preprocessing
st.header("1. Data Upload and Preprocessing")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    real_df = pd.read_csv(uploaded_file)
    real_df1 = real_df.drop('date', axis=1)
    real_df2 = real_df  # Keeping original

    # Store in session state
    st.session_state.real_df1 = real_df1
    st.session_state.real_df2 = real_df2
    st.session_state.real_df = real_df
    st.success("Data loaded successfully!")

    # Preprocessing parameters
    seq_len = st.number_input("Sequence Length (Number of Data Points per Series)", min_value=1, value=24)
    threshold = st.slider("Threshold", min_value=0.0, max_value=10.0, value=1.0)

    if st.button("Process Data"):
        st.session_state.processed_data = dp.split_data(
            st.session_state.real_df1, seq_len, threshold
        ).float()
        st.session_state.time_info = dp.split_time_data(
            st.session_state.real_df2, st.session_state.processed_data.shape[1]
        ).to(device).float()
        st.success("Data processed successfully!")
        st.write(f"Processed Data Shape: {st.session_state.processed_data.shape}")
        st.write(f"Time Info Shape: {st.session_state.time_info.shape}")


# Step 2: Autoencoder Training
if st.session_state.processed_data is not None:
    st.header("2. Autoencoder Training")
    st.write("Configure the autoencoder training parameters:")

    # Autoencoder parameters
    autoencoder_params = {
        "n_epochs": st.number_input("Autoencoder: Number of Epochs", min_value=1, value=5000),
        "lr": st.number_input("Autoencoder: Learning Rate", min_value=1e-6, value=2e-4, format="%.6f"),
        "batch_size": st.number_input("Autoencoder: Batch Size", min_value=1, value=50),
        "channels": st.number_input("Autoencoder: Channels", min_value=1, value=64),
        "hidden_size": st.number_input("Autoencoder: Hidden Size", min_value=1, value=200),
        "num_layers": st.number_input("Autoencoder: Number of Layers", min_value=1, value=1),
        "threshold": threshold,
        "min_beta": st.number_input("Autoencoder: Min Beta", min_value=1e-6, value=1e-4, format="%.6f"),
        "max_beta": st.number_input("Autoencoder: Max Beta", min_value=1e-6, value=0.02, format="%.6f"),
        "emb_dim": st.number_input("Autoencoder: Embedding Dimension", min_value=1, value=128),
        "time_dim": st.number_input("Autoencoder: Time Dimension", min_value=1, value=8),
        "lat_dim": st.number_input("Autoencoder: Latent Dimension", min_value=1, value=7),
        "weight_decay": st.number_input("Autoencoder: Weight Decay", min_value=0.0, value=1e-6, format="%.7f"),
    }

    # Store autoencoder_params in session state after defining it
    st.session_state.autoencoder_params = autoencoder_params

    if st.button("Train Autoencoder"):
        # Create a progress bar
        autoencoder_progress_bar = st.progress(0)
        st.write("Training Autoencoder...")

        # Call the training function with the progress bar
        st.session_state.autoencoder_model, losses = at.train_autoencoder(
            st.session_state.real_df1,
            st.session_state.processed_data.float(),
            autoencoder_params,
            device,
            progress_bar=autoencoder_progress_bar  # Pass the progress bar
        )

        # Clear the progress bar after training
        autoencoder_progress_bar.empty()

        st.session_state.latent_features = st.session_state.autoencoder_model.encoder(
            st.session_state.processed_data.to(device)
        )[0].detach()
        st.success("Autoencoder training completed!")
        st.line_chart(losses)


# Step 3: Diffusion Model Training
if st.session_state.latent_features is not None:
    st.header("3. Diffusion Model Training")
    st.write("Configure the diffusion model training parameters:")

    # Diffusion model parameters
    diffusion_params = {
        "hidden_dim": st.number_input("Hidden Dimension", min_value=1, value=200),
        "num_layers": st.number_input("Number of Layers", min_value=1, value=2),
        "diffusion_steps": st.number_input("Diffusion Steps", min_value=1, value=200),
        "n_epochs": st.number_input("Number of Epochs", min_value=1, value=5000),
        "batch_size": st.number_input("Batch Size", min_value=1, value=50),
        "lr": st.number_input("Learning Rate", min_value=1e-6, value=1e-4, format="%.6f"),
    }

    if st.button("Train Diffusion Model"):
        # Create a progress bar
        diffusion_progress_bar = st.progress(0)
        st.write("Training Diffusion Model...")

        # Call the training function with the progress bar
        st.session_state.diffusion_model, diffusion_losses = dt.train_diffusion(
            st.session_state.latent_features,
            st.session_state.time_info,
            diffusion_params,
            device,
            progress_bar=diffusion_progress_bar  # Pass the progress bar
        )

        # Clear the progress bar after training
        diffusion_progress_bar.empty()

        st.success("Diffusion model training completed!")
        st.line_chart(diffusion_losses)

# Step 4: Sampling and Post-Processing
if st.session_state.diffusion_model is not None:
    st.header("4. Sampling and Post-Processing")

    # Input for number of samples per iteration and number of iterations
    max_samples = st.session_state.latent_features.shape[0]
    num_samples_per_iteration = st.number_input(
        "Number of Samples per Iteration (Number of Data Series)",
        min_value=1,
        max_value=max_samples,
        value=100
    )
    num_iterations = st.number_input("Number of Sampling Iterations", min_value=1, value=1)
    total_samples = num_samples_per_iteration * num_iterations
    st.write(f"Total Samples to Generate: {total_samples}")

    if st.button("Generate Samples"):
        st.session_state.samples = []  # Initialize as a list
        st.session_state.synth_data_list = []  # Initialize synthetic data list

        # Create a progress bar for iterations
        sampling_progress_bar = st.progress(0)
        st.write("Generating Samples and Post-processing...")

        for i in range(num_iterations):
            # Update the progress bar for iterations
            iteration_progress = (i + 1) / num_iterations
            sampling_progress_bar.progress(iteration_progress)
            t = torch.linspace(0, 1, st.session_state.processed_data.shape[1]).view(1, -1, 1).to(device)
            indices = np.random.choice(max_samples, num_samples_per_iteration, replace=True)
            latent_features = st.session_state.latent_features[indices]
            time_info = st.session_state.time_info[indices]

            samples = sp.sample(
                t.repeat(num_samples_per_iteration, 1, 1),
                latent_features,
                st.session_state.diffusion_model,
                time_info,
                device,
            )
            st.session_state.samples.append(samples)  # Append each iteration's samples
            st.success(f"Samples generated successfully for iteration {i+1}!")
            st.write(f"Generated Samples Shape for iteration {i+1}: {samples.shape}")

            # Post-process and save data immediately after generating samples
            gen_output = st.session_state.autoencoder_model.decoder(samples.float().to(device))
            data_size, seq_len, _ = samples.shape

            synth_data = pce.convert_to_tensor(
                org_df=st.session_state.real_df1,
                gen_output=gen_output,
                threshold=autoencoder_params['threshold'],
                data_size=data_size,
                seq_len=seq_len,
                device=device,
            )
            _synth_data = pce.convert_to_table(
                org_df=st.session_state.real_df1,
                synth_data=synth_data,
                threshold=autoencoder_params['threshold']
            )

            # Store synthetic data for analysis
            st.session_state.synth_data_list.append(synth_data.cpu())

            # Save data to files with iteration index
            processed_data_filepath = os.path.join(
                output_folder, f'origi_test_challenging_patterns_iter_{i+1}.npy')
            synth_data_filepath = os.path.join(
                output_folder, f'synth_test_challenging_patterns_iter_{i+1}.npy')

            np.save(processed_data_filepath, st.session_state.processed_data.numpy())
            np.save(synth_data_filepath, synth_data.cpu().numpy())

            # Save real and synthetic data as CSVs
            real_df = pd.DataFrame(
                st.session_state.processed_data.numpy().reshape(-1, st.session_state.processed_data.shape[2]))
            synth_df = pd.DataFrame(synth_data.numpy().reshape(-1, synth_data.shape[2]))

            real_df.to_csv(f"test_challenging_patterns_real_iter_{i+1}.csv", index=False)
            synth_df.to_csv(f"test_challenging_patterns_synthetic_iter_{i+1}.csv", index=False)

            st.success(f"Post-processing completed and data saved for iteration {i+1}!")
            st.write(f"Processed Data File: {processed_data_filepath}")
            st.write(f"Synthetic Data File: {synth_data_filepath}")

            st.success(f"Iteration {i + 1} completed.")
        # Clear the progress bar after sampling
        sampling_progress_bar.empty()

# Step 5: Analysis
if st.session_state.synth_data_list:
    st.header("5. Analysis")

    # Option to select the number of sequences to plot
    num_sequences_to_plot = st.number_input(
        "Number of Sequences to Plot",
        min_value=1,
        max_value=st.session_state.processed_data.shape[0],
        value=10
    )

    # Option to select plotting mode
    plot_mode = st.selectbox(
        "Plot Option",
        [
            "Plot Real and Synthetic Data Together for Each Feature",
            "Plot Synthetic Data Separately for Each Iteration and Feature"
        ]
    )

    if st.button("Plot Time Series"):
        # Number of features
        num_features = st.session_state.processed_data.shape[2]

        # Extract the selected number of sequences for original data
        orig_data = st.session_state.processed_data[:int(num_sequences_to_plot)].cpu().numpy()
        # Concatenate sequences along the time axis for original data
        orig_data_concat = orig_data.reshape(-1, num_features)

        # For synthetic data, create a list of concatenated data for each iteration
        synth_data_concat_list = []
        num_iterations = len(st.session_state.synth_data_list)
        for synth_data in st.session_state.synth_data_list:
            synth_data_np = synth_data[:int(num_sequences_to_plot)].cpu().numpy()
            synth_data_concat = synth_data_np.reshape(-1, num_features)
            synth_data_concat_list.append(synth_data_concat)

        # Generate colors for the iterations
        cmap_name = 'tab10' if num_iterations <= 10 else 'tab20'
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(num_iterations)]

        # For each feature
        for feature_idx in range(num_features):
            feature_name = f"Feature {feature_idx+1}"

            if plot_mode == "Plot Real and Synthetic Data Together for Each Feature":
                plt.figure(figsize=(12, 6))
                # Plot real data
                plt.plot(orig_data_concat[:, feature_idx], color='blue')
                # Plot synthetic data from all iterations
                for idx, synth_data_concat in enumerate(synth_data_concat_list):
                    plt.plot(
                        synth_data_concat[:, feature_idx],
                        color=colors[idx % len(colors)],
                        alpha=0.6
                    )
                plt.title(f"Time Series for {feature_name} (Real and Synthetic)")
                plt.xlabel("Time")
                plt.ylabel("Value")
                # No legend as per requirement
                st.pyplot(plt)
                plt.close()

            elif plot_mode == "Plot Synthetic Data Separately for Each Iteration and Feature":
                # First, plot the original data
                plt.figure(figsize=(12, 6))
                plt.plot(orig_data_concat[:, feature_idx], color='blue')
                plt.title(f"Original Time Series for {feature_name}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                # No legend as per requirement
                st.pyplot(plt)
                plt.close()

                # Then, plot synthetic data for each iteration
                for idx, synth_data_concat in enumerate(synth_data_concat_list):
                    plt.figure(figsize=(12, 6))
                    # Plot synthetic data for the iteration
                    plt.plot(
                        synth_data_concat[:, feature_idx],
                        color='orange'
                    )
                    plt.title(f"Synthetic Time Series for {feature_name} - Iteration {idx+1}")
                    plt.xlabel("Time")
                    plt.ylabel("Value")
                    # No legend as per requirement
                    st.pyplot(plt)
                    plt.close()

    # Add the "Compare Statistics" button
    if st.button("Compare Statistics"):
        st.subheader("Statistical Comparison of Real Data and Synthetic Iterations")
        stats = []

        # Compute statistics for original data
        orig_data = st.session_state.processed_data.cpu().numpy()
        orig_data_reshaped = orig_data.reshape(-1, orig_data.shape[2])
        orig_min = orig_data_reshaped.min(axis=0)
        orig_max = orig_data_reshaped.max(axis=0)
        orig_mean = orig_data_reshaped.mean(axis=0)
        orig_std = orig_data_reshaped.std(axis=0)
        stats.append({
            'Iteration': 'Original Data',
            'Min': orig_min,
            'Max': orig_max,
            'Mean': orig_mean,
            'Std Dev': orig_std
        })

        # Compute statistics for each synthetic iteration
        for idx, synth_data in enumerate(st.session_state.synth_data_list):
            synth_data_np = synth_data.cpu().numpy()
            synth_data_reshaped = synth_data_np.reshape(-1, synth_data_np.shape[2])
            synth_min = synth_data_reshaped.min(axis=0)
            synth_max = synth_data_reshaped.max(axis=0)
            synth_mean = synth_data_reshaped.mean(axis=0)
            synth_std = synth_data_reshaped.std(axis=0)
            stats.append({
                'Iteration': f'Iteration {idx+1}',
                'Min': synth_min,
                'Max': synth_max,
                'Mean': synth_mean,
                'Std Dev': synth_std
            })

        # Prepare DataFrame for statistics
        num_features = orig_data.shape[2]
        feature_names = [f'Feature {i+1}' for i in range(num_features)]

        # Create a list of DataFrames for each statistic
        stats_tables = []
        for stat_name in ['Min', 'Max', 'Mean', 'Std Dev']:
            data = []
            for entry in stats:
                row = [entry['Iteration']] + entry[stat_name].tolist()
                data.append(row)
            columns = ['Iteration'] + feature_names
            df_stat = pd.DataFrame(data, columns=columns)
            stats_tables.append((stat_name, df_stat))

        # Display each statistics table
        for stat_name, df_stat in stats_tables:
            st.write(f"**{stat_name}**")
            st.dataframe(df_stat.style.format({col: '{:.4f}' for col in feature_names}))






