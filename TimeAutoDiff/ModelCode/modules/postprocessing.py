# modules/postprocessing.py
import os
import torch
import numpy as np
import pandas as pd
import ModelCode.process_edited as pce  # Assuming pce contains the required conversion functions

def postprocess_generated_data(real_df1, gen_output, latent_features, threshold, output_folder):
    """
    Post-process the generated data, convert it to the original structure, and save it.

    Parameters:
    - real_df1 (DataFrame): The original DataFrame without the 'date' column.
    - gen_output (Tensor): The generated output from the decoder.
    - latent_features (Tensor): Latent features from the model.
    - threshold (float): Threshold used during processing.
    - output_folder (str): Path to save processed and synthetic data.

    Returns:
    - synth_data (Tensor): The processed synthetic data.
    """
    # Dimensions
    data_size, seq_len, _ = latent_features.shape

    # Convert generated data to tensor and table
    synth_data = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)
    synth_table = pce.convert_to_table(real_df1, synth_data, threshold)

    # Save processed and synthetic data
    processed_data_filepath = os.path.join(output_folder, 'origi_app_test.npy')
    synth_data_filepath = os.path.join(output_folder, 'synth_app_test.npy')
    np.save(processed_data_filepath, synth_data.numpy())
    np.save(synth_data_filepath, synth_data.cpu().numpy())

    # Save real and synthetic data as CSV
    real_table = pd.DataFrame(real_df1.values)
    synth_table_df = pd.DataFrame(synth_table.numpy())
    real_table.to_csv(os.path.join(output_folder, "test_challenging_patterns_real_date.csv"), index=False)
    synth_table_df.to_csv(os.path.join(output_folder, "test_challenging_patterns_synthetic_date.csv"), index=False)

    return synth_data
