import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import process_GQ as pce
import process_edited as pce
from datetime import date
from sklearn.preprocessing import FunctionTransformer

################################################################################################################
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# cyclical encoding function
def cyclical_encode(df, year_period=3, month_period=12, day_period=365, hour_period=24):
    # Assuming df datetime follows the following format: 'YYYY-MM-DD HH:MM:SS' with column name 'date'
    res = df.copy()
    res.date = pd.to_datetime(res.date)
    res.set_index('date', inplace=True)
    time = res.index

    # If not using any period then set to False
    if year_period is not None:
        res['year_sin'] = sin_transformer(year_period).fit_transform(time.year)
        res['year_cos'] = cos_transformer(year_period).fit_transform(time.year)

    if month_period is not None:
        res['month_sin'] = sin_transformer(month_period).fit_transform(time.month)
        res['month_cos'] = cos_transformer(month_period).fit_transform(time.month)

    if day_period is not None:
        res['day_sin'] = sin_transformer(day_period).fit_transform(time.day_of_year)
        res['day_cos'] = cos_transformer(day_period).fit_transform(time.day_of_year)

    if hour_period is not None:
        res['hour_sin'] = sin_transformer(hour_period).fit_transform(time.hour)
        res['hour_cos'] = cos_transformer(hour_period).fit_transform(time.hour)

    return res

################################################################################################################
def partition_multi_seq(real_df, threshold, column_to_partition):    
    
    # column_to_partition
    real_df1 = real_df.drop('date', axis=1)
    parser = pce.DataFrameParser().fit(real_df1, threshold)
    processed_data = torch.from_numpy(parser.transform()).unsqueeze(0)
    column_name = parser._column_order
    column_index = column_name.index(column_to_partition)

    # Partition multi-sequence data
    unique_values = np.unique(processed_data[:, :, column_index])

    partitioned_tensors = torch.zeros(len(unique_values), int(len(processed_data[0,:,:])/len(unique_values)), processed_data.shape[2])

    # Partition the tensor based on unique values in the specified column
    i = 0
    for value in unique_values:
        mask = processed_data[:, :, column_index] == value
        partitioned_tensors[i, :, :] = processed_data[mask]
        i = i + 1
    
    # Partition the multi-sequence data's date information
    df2 = cyclical_encode(real_df); 
    partitioned_tensors_ts = torch.zeros(len(unique_values), int(len(processed_data[0,:,:])/len(unique_values)), 8)
    time_info = torch.tensor(df2.iloc[:,-8:].values).unsqueeze(0)
    
    i = 0
    for value in unique_values:
        mask = processed_data[:, :, column_index] == value
        partitioned_tensors_ts[i, :, :] = time_info[mask]
        i = i + 1
    
    # Remove the column at column_index_to_remove
    partitioned_tensors = torch.cat((partitioned_tensors[:, :, :column_index], 
                                   partitioned_tensors[:, :, column_index+1:]), dim=2)

    return (partitioned_tensors, partitioned_tensors_ts)





################################################################################################################
def splitData(real_df, seq_len, threshold, step_size=1):
    """
    Load and preprocess real-world datasets with a sliding-window approach
    (controllable stride).

    Args:
      real_df (pd.DataFrame): The raw input dataframe (non-time columns).
      seq_len (int): Number of time steps in each window (e.g., 24).
      threshold (float): Threshold for DataFrameParser (if applicable).
      step_size (int): Stride between consecutive windows.
                       If 1, you get maximal overlap (original behavior).
                       If > 1, you reduce overlap and memory usage.

    Returns:
      torch.Tensor: A 3D tensor of shape (num_windows, seq_len, num_features).
    """
    # 1) Normalize / threshold the data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()  # shape (N, K)

    # 2) Convert to numpy float32
    ori_data = torch.tensor(data.astype('float32')).numpy()

    # 3) Sliding window
    # Instead of i in range(0, len(ori_data) - seq_len),
    # we go i in steps of `step_size`
    temp_data = []
    for i in range(0, len(ori_data) - seq_len + 1, step_size):
        window = ori_data[i: i + seq_len]
        temp_data.append(window)

    # 4) Convert to a single tensor
    data_tensor = torch.tensor(temp_data)

    return data_tensor


def splitTimeData(real_df, seq_len, step_size=1):
    """
    Load and preprocess time-related features in a sliding-window format
    (controllable stride).

    Args:
      real_df (pd.DataFrame): The raw input dataframe with date/time columns.
      seq_len (int): Number of time steps in each window (e.g., 24).
      step_size (int): Stride between consecutive windows.

    Returns:
      torch.Tensor: A 3D tensor of shape (num_windows, seq_len, 8)
                    (assuming 8 cyclical time columns).
    """
    # 1) Apply cyclical encoding to date/time columns
    df2 = cyclical_encode(real_df)  # Must produce at least 8 cyclical cols
    # e.g. last 8 columns are year_sin, year_cos, month_sin, month_cos, etc.

    # 2) Convert to numpy float32
    time_array = df2.iloc[:, -8:].values  # shape (N, 8)
    time_array = torch.tensor(time_array.astype('float32')).numpy()

    # 3) Sliding window
    temp_data = []
    for i in range(0, len(time_array) - seq_len + 1, step_size):
        window = time_array[i: i + seq_len]
        temp_data.append(window)

    # 4) Convert to tensor
    data_tensor = torch.tensor(temp_data)

    return data_tensor


'''
# latest version (no sliding window)
def splitData(real_df, seq_len, threshold):
    """Load and preprocess real-world datasets without overlapping sequences."""
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    ori_data = torch.tensor(data.astype('float32'))

    # Diagnostic: Print `F1` values before reshaping
    print("Sample of `F1` in splitData before reshaping:")
    print(ori_data[:10, 0])  # Assuming `F1` is the first column

    # Calculate the number of non-overlapping sequences
    num_sequences = len(ori_data) // seq_len

    # Trim to make sure only full sequences are used
    ori_data = ori_data[:num_sequences * seq_len]

    # Reshape and ensure contiguity
    data = ori_data.reshape(num_sequences, seq_len, -1).contiguous()

    return data

################################################################################################################
def splitTimeData(real_df, seq_len):
    """Load and preprocess time-related features in a sequence-compatible format without overlapping sequences."""
    df2 = cyclical_encode(real_df)  # Apply cyclical encoding

    # Print all columns in df2 to confirm that all 8 cyclical encoding columns are present
    print("Columns in df2 after encoding:", df2.columns.tolist())  # Diagnostic

    # Explicitly select the 8 expected cyclical encoding columns
    expected_columns = ['year_sin', 'year_cos', 'month_sin', 'month_cos',
                        'day_sin', 'day_cos', 'hour_sin', 'hour_cos']

    # Ensure these columns are present and select them
    if not all(col in df2.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df2.columns]
        raise ValueError(f"Missing expected cyclical encoding columns: {missing_cols}")

    time_info = torch.tensor(df2[expected_columns].values).float()

    # Calculate the number of full, non-overlapping sequences
    num_sequences = len(time_info) // seq_len

    # Trim and ensure contiguity after reshaping
    time_info = time_info[:num_sequences * seq_len].reshape(num_sequences, seq_len, -1).contiguous()

    #print("Shape of time_info tensor:", time_info.shape)  # Diagnostic output to confirm shape
    return time_info


# orignal 
def splitData(real_df, seq_len, threshold):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    # ori_data = torch.tensor(data.astype('float32')).numpy()
    ori_data = torch.tensor(data.astype('float32')).numpy()

    batch_size = len(ori_data) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, batch_size):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    # idx = np.random.permutation(len(temp_data))
    # data = []
    # for i in range(len(temp_data)):
    #    data.append(temp_data[idx[i]])

    data = torch.tensor(temp_data)

    return data


################################################################################################################
def splitTimeData(real_df, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    df2 = cyclical_encode(real_df);
    tlen = df2.shape[1]
    time_info = torch.tensor(df2.iloc[:, -8:].values).numpy()

    batch_size = len(time_info) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(time_info) - seq_len):
        _x = time_info[i:i + seq_len]
        temp_data.append(_x)

    data = torch.tensor(temp_data)

    return data

def splitData(real_df, seq_len, threshold, step_size=1):
    """
    Load and preprocess real-world feature data using sliding windows.

    Args:
        real_df (pd.DataFrame): The original dataset with numeric features
                                (excluding date/time columns).
        seq_len (int): Length of each sequence window (e.g., 24 for daily).
        threshold (float): Threshold used by DataFrameParser (if applicable).
        step_size (int): The step size (stride) for the sliding window.
                         - For full overlap, use step_size=1.
                         - For half overlap, use step_size=seq_len//2 (etc.).
                         - For no overlap, use step_size=seq_len.

    Returns:
        data (torch.Tensor): Shape (num_windows, seq_len, num_features).
                             Each window is a slice of length `seq_len`.
    """
    # 1) Parse and transform your dataset (normalization, thresholding, etc.)
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()  # NumPy array of shape (N, K)

    # 2) Convert to float32 Tensor
    ori_data = torch.tensor(data.astype('float32'))

    # 3) Collect sliding windows
    windows = []
    # We go up to (len(ori_data) - seq_len + 1) so that i+seq_len does not overflow
    for i in range(0, len(ori_data) - seq_len + 1, step_size):
        window = ori_data[i: i + seq_len]
        windows.append(window)

    # 4) Convert to a single Tensor of shape (num_windows, seq_len, num_features)
    data = torch.stack(windows, dim=0)

    print(f"[splitData] Generated {data.shape[0]} windows, "
          f"each of length {seq_len}, step size = {step_size}.")
    return data


################################################################################################################
def splitTimeData(real_df, seq_len, step_size=1):
    """
    Load and preprocess cyclical time features in a sequence-compatible format
    using sliding windows.

    Args:
        real_df (pd.DataFrame): The original dataset including date/time columns.
        seq_len (int): Length of each sequence window (e.g., 24 for daily).
        step_size (int): Stride for the sliding windows.

    Returns:
        time_info (torch.Tensor): Shape (num_windows, seq_len, 8)
                                  (assuming 8 cyclical time columns).
    """
    # 1) Apply cyclical encoding to your date/time columns
    df2 = cyclical_encode(real_df)  # Must produce 8 columns: year_sin, year_cos, etc.

    # 2) Check for the expected cyclical columns
    expected_columns = [
        'year_sin', 'year_cos', 'month_sin', 'month_cos',
        'day_sin', 'day_cos', 'hour_sin', 'hour_cos'
    ]
    missing_cols = [col for col in expected_columns if col not in df2.columns]
    if missing_cols:
        raise ValueError(f"Missing cyclical encoding columns: {missing_cols}")

    # 3) Select these 8 columns, convert to float32
    time_array = df2[expected_columns].values  # shape (N, 8)
    time_array = torch.tensor(time_array, dtype=torch.float32)

    # 4) Collect sliding windows
    windows = []
    for i in range(0, len(time_array) - seq_len + 1, step_size):
        window = time_array[i: i + seq_len]
        windows.append(window)

    time_info = torch.stack(windows, dim=0)

    print(f"[splitTimeData] Generated {time_info.shape[0]} windows, "
          f"each of length {seq_len}, step size = {step_size}.")
    return time_info
'''



