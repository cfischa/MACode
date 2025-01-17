import os
import pickle


import numpy as np
import torch

from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def normalize(data):
    
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    data = data / (max_val + 1e-7)
    
    data = data.astype(np.float32)
    
    return data
#################################################

class Sine_Pytorch(torch.utils.data.Dataset):
    
    def __init__(self, no_samples, seq_len, features):
        
        self.data = []
        
        for i in range(no_samples):
            
            temp = []
            
            for k in range(features):
                
                freq = np.random.uniform(0, 0.1)
                
                phase = np.random.uniform(0, 0.1)
                
                temp_data = [np.sin(freq*j + phase) for j in range(seq_len)]
                
                temp.append(temp_data)
                
            temp = np.transpose(np.asarray(temp))
            
            temp = (temp + 1) * 0.5
            
            self.data.append(temp)
        
        self.data = np.asarray(self.data, dtype = np.float32)
        
    def __len__(self):
        
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        return self.data[idx, :, :]
#################################################    
# here you can input your own data
def data_preprocess(dataset_name):
    
    data_dir = f'data'
    
    if dataset_name == 'air':
        
        data = pd.read_csv(f'{data_dir}/AirQualityUCI.csv', delimiter= ';', decimal = ',')
        
        # Last 114 rows does not contain any values
        
        data = data.iloc[:-114, 2:15]
        
    elif dataset_name == 'energy':
        
        data = pd.read_csv(f'{data_dir}/energy_data.csv')
        
        data = data.iloc[:, 0:]
        
    elif dataset_name == 'stock':
        
        data = pd.read_csv(f'{data_dir}/GOOG.csv')
        
        data = data.iloc[:, 1:]

    elif dataset_name == 'sinecurve_and_number_seq':

        data = pd.read_csv(f'{data_dir}/sinecurve_and_number_seq.csv')

        data = data.iloc[:, 1:]

    elif dataset_name == 'test_dataset_960':

        data = pd.read_csv(f'{data_dir}/test_dataset_960.csv')

        data = data.iloc[:, 0:]

    elif dataset_name == 'test_dataset_50016_0.2':

        data = pd.read_csv(f'{data_dir}/test_dataset_50016_0.2.csv')

        data = data.iloc[:, 0:]

    elif dataset_name == 'test_challenging_patterns_6a':

        data = pd.read_csv(f'{data_dir}/test_challenging_patterns_6a.csv')

        data = data.iloc[:, 0:]

    elif dataset_name == 'hellisheidi_weather_cleaned_final':

        data = pd.read_csv(f'{data_dir}/hellisheidi_weather_cleaned_final.csv')

        data = data.iloc[:, 0:]

    return data


class MakeDATA(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        # Convert input data to a NumPy array if not already
        data = np.asarray(data, dtype=np.float32)

        print("Original data (first 5 rows):\n", data[:5])
        norm_data = normalize(data)
        print("Normalized data (first 5 rows):\n", norm_data[:5])



        # Split the normalized data into non-overlapping sequences
        num_sequences = len(norm_data) // seq_len  # Determine how many full sequences can be created
        trimmed_data = norm_data[:num_sequences * seq_len]  # Trim data to fit into full sequences
        seq_data = trimmed_data.reshape(num_sequences, seq_len,
                                        -1)  # Reshape into 3D array (num_sequences, seq_len, features)

        # Store the sequences
        self.samples = np.asarray(seq_data, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    
def LoadData(dataset_name, seq_len):

    if dataset_name == 'sine':
        # this generates sine data set as an input
        data = Sine_Pytorch(5000, seq_len, 5)

        # Convert the dataset into NumPy or Tensor (depending on what you need)
        data_array = np.array([data[i] for i in range(len(data))])

        # Save the data
        save_dir = '/home/sc.uni-leipzig.de/ys09emuw/MACode/TransFusion-main/saved_files'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'origi_normalized_data.npy'), data_array)

        # Split data into train and test
        #train_data, test_data = train_test_split(data_array, train_size=0.8, random_state=2021)
        train_data, test_data = train_test_split(data_array, train_size=0.8, shuffle=False)

        print(f'Sine data loaded with sequence {seq_len}')

    else:
        # Load and preprocess external data
        data = data_preprocess(dataset_name)

        # Create a dataset object
        dataset = MakeDATA(data, seq_len)

        # Convert the dataset into a NumPy array from 'self.samples'
        data_array = dataset.samples

        # Save the normalized data in the desired location
        save_dir = '/home/sc.uni-leipzig.de/ys09emuw/MACode/TransFusion-main/saved_files'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'origi_normalized_data.npy'), data_array)

        # Split data into train and test
        #train_data, test_data = train_test_split(data_array, train_size=0.8, random_state=2021)
        train_data, test_data = train_test_split(data_array, train_size=0.8, shuffle=False)

        print(f'{dataset_name} data loaded with sequence {seq_len}')




    return train_data, test_data


class Sine_Pytorch(torch.utils.data.Dataset):
    def __init__(self, no_samples, seq_len, features):
        self.data = []
        for i in range(no_samples):
            temp = []
            for k in range(features):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)
            temp = np.transpose(np.asarray(temp))
            temp = (temp + 1) * 0.5
            self.data.append(temp)
        self.data = np.asarray(self.data, dtype=np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :]
