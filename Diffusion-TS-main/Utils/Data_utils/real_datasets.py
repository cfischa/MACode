import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask

class CustomDataset(Dataset):
    def __init__(
            self,
            name,
            data_root,
            window=24,
            proportion=0.8,
            save2npy=True,
            neg_one_to_one=True,
            seed=None,
            period='train',
            output_dir='./OUTPUT',
            predict_length=None,
            missing_ratio=None,
            style='separate',
            distribution='geometric',
            mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'Period must be train or test.'
        if period == 'train':
            assert not (predict_length is not None or missing_ratio is not None), \
                'Predict length or missing ratio should not be defined during training.'

        self.name = name
        self.pred_len = predict_length
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length

        # Use the global seed or default to a fixed value for deterministic behavior
        self.seed = seed if seed is not None else 12345
        # Load raw data and scaler
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.window = window
        self.period = period
        self.len = self.rawdata.shape[0]
        self.var_num = self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.auto_norm = neg_one_to_one
        self.save2npy = save2npy
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        # Normalize raw data
        self.data = self.__normalize(self.rawdata)

        # Generate and split samples
        train, inference = self.__getsamples(self.data, proportion, self.seed)
        self.samples = train if period == 'train' else inference

        # Handle masking for testing
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(self.seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError("Masking requires missing_ratio or predict_length.")

        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        # Calculate the number of non-overlapping sequences
        step = self.window  # Use seq_len (self.window) as the step size
        num_sequences = (self.len - self.window) // step + 1  # Total sequences that fit the data

        # Initialize the sequence array
        x = np.zeros((num_sequences, self.window, self.var_num))

        # Extract non-overlapping sequences
        for i in range(num_sequences):
            start = i * step
            end = start + self.window
            x[i, :, :] = data[start:end, :]  # Slice data without overlap

        # Save full unnormalized data for debugging
        np.save(os.path.join(self.dir, f"{self.name}_full_unnormalized_data.npy"), x)

        # Split data into training and test sets
        train_data, test_data = self.divide(x, proportion, seed)

        # Debugging outputs
        print(f"Generated {num_sequences} non-overlapping sequences.")
        print("Training set shape:", train_data.shape)
        print("Testing set shape:", test_data.shape)

        return train_data, test_data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def divide(self, data, ratio, seed=None):
        size = data.shape[0]
        np.random.seed(seed if seed is not None else self.seed)
        split_idx = int(np.ceil(size * ratio))
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv file."""
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler.fit(data)
        return data, scaler

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.period == 'test':
            m = self.masking[ind, :, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
