# modules/data_processing.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import FunctionTransformer
import ModelCode.process_edited as pce


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def cyclical_encode(df, year_period=3, month_period=12, day_period=365, hour_period=24):
    df = df.copy()
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    time = df.index

    if year_period is not None:
        df['year_sin'] = sin_transformer(year_period).fit_transform(time.year)
        df['year_cos'] = cos_transformer(year_period).fit_transform(time.year)

    if month_period is not None:
        df['month_sin'] = sin_transformer(month_period).fit_transform(time.month)
        df['month_cos'] = cos_transformer(month_period).fit_transform(time.month)

    if day_period is not None:
        df['day_sin'] = sin_transformer(day_period).fit_transform(time.day_of_year)
        df['day_cos'] = cos_transformer(day_period).fit_transform(time.day_of_year)

    if hour_period is not None:
        df['hour_sin'] = sin_transformer(hour_period).fit_transform(time.hour)
        df['hour_cos'] = cos_transformer(hour_period).fit_transform(time.hour)

    return df


def split_data(real_df, seq_len, threshold):
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data_tensor = torch.tensor(data.astype('float32'))

    num_sequences = len(data_tensor) // seq_len
    data_tensor = data_tensor[:num_sequences * seq_len]
    data_tensor = data_tensor.reshape(num_sequences, seq_len, -1).contiguous()

    return data_tensor


def split_time_data(real_df, seq_len):
    df = cyclical_encode(real_df)
    columns = ['year_sin', 'year_cos', 'month_sin', 'month_cos',
               'day_sin', 'day_cos', 'hour_sin', 'hour_cos']

    if not all(col in df.columns for col in columns):
        raise ValueError(f"Missing columns in cyclical encoding: {columns}")

    time_tensor = torch.tensor(df[columns].values).float()
    num_sequences = len(time_tensor) // seq_len
    time_tensor = time_tensor[:num_sequences * seq_len].reshape(num_sequences, seq_len, -1).contiguous()

    return time_tensor
