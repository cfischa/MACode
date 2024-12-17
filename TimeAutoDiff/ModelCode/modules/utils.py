# modules/utils.py
import torch


def scale_data(data, scaler):
    return scaler.transform(data)


def inverse_scale_data(data, scaler):
    return scaler.inverse_transform(data)
