import numpy as np

def create_windows(data, window_size, target_idx):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i : i+window_size])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y).reshape(-1, 1)