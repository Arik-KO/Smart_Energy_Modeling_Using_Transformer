import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import create_windows
from config import *
import joblib
from src.dataset import EnergyModelData
from torch.utils.data import DataLoader
from src.model import Transformer



def main():
    torch.manual_seed(RANDOM_SEED)

    X_train = np.load('data/processed/X_train.npy')
    _,_, D_INPUT = X_train.shape
    X_val =np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')

    # print(len(X_train))

    train_dataset = EnergyModelData(X_train, y_train)
    val_dataset = EnergyModelData(X_val, y_val)
    test_dataset = EnergyModelData(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0 )
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0 )
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0 )

    # X_batch, y_batch = next(iter(train_loader))
    # print(f"\nOne batch: X={X_batch.shape}, y={y_batch.shape}")

    model = Transformer(
        D_INPUT,
        EMBEDDING_DIM,
        HEADS,
        D_FF,
        ENCODER_LAYERS,
        DROPOUT,
        WINDOW_SIZE
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

if __name__ == "__main__":
    main()