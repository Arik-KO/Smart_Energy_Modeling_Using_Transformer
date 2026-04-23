import numpy as np
import torch.nn as nn
from config import *
from src.dataset import EnergyModelData
from torch.utils.data import DataLoader
from src.model import Transformer
from src.trainer import Trainer
from utils.helper import plot_losses, save_model

def main():
    torch.manual_seed(RANDOM_SEED)

    X_train = np.load('data/processed/X_train.npy')
    _,_, D_INPUT = X_train.shape
    X_val =np.load('data/processed/X_val.npy')

    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')


    # print(len(X_train))

    train_dataset = EnergyModelData(X_train, y_train)
    val_dataset = EnergyModelData(X_val, y_val)


    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0 )
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0 )


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
    trainer = Trainer(model, criterion, optimizer, DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nModel parameters: {total_params:,}")
    # print(model)

    for epoch in range(EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        trainer.train_losses.append(train_loss)

        val_loss = trainer.dev_one_epoch(val_loader)
        trainer.val_losses.append(val_loss)
        print(f"for epoch {epoch}, the training loss is: {train_loss:.4f}, and the validation loss is: {val_loss:.4f}")

    plot_losses(trainer.train_losses, trainer.val_losses, "Transformer_with_20epoch")
    save_model(model, 'results/models/transformerv1.pth')



if __name__ == "__main__":
    main()