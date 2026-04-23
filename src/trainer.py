from torch.utils.data import DataLoader
import torch


class Trainer:

    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []


    def train_one_epoch(self, dataloader:DataLoader) -> float:
        total_loss = 0
        self.model.train()

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss/len(dataloader)   #avg loss returned


    def dev_one_epoch(self, dataloader:DataLoader) -> float:
        total_loss=0
        self.model.eval()

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                val_predictions = self.model(x_batch)
                val_loss = self.criterion(val_predictions, y_batch)
                total_loss += val_loss.item()

        return total_loss/len(dataloader) #avg loss returned