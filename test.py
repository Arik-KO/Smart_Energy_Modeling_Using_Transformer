import numpy as np
import pandas as pd
from config import *
from src.dataset import EnergyModelData
from torch.utils.data import DataLoader
from src.model import Transformer
import os
from utils.helper import  load_model, plot_predictions
import joblib


def main():
    torch.manual_seed(RANDOM_SEED)
    y_test = np.load('data/processed/y_test.npy')
    X_test = np.load('data/processed/X_test.npy')
    test_dataset = EnergyModelData(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    _,_, D_INPUT = X_test.shape
    scaler = joblib.load('data/processed/scaler.pkl')
    target_idx = 1
    target_min = scaler.data_min_[target_idx]
    target_range = scaler.data_range_[target_idx]


    model = Transformer(
        D_INPUT,
        EMBEDDING_DIM,
        HEADS,
        D_FF,
        ENCODER_LAYERS,
        DROPOUT,
        WINDOW_SIZE
    )
    model = load_model(model, 'results/models/transformerv1.pth')
    model = model.to(DEVICE)
    model.eval()
    all_pred = []
    true_pred = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            test_pred = model(x_batch)
            all_pred.append(test_pred.cpu().numpy().flatten())
            true_pred.append(y_batch.cpu().numpy().flatten())

    y_hat = np.concatenate(all_pred)
    gnd_truth = np.concatenate(true_pred)
    y_hat = y_hat * target_range + target_min
    gnd_truth = gnd_truth * target_range + target_min

    mse = np.mean( (y_hat - gnd_truth) ** 2 )
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_hat - gnd_truth))
    ss_res = np.sum((gnd_truth - y_hat) ** 2)
    ss_tot = np.sum((gnd_truth - np.mean(gnd_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"mean squared error: {mse}")
    print(f"root mean squared error: {rmse}")
    print(f"mean absolute error: {mae}")
    print(f"R2 score: {r2}")



    result_row = {
        'model_name':'transformerv1',
        'Epochs' : EPOCHS,
        'MSE' : round(mse, 4),
        'RMSE' : round(rmse, 4),
        'MAE' : round(mae, 4),
        'Optimizer': 'Adam',
        'Batch_size': BATCH_SIZE,
        'Random_Seed': RANDOM_SEED,
        'R2_Score': round(r2, 4),
        'Learning Rate': LEARNING_RATE
    }

    result_df = pd.DataFrame([result_row])
    log_path = 'results/logs/experiment_log.csv'

    if os.path.exists(log_path):
        result_df.to_csv(log_path, mode = 'a', header = False, index = False )
    else:
        result_df.to_csv(log_path, mode = 'w', header = True, index = False)


    plot_predictions(gnd_truth, y_hat,  "december_predictions1")

if __name__ == "__main__":
    main()