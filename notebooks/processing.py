import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import create_windows
from config import WINDOW_SIZE
import joblib



df = pd.read_csv('../data/processed/energy_features_2019.csv')
feature_columns = df.columns.to_list()
feature_columns.remove("date")
D_INPUT = len(feature_columns)
print(D_INPUT)

train_df = df[df["date"]< "2019-11-01"]
val_df =  df[(df["date"] >= "2019-11-01") & (df["date"] < "2019-12-01")]
test_df = df[df["date"] >= "2019-12-01"]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[feature_columns].values)
val_scaled = scaler.transform(val_df[feature_columns].values)
test_scaled = scaler.transform(test_df[feature_columns].values)

target_idx = feature_columns.index("avg_energy")
target_mean = scaler.data_min_[target_idx]
target_std = scaler.data_range_[target_idx]

X_train, y_train = create_windows(train_scaled, WINDOW_SIZE, target_idx )
X_val, y_val = create_windows(val_scaled, WINDOW_SIZE, target_idx)
X_test, y_test = create_windows(test_scaled, WINDOW_SIZE, target_idx)

np.save('../data/processed/X_train.npy', X_train)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/X_val.npy', X_val)
np.save('../data/processed/y_val.npy', y_val)
np.save('../data/processed/X_test.npy', X_test)
np.save('../data/processed/y_test.npy', y_test)


joblib.dump(scaler, '../data/processed/scaler.pkl')

if __name__ == "__main__":
    # print(type(feature_columns))
    # print(f"Train: {len(train_df)} hours | Val: {len(val_df)} hours | Test: {len(test_df)} hours")
    print(f"\nAfter windowing:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")