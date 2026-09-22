import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 36
HORIZON = 96

def prepare_dataset(data_path, target_col=None, lookback=LOOKBACK, horizon=HORIZON):
  if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл '{data_path}' не знайдено!")

  df = pd.read_csv(data_path)

  if target_col is None:
    if "T (degC)" in df.columns:
      target_col = "T (degC)"
    elif "OT" in df.columns:
      target_col = "OT"
    else:
      target_col = df.columns[-1]

  print(f"Обрано цільову колонку: '{target_col}'")

  feature_cols = [c for c in df.columns if c.lower() not in ["date", "time", "timestamp"]]

  if target_col not in feature_cols:
    raise ValueError(f"Колонку '{target_col}' не знайдено серед числових ознак.")

  df[feature_cols] = df[feature_cols].replace(-9999.0, np.nan)
  df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")

  target_idx = feature_cols.index(target_col)
  data_values = df[feature_cols].values

  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data_values)

  n = len(scaled_data)
  train_data = scaled_data[: int(n * 0.7)]
  val_data = scaled_data[int(n * 0.7) : int(n * 0.8)]
  test_data = scaled_data[int(n * 0.8) :]

  def create_sequences(data, lookback, horizon, target_idx):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
      X.append(data[i : i + lookback])
      y.append(data[i + lookback : i + lookback + horizon, target_idx])
    return np.array(X), np.array(y)

  X_train, y_train = create_sequences(train_data, lookback, horizon, target_idx)
  X_val, y_val = create_sequences(val_data, lookback, horizon, target_idx)
  X_test, y_test = create_sequences(test_data, lookback, horizon, target_idx)

  return {
      "X_train": X_train,
      "y_train": y_train,
      "X_val": X_val,
      "y_val": y_val,
      "X_test": X_test,
      "y_test": y_test,
      "scaler": scaler,
      "target_col": target_col,
      "target_idx": target_idx,
  }

def inverse_scale_target(y_scaled, scaler, target_idx=0):
  shape = y_scaled.shape
  y_flat = y_scaled.flatten()

  dummy = np.zeros((len(y_flat), scaler.scale_.shape[0]))
  dummy[:, target_idx] = y_flat
  y_orig = scaler.inverse_transform(dummy)[:, target_idx]

  return y_orig.reshape(shape)