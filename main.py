import logging
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from data_processing import HORIZON, LOOKBACK, inverse_scale_target, prepare_dataset
from dlinear_model import SeriesDecomp, evaluate_dlinear, train_dlinear
from lstm_model import evaluate_lstm, train_lstm
from transformer_model import evaluate_transformer, train_transformer
from visualization import (
    plot_error_by_horizon,
    plot_metrics_and_time,
    plot_predictions_combined,
)

DATA_PATH = "ETTh1.csv"

dataset_base_name = os.path.splitext(os.path.basename(DATA_PATH))[0].lower()
EXPERIMENT_PREFIX = f"{dataset_base_name}_{LOOKBACK}_{HORIZON}"

LSTM_PATH = f"best_lstm_{EXPERIMENT_PREFIX}.keras"
TRANSFORMER_PATH = f"best_transformer_{EXPERIMENT_PREFIX}.keras"
DLINEAR_PATH = f"best_dlinear_{EXPERIMENT_PREFIX}.keras"

DEFAULT_TIMES = (
    [2469.4, 5249.2, 56.8]
    if "weather" in dataset_base_name
    else [422.9, 813.8, 27.1]
)

def load_or_train_lstm(X_train, y_train, X_val, y_val, filepath):
  if os.path.exists(filepath):
    print(f"Знайдено збережену LSTM модель: {filepath}")
    model = tf.keras.models.load_model(filepath)
    return model, None, 0.0
  else:
    print(f"Навчання нової LSTM моделі ({filepath})...")
    model, history, train_time = train_lstm(
        X_train, y_train, X_val, y_val, filepath=filepath, epochs=50, batch_size=128
    )
    return model, history, train_time

def load_or_train_transformer(X_train, y_train, X_val, y_val, filepath):
  if os.path.exists(filepath):
    print(f"Знайдено збережену Transformer модель: {filepath}")
    model = tf.keras.models.load_model(filepath)
    return model, None, 0.0
  else:
    print(f"Навчання нової Transformer моделі ({filepath})...")
    model, history, train_time = train_transformer(
        X_train, y_train, X_val, y_val, filepath=filepath, epochs=50, batch_size=128
    )
    return model, history, train_time

def load_or_train_dlinear(X_train, y_train, X_val, y_val, filepath):
  if os.path.exists(filepath):
    print(f"Знайдено збережену DLinear модель: {filepath}")
    model = tf.keras.models.load_model(filepath, custom_objects={"SeriesDecomp": SeriesDecomp})
    return model, None, 0.0
  else:
    print(f"Навчання нової DLinear моделі ({filepath})...")
    model, history, train_time = train_dlinear(
        X_train, y_train, X_val, y_val, filepath=filepath, epochs=50, batch_size=128
    )
    return model, history, train_time

def main():
  print("=" * 65)
  print(f"Запуск експерименту [{dataset_base_name.upper()}] (Lookback: {LOOKBACK}, Horizon: {HORIZON})")
  print("=" * 65)

  data = prepare_dataset(DATA_PATH)
  X_train, y_train = data["X_train"], data["y_train"]
  X_val, y_val = data["X_val"], data["y_val"]
  X_test, y_test = data["X_test"], data["y_test"]
  scaler = data["scaler"]
  target_col = data["target_col"]
  target_idx = data["target_idx"]

  unit = "°C" if ("degC" in target_col or "OT" in target_col) else ""

  print("\n1. LSTM")
  lstm_model, _, lstm_time = load_or_train_lstm(X_train, y_train, X_val, y_val, LSTM_PATH)

  print("\n2. Transformer")
  transformer_model, _, transformer_time = load_or_train_transformer(X_train, y_train, X_val, y_val, TRANSFORMER_PATH)

  print("\n3. DLinear")
  dlinear_model, _, dlinear_time = load_or_train_dlinear(X_train, y_train, X_val, y_val, DLINEAR_PATH)

  print("\nОцінювання моделей на тестовій вибірці...")
  lstm_metrics, lstm_pred, true_values = evaluate_lstm(lstm_model, X_test, y_test, scaler, target_idx)
  transformer_metrics, transformer_pred, _ = evaluate_transformer(transformer_model, X_test, y_test, scaler, target_idx)
  dlinear_metrics, dlinear_pred = evaluate_dlinear(dlinear_model, X_test, y_test, scaler, target_idx)

  print("\n" + "=" * 65)
  print(f"Результати [{dataset_base_name.upper()}] (Target: {target_col}, Горизонт: {HORIZON})")
  print("=" * 65)
  print(f"{'Метрика':<20} {'LSTM':<15} {'Transformer':<15} {'DLinear'}")
  print("-" * 65)
  print(f"{f'MAE ({unit})':<20} {lstm_metrics['mae']:<15.4f} {transformer_metrics['mae']:<15.4f} {dlinear_metrics['mae']:.4f}")
  print(f"{f'RMSE ({unit})':<20} {lstm_metrics['rmse']:<15.4f} {transformer_metrics['rmse']:<15.4f} {dlinear_metrics['rmse']:.4f}")
  print(f"{'MAPE (%)':<20} {lstm_metrics['mape']:<14.2f}% {transformer_metrics['mape']:<14.2f}% {dlinear_metrics['mape']:.2f}%")
  print(f"{'Час навч. (с)':<20} {lstm_time:<15.2f} {transformer_time:<15.2f} {dlinear_time:.2f}")
  print("-" * 65)

  times = {
      "lstm": lstm_time,
      "transformer": transformer_time,
      "dlinear": dlinear_time,
  }

  lstm_pred_all = inverse_scale_target(lstm_model.predict(X_test, verbose=0), scaler, target_idx)
  transf_pred_all = inverse_scale_target(transformer_model.predict(X_test, verbose=0), scaler, target_idx)
  dlinear_pred_all = inverse_scale_target(dlinear_model.predict(X_test, verbose=0), scaler, target_idx)
  y_test_orig = inverse_scale_target(y_test, scaler, target_idx)

  pred_file = f"predictions_combined_{EXPERIMENT_PREFIX}.png"
  error_file = f"error_by_horizon_{EXPERIMENT_PREFIX}.png"
  metrics_file = f"metrics_and_time_{EXPERIMENT_PREFIX}.png"

  print("\nГенерація адаптованих графіків...")

  plot_predictions_combined(
      true_values, lstm_pred, transformer_pred, dlinear_pred,
      dataset_name=dataset_base_name.upper(), target_name=target_col, unit=unit, save_path=pred_file,
  )

  plot_error_by_horizon(
      y_test_orig, lstm_pred_all, transf_pred_all, dlinear_pred_all,
      dataset_name=dataset_base_name.upper(), unit=unit, save_path=error_file,
  )

  plot_metrics_and_time(
      lstm_metrics, transformer_metrics, dlinear_metrics, times,
      default_times=DEFAULT_TIMES, unit=unit, save_path=metrics_file,
  )

  print(f"Збережено файли:\n - {pred_file}\n - {error_file}\n - {metrics_file}")

if __name__ == "__main__":
  main()