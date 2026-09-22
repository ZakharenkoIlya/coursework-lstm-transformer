import time
from data_processing import HORIZON, LOOKBACK
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="dlinear_model")
class SeriesDecomp(layers.Layer):
  def __init__(self, kernel_size=25, **kwargs):
    super().__init__(**kwargs)
    self.kernel_size = kernel_size
    self.avg = layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding="same")

  def call(self, x):
    trend = self.avg(x)
    res = x - trend
    return res, trend

  def get_config(self):
    config = super().get_config()
    config.update({"kernel_size": self.kernel_size})
    return config

def build_dlinear(num_features):
  inputs = keras.Input(shape=(LOOKBACK, num_features))
  decomp = SeriesDecomp(kernel_size=25)
  seasonal, trend = decomp(inputs)

  s_flat = layers.Flatten()(seasonal)
  t_flat = layers.Flatten()(trend)

  s_out = layers.Dense(HORIZON)(s_flat)
  t_out = layers.Dense(HORIZON)(t_flat)

  outputs = s_out + t_out

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
  return model

def get_callbacks(filepath, patience=5):
  early_stopping = keras.callbacks.EarlyStopping(
      monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
  )
  checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=filepath, monitor="val_loss", save_best_only=True, verbose=0
  )
  return [early_stopping, checkpoint]

def train_dlinear(X_train, y_train, X_val, y_val, filepath, epochs=50, batch_size=256):
  num_features = X_train.shape[2]
  model = build_dlinear(num_features=num_features)

  print("Архітектура DLinear:")
  model.summary()
  print()

  start_time = time.time()
  history = model.fit(
      X_train, y_train,
      validation_data=(X_val, y_val),
      epochs=epochs,
      batch_size=batch_size,
      callbacks=get_callbacks(filepath),
      verbose=1,
  )
  training_time = time.time() - start_time

  best_epoch = np.argmin(history.history["val_loss"]) + 1
  best_val_loss = min(history.history["val_loss"])
  print(f"Найкраща епоха: {best_epoch}, val_loss: {best_val_loss:.6f}")

  return model, history, training_time

def evaluate_dlinear(model, X_test, y_test, scaler, target_idx):
  from data_processing import inverse_scale_target

  predictions_scaled = model.predict(X_test, verbose=0)

  pred_original = inverse_scale_target(predictions_scaled[:, 0], scaler, target_idx)
  true_original = inverse_scale_target(y_test[:, 0], scaler, target_idx)

  mae = np.mean(np.abs(pred_original - true_original))
  rmse = np.sqrt(np.mean((pred_original - true_original) ** 2))
  mask = np.abs(true_original) > 1.0
  mape = np.mean(np.abs((true_original[mask] - pred_original[mask]) / true_original[mask])) * 100

  print("\n--- Результати DLinear на тестових даних ---")
  print(f"MAE:  {mae:.4f} градусів")
  print(f"RMSE: {rmse:.4f} градусів")
  print(f"MAPE: {mape:.2f}%")

  return {"mae": mae, "rmse": rmse, "mape": mape}, pred_original