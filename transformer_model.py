import time
from data_processing import HORIZON, LOOKBACK
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def positional_encoding(length, depth):
  positions = np.arange(length)[:, np.newaxis]
  dims = np.arange(depth)[np.newaxis, :]
  angles = positions / np.power(10000, (2 * (dims // 2)) / depth)
  angles[:, 0::2] = np.sin(angles[:, 0::2])
  angles[:, 1::2] = np.cos(angles[:, 1::2])
  return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate):
  attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1] // num_heads)(x, x)
  attention_output = layers.Dropout(dropout_rate)(attention_output)
  x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

  ff_output = layers.Dense(ff_dim, activation="relu")(x)
  ff_output = layers.Dense(x.shape[-1])(ff_output)
  ff_output = layers.Dropout(dropout_rate)(ff_output)
  x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
  return x

def build_transformer(num_features, d_model=64, num_heads=4, ff_dim=128, num_blocks=2, dropout_rate=0.1):
  inputs = keras.Input(shape=(LOOKBACK, num_features))
  x = layers.Dense(d_model)(inputs)
  x = x + positional_encoding(LOOKBACK, d_model)
  x = layers.Dropout(dropout_rate)(x)

  for _ in range(num_blocks):
    x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(64, activation="relu")(x)
  x = layers.Dropout(dropout_rate)(x)
  outputs = layers.Dense(HORIZON)(x)

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

def train_transformer(X_train, y_train, X_val, y_val, filepath, epochs=50, batch_size=256):
  num_features = X_train.shape[2]
  model = build_transformer(num_features=num_features)

  print("Архітектура Transformer:")
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

  print("\nНавчання завершено.")
  best_epoch = np.argmin(history.history["val_loss"]) + 1
  best_val_loss = min(history.history["val_loss"])
  print(f"Найкраща епоха: {best_epoch}, val_loss: {best_val_loss:.6f}")

  return model, history, training_time

def evaluate_transformer(model, X_test, y_test, scaler, target_idx):
  from data_processing import inverse_scale_target

  predictions_scaled = model.predict(X_test, verbose=0)

  pred_original = inverse_scale_target(predictions_scaled[:, 0], scaler, target_idx)
  true_original = inverse_scale_target(y_test[:, 0], scaler, target_idx)

  mae = np.mean(np.abs(pred_original - true_original))
  rmse = np.sqrt(np.mean((pred_original - true_original) ** 2))
  mask = np.abs(true_original) > 1.0
  mape = np.mean(np.abs((true_original[mask] - pred_original[mask]) / true_original[mask])) * 100

  print("\n--- Результати Transformer на тестових даних ---")
  print(f"MAE:  {mae:.4f} градусів")
  print(f"RMSE: {rmse:.4f} градусів")
  print(f"MAPE: {mape:.2f}%")

  return {"mae": mae, "rmse": rmse, "mape": mape}, pred_original, true_original