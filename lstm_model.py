import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Імпортуємо константи з модуля обробки даних
from data_processing import HORIZON, LOOKBACK, FEATURE_COLUMNS


def build_lstm(units_first=128, units_second=64, dropout_rate=0.2):
    """
    Будує LSTM модель.

    Архітектура:
        Вхід -> LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> Dense(24)

    Параметри:
        units_first   - кількість нейронів у першому LSTM шарі
        units_second  - кількість нейронів у другому LSTM шарі
        dropout_rate  - частка нейронів яка випадково вимикається під час навчання
                        щоб модель не перенавчалась

    Повертає скомпільовану модель.
    """
    model = keras.Sequential([

        # Вхідний шар: кожен зразок це послідовність (96 кроків, 7 ознак)
        keras.Input(shape=(LOOKBACK, len(FEATURE_COLUMNS))),

        # Перший LSTM шар
        # return_sequences=True означає що передаємо весь вихід далі,
        # а не тільки останній крок - це потрібно щоб другий LSTM теж
        # бачив всю послідовність
        layers.LSTM(units_first, return_sequences=True),

        # Dropout вимикає випадкові нейрони під час навчання
        # це змушує модель не покладатись на конкретні нейрони
        # і краще узагальнювати
        layers.Dropout(dropout_rate),

        # Другий LSTM шар
        # return_sequences=False (за замовчуванням) - повертаємо
        # тільки останній крок бо далі йде Dense шар
        layers.LSTM(units_second),

        layers.Dropout(dropout_rate),

        # Вихідний шар: прогнозуємо HORIZON значень (24 години)
        layers.Dense(HORIZON),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def get_callbacks(patience=10):
    """
    Повертає список callbacks для навчання.

    EarlyStopping - зупиняє навчання якщо val_loss не покращується
                    протягом patience епох. Це рятує від перенавчання
                    і економить час.

    ModelCheckpoint - зберігає найкращу версію моделі на диск.
                      Бо після early stopping ваги можуть бути не найкращими.

    Параметри:
        patience - скільки епох чекати без покращення перед зупинкою
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_lstm.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )

    return [early_stopping, checkpoint]


def train_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Будує і навчає LSTM модель.

    Параметри:
        X_train, y_train - тренувальні дані
        X_val,   y_val   - валідаційні дані (для early stopping)
        epochs           - максимальна кількість епох
        batch_size       - скільки зразків обробляємо за один крок навчання

    Повертає навчену модель і історію навчання.
    """
    model = build_lstm()

    print("Архітектура LSTM:")
    model.summary()
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=1
    )

    print("\nНавчання завершено.")
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    print(f"Найкраща епоха: {best_epoch},  val_loss: {best_val_loss:.6f}")

    return model, history


def evaluate_lstm(model, X_test, y_test, scaler):
    """
    Оцінює модель на тестових даних.
    Метрики рахуються в оригінальних одиницях (градуси Цельсія),
    а не в нормалізованих значеннях.

    Повертає словник з метриками і масив прогнозів в оригінальному масштабі.
    """
    from data_processing import inverse_scale_target

    # Отримуємо прогнози (нормалізовані)
    predictions_scaled = model.predict(X_test, verbose=0)

    # Переводимо прогнози і факт назад в оригінальний масштаб
    # Беремо перший крок горизонту для простоти метрик (t+1)
    pred_original = inverse_scale_target(predictions_scaled[:, 0], scaler)
    true_original = inverse_scale_target(y_test[:, 0], scaler)

    mae  = np.mean(np.abs(pred_original - true_original))
    rmse = np.sqrt(np.mean((pred_original - true_original) ** 2))
    mask = np.abs(true_original) > 1.0
    mape = np.mean(np.abs((true_original[mask] - pred_original[mask]) / true_original[mask])) * 100

    print("\n--- Результати LSTM на тестових даних ---")
    print(f"MAE:  {mae:.4f} градусів")
    print(f"RMSE: {rmse:.4f} градусів")
    print(f"MAPE: {mape:.2f}%")

    return {
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
    }, pred_original, true_original