import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_processing import HORIZON, LOOKBACK, FEATURE_COLUMNS


def positional_encoding(length, depth):
    """
    Генерує позиційне кодування для вхідної послідовності.

    Transformer на відміну від LSTM не читає дані послідовно -
    він бачить всі кроки одночасно. Через це він не знає
    який крок йде першим а який останнім.

    Позиційне кодування вирішує цю проблему: ми додаємо до кожного
    кроку унікальний вектор який кодує його позицію в послідовності.
    Використовуємо sin/cos функції як в оригінальній статті
    "Attention is All You Need".

    Параметри:
        length - довжина послідовності (LOOKBACK = 96)
        depth  - розмірність вектора ознак
    """
    positions = np.arange(length)[:, np.newaxis]
    dims      = np.arange(depth)[np.newaxis, :]

    # Парні позиції - sin, непарні - cos
    angles = positions / np.power(10000, (2 * (dims // 2)) / depth)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    # Повертаємо як тензор форми (1, length, depth)
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate):
    """
    Один блок Transformer енкодера.

    Складається з двох частин:
    1. Multi-Head Self-Attention - модель дивиться на всі кроки
       послідовності і вирішує які з них важливі для кожного конкретного кроку
    2. Feed-Forward мережа - два Dense шари які обробляють кожен крок окремо

    Після кожної частини є:
    - Dropout для регуляризації
    - Layer Normalization + залишкове з'єднання (додаємо вхід до виходу)
      це допомагає градієнтам текти через глибоку мережу

    Параметри:
        x            - вхідний тензор
        num_heads    - кількість голів уваги
        ff_dim       - розмірність прихованого шару у feed-forward частині
        dropout_rate - частка нейронів що вимикаються під час навчання
    """
    # --- Частина 1: Multi-Head Self-Attention ---
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, x)

    attention_output = layers.Dropout(dropout_rate)(attention_output)

    # Залишкове з'єднання і нормалізація
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # --- Частина 2: Feed-Forward ---
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)

    # Залишкове з'єднання і нормалізація
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x


def build_transformer(
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout_rate=0.1
):
    """
    Будує Transformer модель для прогнозування часових рядів.

    Архітектура:
        Вхід -> Проекція в d_model -> Позиційне кодування
             -> N блоків енкодера
             -> GlobalAveragePooling
             -> Dense(64) -> Dropout -> Dense(24)

    Параметри:
        d_model      - внутрішня розмірність моделі (всі шари працюють з векторами цього розміру)
        num_heads    - кількість голів у Multi-Head Attention
        ff_dim       - розмірність Feed-Forward шару всередині блоку
        num_blocks   - кількість блоків енкодера
        dropout_rate - dropout для регуляризації

    Повертає скомпільовану модель.
    """
    inputs = keras.Input(shape=(LOOKBACK, len(FEATURE_COLUMNS)))

    # Проектуємо вхід з 7 ознак в d_model вимірів
    # щоб Multi-Head Attention мав з чим працювати
    x = layers.Dense(d_model)(inputs)

    # Додаємо позиційне кодування щоб модель знала порядок кроків
    x = x + positional_encoding(LOOKBACK, d_model)

    x = layers.Dropout(dropout_rate)(x)

    # Проганяємо через N блоків енкодера
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    # GlobalAveragePooling згортає послідовність (96, 64) в один вектор (64,)
    # беручи середнє по всіх кроках часу
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Вихід: прогноз на HORIZON кроків вперед
    outputs = layers.Dense(HORIZON)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def get_callbacks(patience=10):
    """
    Повертає callbacks для навчання Transformer.
    Аналогічно до LSTM модуля.

    EarlyStopping  - зупиняє навчання якщо val_loss не покращується
    ModelCheckpoint - зберігає найкращу версію моделі
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_transformer.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )

    return [early_stopping, checkpoint]


def train_transformer(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Будує і навчає Transformer модель.

    Параметри і логіка повністю аналогічні до train_lstm -
    це зроблено навмисно щоб умови порівняння були однаковими:
    той самий optimizer, та сама кількість епох, той самий batch_size.

    Повертає навчену модель і історію навчання.
    """
    model = build_transformer()

    print("Архітектура Transformer:")
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


def evaluate_transformer(model, X_test, y_test, scaler):
    """
    Оцінює модель на тестових даних.
    Повністю аналогічна до evaluate_lstm - знову навмисно,
    щоб метрики рахувались однаковим способом для обох моделей.

    Повертає словник з метриками і масив прогнозів в оригінальному масштабі.
    """
    from data_processing import inverse_scale_target

    predictions_scaled = model.predict(X_test, verbose=0)

    pred_original = inverse_scale_target(predictions_scaled[:, 0], scaler)
    true_original = inverse_scale_target(y_test[:, 0], scaler)

    mae  = np.mean(np.abs(pred_original - true_original))
    rmse = np.sqrt(np.mean((pred_original - true_original) ** 2))
    mask = np.abs(true_original) > 1.0
    mape = np.mean(np.abs((true_original[mask] - pred_original[mask]) / true_original[mask])) * 100

    print("\n--- Результати Transformer на тестових даних ---")
    print(f"MAE:  {mae:.4f} градусів")
    print(f"RMSE: {rmse:.4f} градусів")
    print(f"MAPE: {mape:.2f}%")

    return {
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
    }, pred_original, true_original