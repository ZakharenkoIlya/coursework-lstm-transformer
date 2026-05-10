import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from data_processing import prepare_dataset
from lstm_model import train_lstm, evaluate_lstm
from transformer_model import train_transformer, evaluate_transformer
from visualization import generate_all_plots

# Шлях до датасету
DATA_PATH = "ETTh1.csv"

# Шляхи де зберігаються навчені моделі
LSTM_PATH        = "best_lstm.keras"
TRANSFORMER_PATH = "best_transformer.keras"


def load_or_train_lstm(X_train, y_train, X_val, y_val):
    """
    Завантажує збережену LSTM модель якщо вона існує.
    Якщо ні — навчає і зберігає.
    """
    if os.path.exists(LSTM_PATH):
        print(f"Знайдено збережену LSTM модель: {LSTM_PATH}")
        print("Завантажуємо...")
        model = tf.keras.models.load_model(LSTM_PATH)
        print("Завантажено.")
        return model, None
    else:
        print("Збереженої LSTM моделі не знайдено, починаємо навчання...")
        return train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)


def load_or_train_transformer(X_train, y_train, X_val, y_val):
    """
    Завантажує збережену Transformer модель якщо вона існує.
    Якщо ні — навчає і зберігає.
    """
    if os.path.exists(TRANSFORMER_PATH):
        print(f"Знайдено збережену Transformer модель: {TRANSFORMER_PATH}")
        print("Завантажуємо...")
        model = tf.keras.models.load_model(TRANSFORMER_PATH)
        print("Завантажено.")
        return model, None
    else:
        print("Збереженої Transformer моделі не знайдено, починаємо навчання...")
        return train_transformer(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)


def main():
    # Крок 1: завантаження і підготовка даних
    data = prepare_dataset(DATA_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    scaler  = data["scaler"]

    # Крок 2: навчання або завантаження LSTM
    print("\n" + "=" * 50)
    print("LSTM")
    print("=" * 50)
    lstm_model, lstm_history = load_or_train_lstm(X_train, y_train, X_val, y_val)

    # Крок 3: навчання або завантаження Transformer
    print("\n" + "=" * 50)
    print("Transformer")
    print("=" * 50)
    transformer_model, transformer_history = load_or_train_transformer(X_train, y_train, X_val, y_val)

    # Крок 4: оцінка обох моделей на тестових даних
    print("\n" + "=" * 50)
    print("Оцінка моделей на тестових даних")
    print("=" * 50)

    lstm_metrics, lstm_pred, true_values = evaluate_lstm(
        lstm_model, X_test, y_test, scaler
    )

    transformer_metrics, transformer_pred, _ = evaluate_transformer(
        transformer_model, X_test, y_test, scaler
    )

    # Крок 5: фінальне порівняння
    print("\n" + "=" * 50)
    print("Порівняння результатів")
    print("=" * 50)
    print(f"{'Метрика':<10} {'LSTM':>12} {'Transformer':>14}")
    print("-" * 38)
    print(f"{'MAE':<10} {lstm_metrics['mae']:>12.4f} {transformer_metrics['mae']:>14.4f}")
    print(f"{'RMSE':<10} {lstm_metrics['rmse']:>12.4f} {transformer_metrics['rmse']:>14.4f}")
    print(f"{'MAPE':<10} {lstm_metrics['mape']:>11.2f}% {transformer_metrics['mape']:>13.2f}%")
    print("-" * 38)

    better = "LSTM" if lstm_metrics["mae"] < transformer_metrics["mae"] else "Transformer"
    print(f"Краща модель за MAE: {better}")
# Крок 6: графіки
    generate_all_plots(true_values, lstm_pred, transformer_pred, lstm_metrics, transformer_metrics)


if __name__ == "__main__":
    main()