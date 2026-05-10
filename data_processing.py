import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Назви всіх колонок з ознаками (без дати)
FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# Колонка яку прогнозуємо
TARGET_COLUMN = "OT"

# Скільки годин дивимось назад
LOOKBACK = 96

# Скільки годин прогнозуємо вперед
HORIZON = 24

# Розбиття датасету: 70% train, 10% validation, 20% test
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1


def load_data(filepath):
    """
    Завантажує CSV файл і повертає DataFrame.
    Колонка date парситься як datetime і стає індексом.
    """
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    return df


def check_data(df):
    """
    Перевіряє датафрейм на пропуски та друкує коротку статистику.
    Повертає True якщо все ок, False якщо є проблеми.
    """
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("Пропуски: відсутні")
    else:
        print("Знайдено пропуски:")
        print(missing[missing > 0])

    print(f"\nРозмір датасету: {len(df)} рядків, {len(df.columns)} колонок")
    print(f"Період: {df.index[0]}  ->  {df.index[-1]}")
    print(f"\nСтатистика по колонках:")
    print(df.describe().round(2))

    return missing.sum() == 0


def split_data(df):
    """
    Розбиває датафрейм на train / validation / test.
    Повертає три датафрейми в тому ж порядку.
    """
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Train:      {len(train)} рядків  ({train.index[0]}  ->  {train.index[-1]})")
    print(f"Validation: {len(val)} рядків  ({val.index[0]}  ->  {val.index[-1]})")
    print(f"Test:       {len(test)} рядків  ({test.index[0]}  ->  {test.index[-1]})")

    return train, val, test


def fit_scaler(train_df):
    """
    Навчає MinMaxScaler тільки на тренувальних даних.
    Важливо: scaler навчається тільки на train щоб не було витоку
    інформації з val/test в нормалізацію.
    Повертає навчений scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


def apply_scaler(df, scaler):
    """
    Застосовує вже навчений scaler до датафрейму.
    Повертає новий датафрейм з нормалізованими значеннями.
    """
    scaled_values = scaler.transform(df[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(
        scaled_values,
        columns=FEATURE_COLUMNS,
        index=df.index
    )
    return scaled_df


def inverse_scale_target(values, scaler):
    """
    Повертає значення OT назад в оригінальний масштаб.
    Потрібно щоб метрики були в реальних одиницях (градуси Цельсія),
    а не в нормалізованих [0, 1].

    values - масив формою (N,) або (N, 1)
    """
    values = values.reshape(-1, 1)

    # Знаходимо індекс колонки OT серед всіх ознак
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

    # Створюємо нульовий масив потрібної форми і вставляємо наші значення
    # в потрібну позицію, бо scaler очікує всі колонки разом
    dummy = np.zeros((len(values), len(FEATURE_COLUMNS)))
    dummy[:, target_index] = values[:, 0]

    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_index]


def make_windows(scaled_df):
    """
    Перетворює нормалізований датафрейм на пари (X, y) для навчання.

    X - вхідна послідовність: (кількість вікон, LOOKBACK, кількість ознак)
    y - цільова послідовність: (кількість вікон, HORIZON)

    Принцип: беремо 96 годин всіх ознак як вхід,
    і наступні 24 години тільки OT як ціль.
    """
    data = scaled_df[FEATURE_COLUMNS].values
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

    X_list = []
    y_list = []

    total_steps = len(data) - LOOKBACK - HORIZON + 1

    for i in range(total_steps):
        # Вхід: всі ознаки за LOOKBACK кроків
        x_window = data[i : i + LOOKBACK]

        # Ціль: тільки OT за наступні HORIZON кроків
        y_window = data[i + LOOKBACK : i + LOOKBACK + HORIZON, target_index]

        X_list.append(x_window)
        y_list.append(y_window)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def prepare_dataset(filepath):
    """
    Головна функція модуля.
    Приймає шлях до CSV і повертає все необхідне для навчання моделей.

    Повертає словник з:
        X_train, y_train  - тренувальна вибірка
        X_val,   y_val    - валідаційна вибірка
        X_test,  y_test   - тестова вибірка
        scaler            - навчений scaler (потрібен для inverse_scale_target)
    """
    print("=" * 50)
    print(f"Завантажуємо: {filepath}")
    print("=" * 50)

    # Крок 1: завантаження
    df = load_data(filepath)

    # Крок 2: перевірка
    print("\n--- Перевірка даних ---")
    check_data(df)

    # Крок 3: розбиття на train/val/test (до нормалізації)
    print("\n--- Розбиття датасету ---")
    train_df, val_df, test_df = split_data(df)

    # Крок 4: навчаємо scaler тільки на train
    scaler = fit_scaler(train_df)

    # Крок 5: нормалізуємо кожну частину
    train_scaled = apply_scaler(train_df, scaler)
    val_scaled   = apply_scaler(val_df,   scaler)
    test_scaled  = apply_scaler(test_df,  scaler)

    # Крок 6: створюємо вікна
    X_train, y_train = make_windows(train_scaled)
    X_val,   y_val   = make_windows(val_scaled)
    X_test,  y_test  = make_windows(test_scaled)

    print("\n--- Форми тензорів ---")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    print("\nГотово. Дані підготовлені для навчання.")
    print("=" * 50)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "scaler":  scaler,
    }