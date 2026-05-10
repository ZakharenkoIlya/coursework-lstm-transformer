import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Налаштування загального стилю графіків
plt.rcParams["figure.dpi"]       = 120
plt.rcParams["font.size"]        = 11
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3
plt.rcParams["axes.spines.top"]  = False
plt.rcParams["axes.spines.right"] = False


def plot_predictions(true_values, lstm_pred, transformer_pred, n_points=300):
    """
    Графік факт vs прогноз для обох моделей на тестовому відрізку.

    Показуємо тільки перші n_points точок щоб графік був читабельним,
    бо 3366 точок злились би в суцільну масу.

    Параметри:
        true_values       - реальні значення OT
        lstm_pred         - прогнози LSTM
        transformer_pred  - прогнози Transformer
        n_points          - скільки точок показувати
    """
    true   = true_values[:n_points]
    lstm   = lstm_pred[:n_points]
    transf = transformer_pred[:n_points]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Прогнозування температури масла трансформатора (OT)", fontsize=14)

    # Верхній графік - LSTM
    axes[0].plot(true,  label="Факт",  color="#2c7bb6", linewidth=1.5)
    axes[0].plot(lstm,  label="LSTM",  color="#d7191c", linewidth=1.2, linestyle="--")
    axes[0].set_title("LSTM")
    axes[0].set_ylabel("Температура (°C)")
    axes[0].legend(loc="upper right")

    # Нижній графік - Transformer
    axes[1].plot(true,  label="Факт",        color="#2c7bb6", linewidth=1.5)
    axes[1].plot(transf, label="Transformer", color="#1a9641", linewidth=1.2, linestyle="--")
    axes[1].set_title("Transformer")
    axes[1].set_ylabel("Температура (°C)")
    axes[1].set_xlabel("Години (тестовий відрізок)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("predictions.png", bbox_inches="tight")
    plt.show()
    print("Збережено: predictions.png")


def plot_predictions_combined(true_values, lstm_pred, transformer_pred, n_points=300):
    """
    Альтернативний варіант - всі три лінії на одному графіку.
    Зручно для прямого порівняння.
    """
    true   = true_values[:n_points]
    lstm   = lstm_pred[:n_points]
    transf = transformer_pred[:n_points]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(true,   label="Факт",        color="#2c7bb6", linewidth=1.5)
    ax.plot(lstm,   label="LSTM",        color="#d7191c", linewidth=1.2, linestyle="--")
    ax.plot(transf, label="Transformer", color="#1a9641", linewidth=1.2, linestyle=":")

    ax.set_title("Порівняння прогнозів LSTM та Transformer", fontsize=13)
    ax.set_xlabel("Години (тестовий відрізок)")
    ax.set_ylabel("Температура (°C)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("predictions_combined.png", bbox_inches="tight")
    plt.show()
    print("Збережено: predictions_combined.png")


def plot_error_distribution(true_values, lstm_pred, transformer_pred):
    """
    Гістограма розподілу похибок для кожної моделі.
    Показує наскільки рівномірно модель помиляється -
    ідеально розподіл має бути схожий на дзвін centered біля нуля.
    """
    lstm_errors   = lstm_pred - true_values
    transf_errors = transformer_pred - true_values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Розподіл похибок прогнозування", fontsize=13)

    axes[0].hist(lstm_errors, bins=60, color="#d7191c", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1.2, linestyle="--", label="Нуль")
    axes[0].axvline(lstm_errors.mean(), color="#d7191c", linewidth=1.5,
                    linestyle="-", label=f"Середня похибка: {lstm_errors.mean():.3f}")
    axes[0].set_title("LSTM")
    axes[0].set_xlabel("Похибка (°C)")
    axes[0].set_ylabel("Кількість")
    axes[0].legend()

    axes[1].hist(transf_errors, bins=60, color="#1a9641", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--", label="Нуль")
    axes[1].axvline(transf_errors.mean(), color="#1a9641", linewidth=1.5,
                    linestyle="-", label=f"Середня похибка: {transf_errors.mean():.3f}")
    axes[1].set_title("Transformer")
    axes[1].set_xlabel("Похибка (°C)")
    axes[1].set_ylabel("Кількість")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("error_distribution.png", bbox_inches="tight")
    plt.show()
    print("Збережено: error_distribution.png")


def plot_metrics_comparison(lstm_metrics, transformer_metrics):
    """
    Стовпчиковий графік порівняння метрик MAE і RMSE.
    MAPE не включаємо бо вона в інших одиницях (відсотки)
    і зіпсує масштаб графіку.
    """
    metrics = ["MAE", "RMSE"]
    lstm_values   = [lstm_metrics["mae"],        lstm_metrics["rmse"]]
    transf_values = [transformer_metrics["mae"], transformer_metrics["rmse"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_lstm   = ax.bar(x - width/2, lstm_values,   width, label="LSTM",
                         color="#d7191c", alpha=0.85)
    bars_transf = ax.bar(x + width/2, transf_values, width, label="Transformer",
                         color="#1a9641", alpha=0.85)

    # Підписи значень над стовпцями
    for bar in bars_lstm:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

    for bar in bars_transf:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Порівняння метрик LSTM та Transformer (°C)", fontsize=13)
    ax.set_ylabel("Похибка (°C)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig("metrics_comparison.png", bbox_inches="tight")
    plt.show()
    print("Збережено: metrics_comparison.png")


def plot_scatter(true_values, lstm_pred, transformer_pred):
    """
    Scatter plot: факт vs прогноз.
    Ідеальна модель дає всі точки на діагональній лінії y=x.
    Чим ближче точки до діагоналі - тим краща модель.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Факт vs Прогноз", fontsize=13)

    # Діапазон для діагональної лінії
    min_val = min(true_values.min(), lstm_pred.min(), transformer_pred.min())
    max_val = max(true_values.max(), lstm_pred.max(), transformer_pred.max())

    for ax, pred, title, color in zip(
        axes,
        [lstm_pred, transformer_pred],
        ["LSTM", "Transformer"],
        ["#d7191c", "#1a9641"]
    ):
        ax.scatter(true_values, pred, alpha=0.2, s=8, color=color)
        ax.plot([min_val, max_val], [min_val, max_val],
                color="black", linewidth=1.2, linestyle="--", label="Ідеальний прогноз")
        ax.set_title(title)
        ax.set_xlabel("Факт (°C)")
        ax.set_ylabel("Прогноз (°C)")
        ax.legend()

    plt.tight_layout()
    plt.savefig("scatter.png", bbox_inches="tight")
    plt.show()
    print("Збережено: scatter.png")


def generate_all_plots(true_values, lstm_pred, transformer_pred,
                       lstm_metrics, transformer_metrics):
    """
    Генерує всі графіки одним викликом.
    """
    print("Генеруємо графіки...\n")

    plot_predictions(true_values, lstm_pred, transformer_pred)
    plot_predictions_combined(true_values, lstm_pred, transformer_pred)
    plot_error_distribution(true_values, lstm_pred, transformer_pred)
    plot_metrics_comparison(lstm_metrics, transformer_metrics)
    plot_scatter(true_values, lstm_pred, transformer_pred)

    print("\nВсі графіки збережені в папці проекту.")