import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COLORS = {
    "Fact": "#2b2b2b",
    "LSTM": "#e41a1c",
    "Transformer": "#4daf4a",
    "DLinear": "#377eb8",
}

def plot_predictions_combined(
    true_values, lstm_pred, transformer_pred, dlinear_pred,
    dataset_name="Dataset", target_name="Target", unit="°C",
    n_points=200, save_path="predictions_combined.png"
):
  true = true_values[:n_points]
  lstm = lstm_pred[:n_points]
  transf = transformer_pred[:n_points]
  dlinear = dlinear_pred[:n_points]

  fig, ax = plt.subplots(figsize=(12, 5))

  ax.plot(true, label=f"Факт ({target_name})", color=COLORS["Fact"], linewidth=1.8, alpha=0.8)
  ax.plot(lstm, label="LSTM", color=COLORS["LSTM"], linewidth=1.2, linestyle="--")
  ax.plot(transf, label="Transformer", color=COLORS["Transformer"], linewidth=1.2, linestyle=":")
  ax.plot(dlinear, label="DLinear (LTSF)", color=COLORS["DLinear"], linewidth=1.4, linestyle="-.")

  ax.set_title(f"Порівняння прогнозів моделей ({dataset_name}, target: {target_name})", fontsize=12)
  ax.set_xlabel("Часові кроки")
  ax.set_ylabel(f"Значення ({unit})" if unit else "Значення")
  ax.legend(loc="upper right", frameon=True)

  y_min = min(true.min(), lstm.min(), transf.min(), dlinear.min())
  y_max = max(true.max(), lstm.max(), transf.max(), dlinear.max())
  margin = (y_max - y_min) * 0.08 if y_max != y_min else 1.0
  ax.set_ylim(y_min - margin, y_max + margin)

  plt.tight_layout()
  plt.savefig(save_path, bbox_inches="tight")
  plt.close()

def plot_error_by_horizon(
    y_test_orig, lstm_pred_all, transf_pred_all, dlinear_pred_all,
    dataset_name="Dataset", unit="°C", save_path="error_by_horizon.png"
):
  horizon = y_test_orig.shape[1]

  lstm_mae = np.mean(np.abs(lstm_pred_all - y_test_orig), axis=0)
  transf_mae = np.mean(np.abs(transf_pred_all - y_test_orig), axis=0)
  dlinear_mae = np.mean(np.abs(dlinear_pred_all - y_test_orig), axis=0)

  steps = np.arange(1, horizon + 1)

  fig, ax = plt.subplots(figsize=(10, 5))

  ax.plot(steps, lstm_mae, label="LSTM", color=COLORS["LSTM"], linewidth=2)
  ax.plot(steps, transf_mae, label="Transformer", color=COLORS["Transformer"], linewidth=2)
  ax.plot(steps, dlinear_mae, label="DLinear (LTSF)", color=COLORS["DLinear"], linewidth=2)

  ax.set_title(f"Динаміка зростання похибки (MAE) від горизонту ({dataset_name})", fontsize=12)
  ax.set_xlabel(f"Крок горизонту прогнозу (від 1 до {horizon})")
  ax.set_ylabel(f"Середня абсолютна похибка MAE ({unit})" if unit else "Середня абсолютна похибка MAE")
  ax.legend(loc="upper left", frameon=True)

  plt.tight_layout()
  plt.savefig(save_path, bbox_inches="tight")
  plt.close()

def plot_metrics_and_time(
    lstm_m, transf_m, dlinear_m, times,
    default_times=None, unit="°C", save_path="metrics_and_time.png"
):
  models = ["LSTM", "Transformer", "DLinear"]
  mae_values = [lstm_m["mae"], transf_m["mae"], dlinear_m["mae"]]
  time_values = [times["lstm"], times["transformer"], times["dlinear"]]

  if all(t == 0.0 for t in time_values) and default_times is not None:
    time_values = default_times

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  bars1 = ax1.bar(models, mae_values, color=["#e63946", "#2a9d8f", "#457b9d"], width=0.4)
  ax1.set_title("Похибка MAE (нижче = краще)")
  ax1.set_ylabel(f"Значення ({unit})" if unit else "Значення")
  ax1.set_ylim(0, max(mae_values) * 1.25)

  for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + (max(mae_values) * 0.02), f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

  bars2 = ax2.bar(models, time_values, color=["#e63946", "#2a9d8f", "#457b9d"], width=0.4)
  ax2.set_title("Час навчання (нижче = швидше)")
  ax2.set_ylabel("Секунди (с)")
  ax2.set_ylim(0, max(time_values) * 1.25)

  for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max(time_values) * 0.02), f"{yval:.1f}s", ha="center", va="bottom", fontsize=9)

  plt.tight_layout()
  plt.savefig(save_path, dpi=300)
  plt.close()