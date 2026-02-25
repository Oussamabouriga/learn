```
# ============================================================
# ACCURACY (%) FOR THE 4 MODELS + PLOT (0 -> 100%)
# ============================================================
# This computes regression "accuracy" using tolerance-based accuracy:
# A prediction is considered correct if |y_true - y_pred| <= tolerance
#
# We will compute:
# - Accuracy@±0.5
# - Accuracy@±1.0
#
# Expected prediction variables from your previous code:
#   Model 1: pred_test_clipped (or pred_test_baseline_w_clip if your model1 naming changed)
#   Model 2: pred_test_baseline_w_clip   (weighted baseline)
#   Model 3: pred_test_random_w_clip     (random search weighted)
#   Model 4: pred_test_grid_w_clip       (small grid weighted)
#
# IMPORTANT:
# If your Model 1 variable is named differently, adjust below.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Blue colors (same style family)
# ------------------------------------------------------------
BLUE_3 = "#3B82F6"
BLUE_3_DARK = "#1D4ED8"
BLUE_3_LIGHT = "#93C5FD"
GRID_BLUE = "#DBEAFE"

# ------------------------------------------------------------
# 2) Select predictions for each model (edit only if names differ)
# ------------------------------------------------------------
# --- Model 1 (normal baseline)
if "pred_test_clipped" in globals():
    pred_test_model1 = np.asarray(pred_test_clipped, dtype=float)
elif "baseline_pred_test" in globals():
    pred_test_model1 = np.asarray(baseline_pred_test, dtype=float)
else:
    raise NameError("Model 1 test predictions not found. Expected 'pred_test_clipped' or 'baseline_pred_test'.")

# --- Model 2 (weighted baseline)
if "pred_test_baseline_w_clip" in globals():
    pred_test_model2 = np.asarray(pred_test_baseline_w_clip, dtype=float)
else:
    raise NameError("Model 2 test predictions not found. Expected 'pred_test_baseline_w_clip'.")

# --- Model 3 (random search weighted)
if "pred_test_random_w_clip" in globals():
    pred_test_model3 = np.asarray(pred_test_random_w_clip, dtype=float)
else:
    raise NameError("Model 3 test predictions not found. Expected 'pred_test_random_w_clip'.")

# --- Model 4 (small grid search weighted)
if "pred_test_grid_w_clip" in globals():
    pred_test_model4 = np.asarray(pred_test_grid_w_clip, dtype=float)
else:
    raise NameError("Model 4 test predictions not found. Expected 'pred_test_grid_w_clip'.")

# Ground truth
y_true_test = np.asarray(y_test, dtype=float)

# Sanity check lengths
print("Lengths:")
print("y_test:", len(y_true_test))
print("Model1:", len(pred_test_model1))
print("Model2:", len(pred_test_model2))
print("Model3:", len(pred_test_model3))
print("Model4:", len(pred_test_model4))

# ------------------------------------------------------------
# 3) Compute accuracy (%) with tolerance (regression accuracy)
# ------------------------------------------------------------
tol_05 = 0.5
tol_10 = 1.0

acc_model1_05 = (np.abs(y_true_test - pred_test_model1) <= tol_05).mean() * 100
acc_model2_05 = (np.abs(y_true_test - pred_test_model2) <= tol_05).mean() * 100
acc_model3_05 = (np.abs(y_true_test - pred_test_model3) <= tol_05).mean() * 100
acc_model4_05 = (np.abs(y_true_test - pred_test_model4) <= tol_05).mean() * 100

acc_model1_10 = (np.abs(y_true_test - pred_test_model1) <= tol_10).mean() * 100
acc_model2_10 = (np.abs(y_true_test - pred_test_model2) <= tol_10).mean() * 100
acc_model3_10 = (np.abs(y_true_test - pred_test_model3) <= tol_10).mean() * 100
acc_model4_10 = (np.abs(y_true_test - pred_test_model4) <= tol_10).mean() * 100

# ------------------------------------------------------------
# 4) Accuracy table (0 -> 100%)
# ------------------------------------------------------------
accuracy_4models_df = pd.DataFrame({
    "Modèle": [
        "Modèle 1 - Baseline",
        "Modèle 2 - Weighted",
        "Modèle 3 - Random Search Weighted",
        "Modèle 4 - Small Grid Weighted"
    ],
    "Accuracy@±0.5 (%)": [
        acc_model1_05, acc_model2_05, acc_model3_05, acc_model4_05
    ],
    "Accuracy@±1.0 (%)": [
        acc_model1_10, acc_model2_10, acc_model3_10, acc_model4_10
    ]
})

print("\n=== Accuracy (%) des 4 modèles (TEST) ===")
display(accuracy_4models_df)

# ------------------------------------------------------------
# 5) Plot A — Accuracy@±1.0 only (0 -> 100%)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    accuracy_4models_df["Modèle"],
    accuracy_4models_df["Accuracy@±1.0 (%)"],
    color=[BLUE_3_LIGHT, BLUE_3, BLUE_3_DARK, "#64748B"]  # blue + gray dark
)

ax.set_title("Accuracy des 4 modèles (tolérance ±1 point)", pad=18)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Modèle")
ax.set_ylim(0, 100)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style (blue)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

# Border style
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BLUE_3_LIGHT)

# Labels on bars
for b in bars:
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width()/2,
        h + 1,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color=BLUE_3_DARK
    )

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6) Plot B — Grouped bars for Accuracy@±0.5 and Accuracy@±1.0 (0 -> 100%)
# ------------------------------------------------------------
plot_acc_long = accuracy_4models_df.melt(
    id_vars=["Modèle"],
    value_vars=["Accuracy@±0.5 (%)", "Accuracy@±1.0 (%)"],
    var_name="Métrique",
    value_name="Accuracy"
)

pivot_acc = plot_acc_long.pivot(index="Modèle", columns="Métrique", values="Accuracy")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_acc.plot(
    kind="bar",
    ax=ax,
    width=0.8,
    color=[BLUE_3_LIGHT, BLUE_3_DARK]
)

ax.set_title("Accuracy des 4 modèles (±0.5 et ±1.0 point)", pad=18)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Modèle")
ax.set_ylim(0, 100)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style (blue)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

# Border style
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BLUE_3_LIGHT)

# Labels on grouped bars
for container in ax.containers:
    for b in container:
        h = b.get_height()
        if pd.notna(h):
            ax.text(
                b.get_x() + b.get_width()/2,
                h + 1,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.xticks(rotation=15, ha="right")
plt.legend(title="Tolérance", frameon=True)
plt.tight_layout()
plt.show()

```
