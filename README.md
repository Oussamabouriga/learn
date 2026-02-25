```
# ============================================================
# FUNCTION: Plot accuracy (%) for 4 regression models
# Accuracy = % of predictions within a tolerance you choose
# (example: tolerance=0.5 or tolerance=1.0)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_4models_by_tolerance(
    y_true,
    pred_model1,
    pred_model2,
    pred_model3,
    pred_model4,
    tolerance=1.0,
    model_names=None,
    title=None,
    colors=None
):
    """
    Plot one bar chart (4 bars) for regression accuracy with chosen tolerance.

    Accuracy rule:
        correct if abs(y_true - y_pred) <= tolerance

    Parameters
    ----------
    y_true : array-like
        True target values (test set)
    pred_model1, pred_model2, pred_model3, pred_model4 : array-like
        Predictions for the 4 models
    tolerance : float, default=1.0
        Allowed prediction error (in target points)
    model_names : list[str], optional
        Names of the 4 models
    title : str, optional
        Custom chart title
    colors : list[str], optional
        4 bar colors
    """

    # -----------------------------
    # 1) Convert to numpy arrays
    # -----------------------------
    y_true = np.asarray(y_true, dtype=float)
    p1 = np.asarray(pred_model1, dtype=float)
    p2 = np.asarray(pred_model2, dtype=float)
    p3 = np.asarray(pred_model3, dtype=float)
    p4 = np.asarray(pred_model4, dtype=float)

    # -----------------------------
    # 2) Basic checks
    # -----------------------------
    n = len(y_true)
    if not (len(p1) == len(p2) == len(p3) == len(p4) == n):
        raise ValueError("All predictions and y_true must have the same length.")

    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    # -----------------------------
    # 3) Compute accuracy (%)
    # -----------------------------
    acc1 = (np.abs(y_true - p1) <= tolerance).mean() * 100
    acc2 = (np.abs(y_true - p2) <= tolerance).mean() * 100
    acc3 = (np.abs(y_true - p3) <= tolerance).mean() * 100
    acc4 = (np.abs(y_true - p4) <= tolerance).mean() * 100

    # -----------------------------
    # 4) Prepare display table
    # -----------------------------
    if model_names is None:
        model_names = [
            "Modèle 1 - Baseline",
            "Modèle 2 - Weighted",
            "Modèle 3 - Random Search",
            "Modèle 4 - Small Grid"
        ]

    if len(model_names) != 4:
        raise ValueError("model_names must contain exactly 4 names.")

    acc_df = pd.DataFrame({
        "Modèle": model_names,
        "Accuracy (%)": [acc1, acc2, acc3, acc4]
    })

    # -----------------------------
    # 5) Colors (Blue 3 style + dark gray)
    # -----------------------------
    if colors is None:
        colors = ["#93C5FD", "#3B82F6", "#1D4ED8", "#64748B"]  # blue family + gray

    if len(colors) != 4:
        raise ValueError("colors must contain exactly 4 colors.")

    # -----------------------------
    # 6) Plot (one bar chart only)
    # -----------------------------
    if title is None:
        title = f"Accuracy des 4 modèles (tolérance ±{tolerance} point{'s' if tolerance != 1 else ''})"

    GRID_BLUE = "#DBEAFE"
    BLUE_BORDER = "#93C5FD"
    LABEL_BLUE = "#1D4ED8"

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        acc_df["Modèle"],
        acc_df["Accuracy (%)"],
        color=colors,
        width=0.7
    )

    ax.set_title(title, pad=18)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    # White background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Blue grid style
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
    ax.set_axisbelow(True)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(BLUE_BORDER)

    # Labels on bars
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=LABEL_BLUE
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    return acc_df


# Example using your prediction variables
acc_df_05 = plot_accuracy_4models_by_tolerance(
    y_true=y_test,
    pred_model1=pred_test_clipped,            # Model 1 baseline
    pred_model2=pred_test_baseline_w_clip,    # Model 2 weighted
    pred_model3=pred_test_random_w_clip,      # Model 3 random search weighted
    pred_model4=pred_test_grid_w_clip,        # Model 4 small grid weighted
    tolerance=0.5
)

display(acc_df_05)


acc_df_10 = plot_accuracy_4models_by_tolerance(
    y_true=y_test,
    pred_model1=pred_test_clipped,
    pred_model2=pred_test_baseline_w_clip,
    pred_model3=pred_test_random_w_clip,
    pred_model4=pred_test_grid_w_clip,
    tolerance=1.0
)

```
