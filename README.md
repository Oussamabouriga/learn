```

# ============================================================
# PLOTS DES MÉTRIQUES (style bleu) + F1 (optionnel via classes)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ------------------------------------------------------------
# 1) Couleur "Blue 3" (ajuste si tu veux exactement ton code couleur)
# ------------------------------------------------------------
BLUE_3 = "#3B82F6"   # bleu principal
BLUE_3_DARK = "#1D4ED8"
BLUE_3_LIGHT = "#93C5FD"
GRID_BLUE = "#DBEAFE"

# ------------------------------------------------------------
# 2) Vérifier que la table des métriques test existe
#    (créée dans ton code précédent)
# ------------------------------------------------------------
# attendu: saved_metrics_all_models_test ou metrics_all_models_test
if "saved_metrics_all_models_test" in globals():
    metrics_plot_df = saved_metrics_all_models_test.copy()
else:
    metrics_plot_df = metrics_all_models_test.copy()

display(metrics_plot_df)

# ------------------------------------------------------------
# 3) Préparer un tableau "long" pour tracer les métriques de régression
# ------------------------------------------------------------
# Colonnes attendues:
# - Metric
# - Model1_BaselineWeighted_Test
# - Model2_RandomSearchWeighted_Test
# - Model3_SmallGridWeighted_Test

reg_plot_df = metrics_plot_df.copy()

# Séparer les métriques "higher is better" / "lower is better"
higher_better_metrics = ["R2", "ExplainedVariance", "Accuracy@±0.5 (%)", "Accuracy@±1.0 (%)"]
lower_better_metrics = ["MAE", "MSE", "RMSE", "MedianAE", "MaxError", "MAPE_%", "sMAPE_%"]

# ------------------------------------------------------------
# 4) Plot A — Métriques "Higher is better" (bar chart, bleu)
# ------------------------------------------------------------
plot_higher = reg_plot_df[reg_plot_df["Metric"].isin(higher_better_metrics)].copy()

# Long format
plot_higher_long = plot_higher.melt(
    id_vars=["Metric", "Better if"],
    value_vars=[
        "Model1_BaselineWeighted_Test",
        "Model2_RandomSearchWeighted_Test",
        "Model3_SmallGridWeighted_Test"
    ],
    var_name="Model",
    value_name="Value"
)

# Rename model labels for display
model_label_map = {
    "Model1_BaselineWeighted_Test": "Baseline pondéré",
    "Model2_RandomSearchWeighted_Test": "Random Search pondéré",
    "Model3_SmallGridWeighted_Test": "Small Grid pondéré",
}
plot_higher_long["Model"] = plot_higher_long["Model"].map(model_label_map)

# Create pivot for grouped bars
pivot_higher = plot_higher_long.pivot(index="Metric", columns="Model", values="Value")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
pivot_higher.plot(kind="bar", ax=ax, color=[BLUE_3_LIGHT, BLUE_3, BLUE_3_DARK], width=0.8)

ax.set_title("Comparaison des métriques (plus élevé = meilleur)", pad=18)
ax.set_xlabel("Métrique")
ax.set_ylabel("Valeur")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style (bleu)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

# Border style
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BLUE_3_LIGHT)

# Labels on bars
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if pd.notna(h):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{h:.3f}" if h < 10 else f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )

plt.xticks(rotation=20, ha="right")
plt.legend(title="Modèle", frameon=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5) Plot B — Métriques "Lower is better" (bar chart, bleu)
# ------------------------------------------------------------
plot_lower = reg_plot_df[reg_plot_df["Metric"].isin(lower_better_metrics)].copy()

plot_lower_long = plot_lower.melt(
    id_vars=["Metric", "Better if"],
    value_vars=[
        "Model1_BaselineWeighted_Test",
        "Model2_RandomSearchWeighted_Test",
        "Model3_SmallGridWeighted_Test"
    ],
    var_name="Model",
    value_name="Value"
)

plot_lower_long["Model"] = plot_lower_long["Model"].map(model_label_map)

pivot_lower = plot_lower_long.pivot(index="Metric", columns="Model", values="Value")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_lower.plot(kind="bar", ax=ax, color=[BLUE_3_LIGHT, BLUE_3, BLUE_3_DARK], width=0.8)

ax.set_title("Comparaison des métriques (plus faible = meilleur)", pad=18)
ax.set_xlabel("Métrique")
ax.set_ylabel("Valeur")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style (bleu)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

# Border style
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BLUE_3_LIGHT)

# Labels on bars
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if pd.notna(h):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{h:.3f}" if h < 100 else f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

plt.xticks(rotation=20, ha="right")
plt.legend(title="Modèle", frameon=True)
plt.tight_layout()
plt.show()


# ============================================================
# 6) OPTIONNEL — F1 / Accuracy / Precision / Recall
#    (en convertissant la régression en classes métier)
# ============================================================
# IMPORTANT:
# F1 n'est pas adapté à la régression directe.
# Ici on crée des classes à partir de la note (réelle et prédite).

# --- Vérifie que les prédictions existent
# (issues de ton code précédent)
# pred_test_baseline_w_clip
# pred_test_random_w_clip
# pred_test_grid_w_clip
# y_test

# Exemple de classes métier (adapte les seuils selon ton besoin):
# 0 à <7      = "mauvais"
# 7 à <8.5    = "neutre"
# 8.5 à 10    = "satisfait"

y_test_cls = pd.cut(
    y_test,
    bins=[-np.inf, 7, 8.5, np.inf],
    labels=["mauvais", "neutre", "satisfait"],
    right=False
)

pred_baseline_cls = pd.cut(
    pred_test_baseline_w_clip,
    bins=[-np.inf, 7, 8.5, np.inf],
    labels=["mauvais", "neutre", "satisfait"],
    right=False
)

pred_random_cls = pd.cut(
    pred_test_random_w_clip,
    bins=[-np.inf, 7, 8.5, np.inf],
    labels=["mauvais", "neutre", "satisfait"],
    right=False
)

pred_grid_cls = pd.cut(
    pred_test_grid_w_clip,
    bins=[-np.inf, 7, 8.5, np.inf],
    labels=["mauvais", "neutre", "satisfait"],
    right=False
)

# --- Metrics classification (macro = équilibré entre classes)
cls_metrics_df = pd.DataFrame({
    "Model": ["Baseline pondéré", "Random Search pondéré", "Small Grid pondéré"],
    "Accuracy": [
        accuracy_score(y_test_cls, pred_baseline_cls),
        accuracy_score(y_test_cls, pred_random_cls),
        accuracy_score(y_test_cls, pred_grid_cls),
    ],
    "Precision_macro": [
        precision_score(y_test_cls, pred_baseline_cls, average="macro", zero_division=0),
        precision_score(y_test_cls, pred_random_cls, average="macro", zero_division=0),
        precision_score(y_test_cls, pred_grid_cls, average="macro", zero_division=0),
    ],
    "Recall_macro": [
        recall_score(y_test_cls, pred_baseline_cls, average="macro", zero_division=0),
        recall_score(y_test_cls, pred_random_cls, average="macro", zero_division=0),
        recall_score(y_test_cls, pred_grid_cls, average="macro", zero_division=0),
    ],
    "F1_macro": [
        f1_score(y_test_cls, pred_baseline_cls, average="macro", zero_division=0),
        f1_score(y_test_cls, pred_random_cls, average="macro", zero_division=0),
        f1_score(y_test_cls, pred_grid_cls, average="macro", zero_division=0),
    ],
    "F1_weighted": [
        f1_score(y_test_cls, pred_baseline_cls, average="weighted", zero_division=0),
        f1_score(y_test_cls, pred_random_cls, average="weighted", zero_division=0),
        f1_score(y_test_cls, pred_grid_cls, average="weighted", zero_division=0),
    ],
})

print("=== Metrics de classification (à partir des notes regroupées) ===")
display(cls_metrics_df)

# ------------------------------------------------------------
# 7) Plot C — Accuracy / Precision / Recall / F1 (classification dérivée)
# ------------------------------------------------------------
cls_plot_long = cls_metrics_df.melt(
    id_vars=["Model"],
    value_vars=["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "F1_weighted"],
    var_name="Metric",
    value_name="Value"
)

pivot_cls = cls_plot_long.pivot(index="Metric", columns="Model", values="Value")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_cls.plot(kind="bar", ax=ax, color=[BLUE_3_LIGHT, BLUE_3, BLUE_3_DARK], width=0.8)

ax.set_title("Métriques de classification (après regroupement des notes)", pad=18)
ax.set_xlabel("Métrique")
ax.set_ylabel("Score (0 à 1)")
ax.set_ylim(0, 1.05)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style bleu (comme demandé)
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BLUE_3_LIGHT)

# Labels on bars
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if pd.notna(h):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

plt.xticks(rotation=20, ha="right")
plt.legend(title="Modèle", frameon=True)
plt.tight_layout()
plt.show()


# ============================================================
# 8) (Optionnel) Rapport détaillé par classe pour le meilleur modèle
# ============================================================
# Choisit ici le modèle que tu veux inspecter (exemple: small grid)
print("=== Classification report (Small Grid pondéré) ===")
print(classification_report(y_test_cls, pred_grid_cls, zero_division=0))
```
