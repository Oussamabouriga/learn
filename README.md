```
# ============================================================
# AFFICHAGE COMPARATIF DES MÉTRIQUES (MAE / RMSE) + TABLES
# - Français
# - Couleurs (style bleu)
# - Editable
# - Table finale Accuracy Train/Test recalculée avec tabulate
# ============================================================

# ------------------------------------------------------------
# 0) Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabulate import tabulate

# ------------------------------------------------------------
# 1) CONFIGURATION (editable)
# ------------------------------------------------------------
# === Couleurs (style bleu) ===
BLUE_3 = "#3B82F6"
BLUE_3_DARK = "#1D4ED8"
BLUE_3_LIGHT = "#93C5FD"
GRAY_DARK = "#64748B"
GRID_BLUE = "#DBEAFE"
BORDER_BLUE = "#93C5FD"

# === Noms des modèles (modifiable) ===
model_names = [
    "Modèle 1 - Baseline",
    "Modèle 2 - Avec poids",
    "Modèle 3 - Random Search + poids",
    "Modèle 4 - Small Grid + poids",
]

# === Tolérances pour l'accuracy métier (modifiable) ===
tolerances = [0.5, 1.0]

# ------------------------------------------------------------
# 2) Récupération des prédictions (adapte si tes noms diffèrent)
# ------------------------------------------------------------
# ===== TEST predictions =====
# Modèle 1 (baseline normal)
if "pred_test_clipped" in globals():
    pred_test_model1 = np.asarray(pred_test_clipped, dtype=float)
elif "baseline_pred_test" in globals():
    pred_test_model1 = np.asarray(baseline_pred_test, dtype=float)
else:
    raise NameError("Prédictions TEST du Modèle 1 introuvables (pred_test_clipped ou baseline_pred_test).")

# Modèle 2 (weighted)
if "pred_test_baseline_w_clip" in globals():
    pred_test_model2 = np.asarray(pred_test_baseline_w_clip, dtype=float)
else:
    raise NameError("Prédictions TEST du Modèle 2 introuvables (pred_test_baseline_w_clip).")

# Modèle 3 (random search weighted)
if "pred_test_random_w_clip" in globals():
    pred_test_model3 = np.asarray(pred_test_random_w_clip, dtype=float)
else:
    raise NameError("Prédictions TEST du Modèle 3 introuvables (pred_test_random_w_clip).")

# Modèle 4 (small grid weighted)
if "pred_test_grid_w_clip" in globals():
    pred_test_model4 = np.asarray(pred_test_grid_w_clip, dtype=float)
else:
    raise NameError("Prédictions TEST du Modèle 4 introuvables (pred_test_grid_w_clip).")

# ===== TRAIN predictions =====
# Modèle 1 (baseline normal)
if "pred_train_clipped" in globals():
    pred_train_model1 = np.asarray(pred_train_clipped, dtype=float)
elif "baseline_pred_train" in globals():
    pred_train_model1 = np.asarray(baseline_pred_train, dtype=float)
else:
    raise NameError("Prédictions TRAIN du Modèle 1 introuvables (pred_train_clipped ou baseline_pred_train).")

# Modèle 2 (weighted)
if "pred_train_baseline_w_clip" in globals():
    pred_train_model2 = np.asarray(pred_train_baseline_w_clip, dtype=float)
else:
    raise NameError("Prédictions TRAIN du Modèle 2 introuvables (pred_train_baseline_w_clip).")

# Modèle 3 (random search weighted)
if "pred_train_random_w_clip" in globals():
    pred_train_model3 = np.asarray(pred_train_random_w_clip, dtype=float)
else:
    raise NameError("Prédictions TRAIN du Modèle 3 introuvables (pred_train_random_w_clip).")

# Modèle 4 (small grid weighted)
if "pred_train_grid_w_clip" in globals():
    pred_train_model4 = np.asarray(pred_train_grid_w_clip, dtype=float)
else:
    raise NameError("Prédictions TRAIN du Modèle 4 introuvables (pred_train_grid_w_clip).")

# ===== y true =====
y_test_arr = np.asarray(y_test, dtype=float)
y_train_arr = np.asarray(y_train, dtype=float)

# Vérification rapide
assert len(y_test_arr) == len(pred_test_model1) == len(pred_test_model2) == len(pred_test_model3) == len(pred_test_model4)
assert len(y_train_arr) == len(pred_train_model1) == len(pred_train_model2) == len(pred_train_model3) == len(pred_train_model4)

# ------------------------------------------------------------
# 3) Calcul des métriques MAE / RMSE (Train + Test)
# ------------------------------------------------------------
preds_test = [pred_test_model1, pred_test_model2, pred_test_model3, pred_test_model4]
preds_train = [pred_train_model1, pred_train_model2, pred_train_model3, pred_train_model4]

rows_metrics = []
for name, p_train, p_test in zip(model_names, preds_train, preds_test):
    mae_train = mean_absolute_error(y_train_arr, p_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_arr, p_train))

    mae_test = mean_absolute_error(y_test_arr, p_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_arr, p_test))

    rows_metrics.append({
        "Modèle": name,
        "MAE Train": mae_train,
        "MAE Test": mae_test,
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
    })

metrics_df = pd.DataFrame(rows_metrics)

# ------------------------------------------------------------
# 4) Tableau explicatif des métriques (éditable, en français)
# ------------------------------------------------------------
explication_metrics_df = pd.DataFrame([
    {
        "Métrique": "MAE",
        "Ce que ça mesure": "Erreur absolue moyenne (en points de note, même unité que la cible 0–10)",
        "Comment l'expliquer": "En moyenne, le modèle se trompe de X points",
        "Pourquoi c'est utile": "Très simple à comprendre, même pour un public non technique",
        "Meilleur si": "Plus faible",
    },
    {
        "Métrique": "RMSE",
        "Ce que ça mesure": "Erreur quadratique moyenne (racine), en points de note",
        "Comment l'expliquer": "Augmente plus fortement quand le modèle fait de grosses erreurs",
        "Pourquoi c'est utile": "Permet de voir la sensibilité aux grosses erreurs",
        "Meilleur si": "Plus faible",
    },
])

print("\n" + "="*100)
print("EXPLICATION DES MÉTRIQUES À MONTRER (MAE / RMSE)")
print("="*100)
print(tabulate(explication_metrics_df, headers="keys", tablefmt="fancy_grid", showindex=False))

# ------------------------------------------------------------
# 5) Tableau de comparaison MAE / RMSE (Train/Test)
# ------------------------------------------------------------
metrics_df_display = metrics_df.copy()
for col in ["MAE Train", "MAE Test", "RMSE Train", "RMSE Test"]:
    metrics_df_display[col] = metrics_df_display[col].map(lambda x: round(float(x), 4))

print("\n" + "="*100)
print("COMPARAISON DES MÉTRIQUES (TRAIN / TEST)")
print("="*100)
print(tabulate(metrics_df_display, headers="keys", tablefmt="fancy_grid", showindex=False))

# ------------------------------------------------------------
# 6) Plot 1 — MAE Test et RMSE Test (grouped bars)
# ------------------------------------------------------------
plot_test_df = metrics_df[["Modèle", "MAE Test", "RMSE Test"]].copy()
plot_test_long = plot_test_df.melt(
    id_vars=["Modèle"],
    value_vars=["MAE Test", "RMSE Test"],
    var_name="Métrique",
    value_name="Valeur"
)

pivot_test = plot_test_long.pivot(index="Modèle", columns="Métrique", values="Valeur")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_test.plot(kind="bar", ax=ax, width=0.8, color=[BLUE_3_LIGHT, BLUE_3_DARK])

ax.set_title("Comparaison des erreurs sur TEST (MAE / RMSE)", pad=18)
ax.set_xlabel("Modèle")
ax.set_ylabel("Erreur (points de note)")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Grid style bleu
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

# Bordure
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BORDER_BLUE)

# Labels sur les barres
for container in ax.containers:
    for b in container:
        h = b.get_height()
        if pd.notna(h):
            ax.text(
                b.get_x() + b.get_width()/2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.xticks(rotation=15, ha="right")
plt.legend(title="Métrique", frameon=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7) Plot 2 — MAE Train vs Test (pour voir la généralisation)
# ------------------------------------------------------------
plot_mae_df = metrics_df[["Modèle", "MAE Train", "MAE Test"]].copy()
plot_mae_long = plot_mae_df.melt(
    id_vars=["Modèle"],
    value_vars=["MAE Train", "MAE Test"],
    var_name="Jeu",
    value_name="Valeur"
)
pivot_mae = plot_mae_long.pivot(index="Modèle", columns="Jeu", values="Valeur")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_mae.plot(kind="bar", ax=ax, width=0.8, color=[BLUE_3_LIGHT, BLUE_3])

ax.set_title("MAE Train vs Test (généralisation)", pad=18)
ax.set_xlabel("Modèle")
ax.set_ylabel("MAE (points de note)")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BORDER_BLUE)

for container in ax.containers:
    for b in container:
        h = b.get_height()
        if pd.notna(h):
            ax.text(
                b.get_x() + b.get_width()/2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.xticks(rotation=15, ha="right")
plt.legend(title="Jeu", frameon=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8) Plot 3 — RMSE Train vs Test (généralisation + grosses erreurs)
# ------------------------------------------------------------
plot_rmse_df = metrics_df[["Modèle", "RMSE Train", "RMSE Test"]].copy()
plot_rmse_long = plot_rmse_df.melt(
    id_vars=["Modèle"],
    value_vars=["RMSE Train", "RMSE Test"],
    var_name="Jeu",
    value_name="Valeur"
)
pivot_rmse = plot_rmse_long.pivot(index="Modèle", columns="Jeu", values="Valeur")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_rmse.plot(kind="bar", ax=ax, width=0.8, color=[BLUE_3_LIGHT, BLUE_3_DARK])

ax.set_title("RMSE Train vs Test (sensibilité aux grosses erreurs)", pad=18)
ax.set_xlabel("Modèle")
ax.set_ylabel("RMSE (points de note)")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BORDER_BLUE)

for container in ax.containers:
    for b in container:
        h = b.get_height()
        if pd.notna(h):
            ax.text(
                b.get_x() + b.get_width()/2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.xticks(rotation=15, ha="right")
plt.legend(title="Jeu", frameon=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9) Recalcul Accuracy Train/Test (tabulate) pour plusieurs tolérances
# ------------------------------------------------------------
# Accuracy (régression) = % des prédictions avec |erreur| <= tolérance
acc_rows = []
for tol in tolerances:
    for name, p_train, p_test in zip(model_names, preds_train, preds_test):
        acc_train = (np.abs(y_train_arr - p_train) <= tol).mean() * 100
        acc_test = (np.abs(y_test_arr - p_test) <= tol).mean() * 100

        acc_rows.append({
            "Tolérance (écart)": f"±{tol}",
            "Modèle": name,
            "Précision Train (%)": round(float(acc_train), 2),
            "Précision Test (%)": round(float(acc_test), 2),
        })

accuracy_train_test_df = pd.DataFrame(acc_rows)

print("\n" + "="*100)
print("TABLEAU FINAL — PRÉCISION (ACCURACY) TRAIN / TEST RECALCULÉE")
print("="*100)
print(tabulate(accuracy_train_test_df, headers="keys", tablefmt="fancy_grid", showindex=False))

# ------------------------------------------------------------
# 10) (Optionnel) Plot accuracy TEST pour une seule tolérance (modifiable)
# ------------------------------------------------------------
tol_plot = 0.5  # <-- change ici (ex: 0.5 ou 1.0)

acc_plot_df = accuracy_train_test_df[
    accuracy_train_test_df["Tolérance (écart)"] == f"±{tol_plot}"
][["Modèle", "Précision Test (%)"]].copy()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    acc_plot_df["Modèle"],
    acc_plot_df["Précision Test (%)"],
    color=[BLUE_3_LIGHT, BLUE_3, BLUE_3_DARK, GRAY_DARK],
    width=0.7
)

ax.set_title(f"Précision sur TEST (tolérance = écart de ±{tol_plot} point)", pad=18)
ax.set_xlabel("Modèle")
ax.set_ylabel("Précision (%)")
ax.set_ylim(0, 100)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_BLUE, alpha=1.0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(BORDER_BLUE)

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


```
