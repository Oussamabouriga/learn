```
# ============================================================
# CATBOOST CLASSIFICATION — BASELINE (NO TE)
# + Train/Test metrics
# + Confusion matrix + ROC OvR (avec noms de classes)
# + Exemple (1 ligne) -> prédiction + proba
# + SHAP global (summary + bar) + SHAP local (waterfall sur l’exemple)
#
# DATA à utiliser (déjà préparées chez toi) :
#   - X_train_cat_cls_no_te, X_test_cat_cls_no_te
#   - y_train_cat_cls_no_te, y_test_cat_cls_no_te
#   - cat_cols_cb  (liste des colonnes catégorielles CatBoost)
#   - class_names_fr (dict: id -> nom FR)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import shap


# ==============================
# 1) DATA (déjà prêtes)
# ==============================
X_train_cat_cls = X_train_cat_cls_no_te.copy()
X_test_cat_cls  = X_test_cat_cls_no_te.copy()

y_train_cat_cls = pd.to_numeric(pd.Series(y_train_cat_cls_no_te), errors="coerce").astype(int).values
y_test_cat_cls  = pd.to_numeric(pd.Series(y_test_cat_cls_no_te),  errors="coerce").astype(int).values

classes_sorted = sorted(np.unique(y_train_cat_cls))
num_classes = len(classes_sorted)

# Liste des noms FR dans l’ordre des ids
class_labels_fr = [class_names_fr[c] for c in classes_sorted]

print("Train:", X_train_cat_cls.shape, y_train_cat_cls.shape)
print("Test :", X_test_cat_cls.shape,  y_test_cat_cls.shape)
print("Classes:", classes_sorted)
print("Noms classes:", class_labels_fr)
print("Cat cols:", cat_cols_cb)


# ==============================
# 2) POOLS (CatBoost)
# ==============================
train_pool_cat_cls = Pool(
    X_train_cat_cls,
    y_train_cat_cls,
    cat_features=cat_cols_cb
)

test_pool_cat_cls = Pool(
    X_test_cat_cls,
    y_test_cat_cls,
    cat_features=cat_cols_cb
)


# ==============================
# 3) HYPERPARAMS (baseline)
# Fix: bootstrap_type compatible avec subsample
# ==============================
cat_params_cls_base = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 200,

    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,

    # IMPORTANT: Bernoulli permet subsample (Bayesian ne le permet pas)
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "colsample_bylevel": 0.8,

    "random_strength": 0.5
}

model_name = "catboost_cls_baseline_no_te_v1"


# ==============================
# 4) TRAIN
# ==============================
cat_cls = CatBoostClassifier(**cat_params_cls_base)

cat_cls.fit(
    train_pool_cat_cls,
    eval_set=test_pool_cat_cls,
    use_best_model=True,
    early_stopping_rounds=150
)

print("Model trained:", model_name)


# ==============================
# 5) PRED + METRICS (train/test)
# ==============================
pred_train = cat_cls.predict(X_train_cat_cls).astype(int).reshape(-1)
pred_test  = cat_cls.predict(X_test_cat_cls).astype(int).reshape(-1)

proba_train = cat_cls.predict_proba(X_train_cat_cls)
proba_test  = cat_cls.predict_proba(X_test_cat_cls)

def metrics_cls(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

m_train = metrics_cls(y_train_cat_cls, pred_train)
m_test  = metrics_cls(y_test_cat_cls,  pred_test)

metrics_df = pd.DataFrame([
    {"Jeu": "Train", **m_train},
    {"Jeu": "Test",  **m_test},
])

print("\n=== Métriques (CatBoost classification) ===")
display(metrics_df)

print("\n=== Rapport détaillé (test) ===")
print(classification_report(
    y_test_cat_cls,
    pred_test,
    labels=classes_sorted,
    target_names=class_labels_fr,
    zero_division=0
))


# ==============================
# 6) MATRICE DE CONFUSION (test) avec noms
# ==============================
cm = confusion_matrix(y_test_cat_cls, pred_test, labels=classes_sorted)
cm_df = pd.DataFrame(
    cm,
    index=[f"Réel: {n}" for n in class_labels_fr],
    columns=[f"Prédit: {n}" for n in class_labels_fr],
)
print("\n=== Matrice de confusion (test) ===")
display(cm_df)


# ==============================
# 7) ROC OvR (test) avec noms des classes
# ==============================
y_test_bin = label_binarize(y_test_cat_cls, classes=classes_sorted)

plt.figure(figsize=(7, 6))
for i, c in enumerate(classes_sorted):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_test[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_fr[c]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8) EXEMPLE (1 ligne) -> prédiction + proba
#    -> on aligne exactement sur les colonnes de X_train_cat_cls
# ============================================================

# IMPORTANT:
# - Mets ici ton exemple complet si tu veux.
# - Si tu n’as pas une colonne, on la crée en NaN (comme d’habitude).
test_row_cat_cls_no_te = [{
    # exemples (à adapter)
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "code_postal": "59700",
    "operating_system": "Android",
    "marque": "Google",
    "model": "Pixel 7 Pro ",
    "garantie": "Dommage",
    "list_prest": "ADVANCED_SWAP",
    # ... ajoute le reste si tu veux
}]

X_one = pd.DataFrame(test_row_cat_cls_no_te).copy()

# Ajouter les colonnes manquantes + aligner l’ordre
for c in X_train_cat_cls.columns:
    if c not in X_one.columns:
        X_one[c] = np.nan
X_one = X_one[X_train_cat_cls.columns].copy()

# Nettoyage CatBoost (critique) :
# 1) pd.NA -> np.nan
X_one = X_one.astype(object).where(pd.notna(X_one), np.nan)

# 2) colonnes catégorielles -> string + normaliser les "manquants"
for c in cat_cols_cb:
    if c in X_one.columns:
        X_one[c] = X_one[c].astype(str)
        X_one.loc[X_one[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"

# 3) colonnes non-catégorielles -> numeric
num_cols = [c for c in X_train_cat_cls.columns if c not in cat_cols_cb]
for c in num_cols:
    if c in X_one.columns:
        X_one[c] = pd.to_numeric(X_one[c], errors="coerce")

# Prédiction exemple
pred_example = int(cat_cls.predict(X_one).reshape(-1)[0])
proba_example = cat_cls.predict_proba(X_one).reshape(-1)

print("\n=== EXEMPLE (classification) ===")
print("Classe prédite (id):", pred_example)
print("Classe prédite (nom):", class_names_fr.get(pred_example, str(pred_example)))

# Afficher proba par classe avec noms
proba_df = pd.DataFrame({
    "class_id": classes_sorted,
    "class_name_fr": [class_names_fr[c] for c in classes_sorted],
    "proba": [float(proba_example[i]) for i in range(num_classes)]
}).sort_values("proba", ascending=False).reset_index(drop=True)

display(proba_df)


# ============================================================
# 9) SHAP (CatBoostClassifier)
# - Global: summary + bar (sur une classe choisie)
# - Local: waterfall sur l’exemple (classe prédite)
# ============================================================

# Petit échantillon pour accélérer SHAP (modifiable)
sample_size = min(300, len(X_test_cat_cls))
X_shap = X_test_cat_cls.sample(sample_size, random_state=42).copy()

# Même nettoyage CatBoost sur X_shap (au cas où)
X_shap = X_shap.astype(object).where(pd.notna(X_shap), np.nan)
for c in cat_cols_cb:
    if c in X_shap.columns:
        X_shap[c] = X_shap[c].astype(str)
        X_shap.loc[X_shap[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
for c in num_cols:
    if c in X_shap.columns:
        X_shap[c] = pd.to_numeric(X_shap[c], errors="coerce")

# Explainer
explainer = shap.TreeExplainer(cat_cls)
shap_values = explainer.shap_values(X_shap)

# shap_values peut être:
# - list (une matrice par classe)
# - ou array 3D selon versions
def get_shap_matrix_for_class(shap_values_obj, class_index):
    if isinstance(shap_values_obj, list):
        sv = shap_values_obj[class_index]
    else:
        # si array 3D: (n_samples, n_features, n_classes) ou (n_classes, n_samples, n_features)
        arr = np.array(shap_values_obj)
        if arr.ndim == 3 and arr.shape[2] == num_classes:
            sv = arr[:, :, class_index]
        elif arr.ndim == 3 and arr.shape[0] == num_classes:
            sv = arr[class_index, :, :]
        else:
            sv = arr
    sv = np.array(sv)

    # Fix classique: parfois CatBoost/SHAP ajoute une colonne “bias”
    if sv.shape[1] == X_shap.shape[1] + 1:
        sv = sv[:, :-1]
    return sv

# Choisir la classe à afficher globalement
# (par défaut: la classe prédite par l’exemple)
class_for_global = pred_example
idx_global = classes_sorted.index(class_for_global)

sv_global = get_shap_matrix_for_class(shap_values, idx_global)

print("\nSHAP global — classe affichée:", class_names_fr.get(class_for_global, str(class_for_global)))

# Summary (beeswarm)
shap.summary_plot(sv_global, X_shap, show=True)

# Bar plot (importance globale)
shap.summary_plot(sv_global, X_shap, plot_type="bar", show=True)


# -------- SHAP local (waterfall) sur l’exemple --------
sv_one_all = explainer.shap_values(X_one)

# récupérer shap pour la classe prédite
if isinstance(sv_one_all, list):
    sv_one = np.array(sv_one_all[idx_global]).reshape(-1)
else:
    arr_one = np.array(sv_one_all)
    if arr_one.ndim == 3 and arr_one.shape[2] == num_classes:
        sv_one = arr_one[0, :, idx_global]
    elif arr_one.ndim == 3 and arr_one.shape[0] == num_classes:
        sv_one = arr_one[idx_global, 0, :]
    else:
        sv_one = arr_one.reshape(-1)

# drop bias if present
if sv_one.shape[0] == X_one.shape[1] + 1:
    sv_one = sv_one[:-1]

# base_value: parfois scalaire, parfois vecteur (multi-classe)
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val_cls = float(np.array(base_val)[idx_global])
else:
    base_val_cls = float(base_val)

print("\nSHAP local — explication de l’exemple pour la classe:", class_names_fr.get(class_for_global, str(class_for_global)))

exp = shap.Explanation(
    values=sv_one,
    base_values=base_val_cls,
    data=X_one.iloc[0],
    feature_names=X_one.columns
)

shap.plots.waterfall(exp, max_display=20)
plt.show()
```
