```

# ============================================================
# CATBOOST CLASSIFICATION — BAYESIAN OPTIMIZATION (Optuna)
# + Train/Test metrics
# + Confusion matrix + ROC OvR (noms des classes)
# + Exemple (1 ligne) -> prédiction + proba
# + SHAP global + SHAP local (exemple)
#
# DATA à utiliser (déjà préparées chez toi) :
#   - X_train_cat_cls_no_te, X_test_cat_cls_no_te
#   - y_train_cat_cls_no_te, y_test_cat_cls_no_te
#   - cat_cols_cb  (liste des colonnes catégorielles CatBoost)
#   - class_names_fr (dict: id -> nom FR)
#
# Requirements:
#   pip install optuna shap catboost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

import optuna
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
class_labels_fr = [class_names_fr[c] for c in classes_sorted]

print("Train:", X_train_cat_cls.shape, y_train_cat_cls.shape)
print("Test :", X_test_cat_cls.shape,  y_test_cat_cls.shape)
print("Classes:", classes_sorted)
print("Noms classes:", class_labels_fr)
print("Cat cols:", cat_cols_cb)


# ==============================
# 2) Optuna config
# ==============================
n_splits = 5
n_trials = 40          # 20..120 selon budget
random_state = 42

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# NOTE IMPORTANT (CatBoost):
# - bootstrap_type="Bayesian" NE SUPPORTE PAS "subsample"
# - Si tu veux subsample, utilise bootstrap_type="Bernoulli"
#
# Ici on fait un vrai "Bayesian" bootstrap => PAS de subsample dans params.


def _score_for_optuna(y_true, y_pred):
    """
    Score unique à maximiser (multi-objectifs simplifié):
    - on veut: F1_weighted haut, accuracy haut
    - et pénaliser: erreurs macro (F1_macro) faibles
    """
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")

    # Score principal (tu peux ajuster les poids)
    score = 0.55 * f1w + 0.35 * acc + 0.10 * f1m
    return float(score), float(acc), float(f1w), float(f1m)


def objective(trial):
    params = {
        # Core
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": random_state,
        "allow_writing_files": False,
        "verbose": False,

        # Bayesian bootstrap (pas de subsample)
        "bootstrap_type": "Bayesian",
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),

        # Main tunables
        "iterations": trial.suggest_int("iterations", 600, 2400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),

        # Regularization / randomness
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),  # colsample-like in CatBoost

        # Tree growth
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
    }

    fold_scores = []

    for tr_idx, va_idx in cv.split(X_train_cat_cls, y_train_cat_cls):
        X_tr = X_train_cat_cls.iloc[tr_idx]
        y_tr = y_train_cat_cls[tr_idx]

        X_va = X_train_cat_cls.iloc[va_idx]
        y_va = y_train_cat_cls[va_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_tr, y_tr,
            cat_features=cat_cols_cb,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=150
        )

        pred_va = model.predict(X_va).astype(int).reshape(-1)
        score, acc, f1w, f1m = _score_for_optuna(y_va, pred_va)

        fold_scores.append(score)

    return float(np.mean(fold_scores))  # maximize


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_params
best_cv_score = study.best_value

print("\n--- Optuna terminé ---")
print("Best CV score (custom):", best_cv_score)
print("Best params:", best_params)


# ==============================
# 3) Train final best model (sur TRAIN complet)
# ==============================
final_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": random_state,
    "allow_writing_files": False,
    "verbose": 200,

    # IMPORTANT: doit matcher l'objective
    "bootstrap_type": "Bayesian",
    **best_params
}

cat_cls_bayes = CatBoostClassifier(**final_params)

cat_cls_bayes.fit(
    X_train_cat_cls, y_train_cat_cls,
    cat_features=cat_cols_cb,
    eval_set=(X_test_cat_cls, y_test_cat_cls),
    use_best_model=True,
    early_stopping_rounds=150
)

model_name = "catboost_cls_optuna_bayesian_no_te_v1"
print("Model trained:", model_name)


# ==============================
# 4) Predict + Metrics (train/test)
# ==============================
pred_train = cat_cls_bayes.predict(X_train_cat_cls).astype(int).reshape(-1)
pred_test  = cat_cls_bayes.predict(X_test_cat_cls).astype(int).reshape(-1)

proba_train = cat_cls_bayes.predict_proba(X_train_cat_cls)
proba_test  = cat_cls_bayes.predict_proba(X_test_cat_cls)

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

print("\n=== Métriques (CatBoost classification — Optuna Bayesian) ===")
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
# 5) Confusion matrix (test) avec noms
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
# 6) ROC OvR (test) avec noms
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
# 7) EXEMPLE (1 ligne) -> prédiction + proba
# ============================================================
test_row_cat_cls_no_te = [{
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "code_postal": "59700",
    "operating_system": "Android",
    "marque": "Google",
    "model": "Pixel 7 Pro ",
    "garantie": "Dommage",
    "list_prest": "ADVANCED_SWAP",
}]

X_one = pd.DataFrame(test_row_cat_cls_no_te).copy()

# ajouter colonnes manquantes + aligner ordre
for c in X_train_cat_cls.columns:
    if c not in X_one.columns:
        X_one[c] = np.nan
X_one = X_one[X_train_cat_cls.columns].copy()

# nettoyage CatBoost (critique)
X_one = X_one.astype(object).where(pd.notna(X_one), np.nan)

# cat cols -> string + normalize missing-like
for c in cat_cols_cb:
    if c in X_one.columns:
        X_one[c] = X_one[c].astype(str)
        X_one.loc[X_one[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"

# non-cat -> numeric
num_cols = [c for c in X_train_cat_cls.columns if c not in cat_cols_cb]
for c in num_cols:
    X_one[c] = pd.to_numeric(X_one[c], errors="coerce")

pred_example = int(cat_cls_bayes.predict(X_one).reshape(-1)[0])
proba_example = cat_cls_bayes.predict_proba(X_one).reshape(-1)

print("\n=== EXEMPLE (classification) ===")
print("Classe prédite (id):", pred_example)
print("Classe prédite (nom):", class_names_fr.get(pred_example, str(pred_example)))

proba_df = pd.DataFrame({
    "class_id": classes_sorted,
    "class_name_fr": [class_names_fr[c] for c in classes_sorted],
    "proba": [float(proba_example[i]) for i in range(num_classes)]
}).sort_values("proba", ascending=False).reset_index(drop=True)

display(proba_df)


# ============================================================
# 8) SHAP (Global + Local)
# ============================================================

sample_size = min(300, len(X_test_cat_cls))
X_shap = X_test_cat_cls.sample(sample_size, random_state=42).copy()

# même nettoyage que X_one
X_shap = X_shap.astype(object).where(pd.notna(X_shap), np.nan)
for c in cat_cols_cb:
    if c in X_shap.columns:
        X_shap[c] = X_shap[c].astype(str)
        X_shap.loc[X_shap[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
for c in num_cols:
    X_shap[c] = pd.to_numeric(X_shap[c], errors="coerce")

explainer = shap.TreeExplainer(cat_cls_bayes)
shap_values = explainer.shap_values(X_shap)

def get_shap_matrix_for_class(shap_values_obj, class_index, X_ref):
    if isinstance(shap_values_obj, list):
        sv = shap_values_obj[class_index]
    else:
        arr = np.array(shap_values_obj)
        if arr.ndim == 3 and arr.shape[2] == len(classes_sorted):
            sv = arr[:, :, class_index]
        elif arr.ndim == 3 and arr.shape[0] == len(classes_sorted):
            sv = arr[class_index, :, :]
        else:
            sv = arr
    sv = np.array(sv)
    if sv.shape[1] == X_ref.shape[1] + 1:
        sv = sv[:, :-1]
    return sv

# Global SHAP: on affiche la classe prédite sur l’exemple
class_for_global = pred_example
idx_global = classes_sorted.index(class_for_global)
sv_global = get_shap_matrix_for_class(shap_values, idx_global, X_shap)

print("\nSHAP global — classe affichée:", class_names_fr.get(class_for_global, str(class_for_global)))
shap.summary_plot(sv_global, X_shap, show=True)
shap.summary_plot(sv_global, X_shap, plot_type="bar", show=True)

# Local SHAP: waterfall sur l’exemple
sv_one_all = explainer.shap_values(X_one)
if isinstance(sv_one_all, list):
    sv_one = np.array(sv_one_all[idx_global]).reshape(-1)
else:
    arr_one = np.array(sv_one_all)
    if arr_one.ndim == 3 and arr_one.shape[2] == len(classes_sorted):
        sv_one = arr_one[0, :, idx_global]
    elif arr_one.ndim == 3 and arr_one.shape[0] == len(classes_sorted):
        sv_one = arr_one[idx_global, 0, :]
    else:
        sv_one = arr_one.reshape(-1)

if sv_one.shape[0] == X_one.shape[1] + 1:
    sv_one = sv_one[:-1]

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
