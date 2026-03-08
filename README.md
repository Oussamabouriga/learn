```
# ============================================================
# CATBOOST CLASSIFICATION — BAYESIAN OPTIMIZATION (Optuna)
# (same full process)
# - Optuna multi-metric (minimize logloss AND maximize f1_macro + accuracy)
# - Uses StratifiedKFold CV on TRAIN only
# - Train final best model (early stopping)
# - Metrics train/test + confusion matrix + ROC OvR
# - SHAP global + SHAP example (1 row)
# - Save to models/catboost/classification/<model_name>/
#
# Uses your existing prepared data:
#   X_train_cat_cls_no_te, X_test_cat_cls_no_te
#   y_train_cat_cls_no_te, y_test_cat_cls_no_te
#   cat_cols_cb
#   class_names_fr (dict: id -> label)
#
# Requirements:
#   pip install catboost optuna shap scikit-learn
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import optuna
import shap


# ==============================
# 0) Inputs + checks
# ==============================
assert "X_train_cat_cls_no_te" in globals() and "X_test_cat_cls_no_te" in globals()
assert "y_train_cat_cls_no_te" in globals() and "y_test_cat_cls_no_te" in globals()
assert "cat_cols_cb" in globals()
assert "class_names_fr" in globals()

X_train_bayes = X_train_cat_cls_no_te.copy()
X_test_bayes  = X_test_cat_cls_no_te.copy()

y_train_bayes = pd.to_numeric(pd.Series(y_train_cat_cls_no_te), errors="coerce").astype(int).values
y_test_bayes  = pd.to_numeric(pd.Series(y_test_cat_cls_no_te),  errors="coerce").astype(int).values

classes_sorted = sorted(np.unique(y_train_bayes))
num_classes = len(classes_sorted)
class_labels_fr = [class_names_fr[c] for c in classes_sorted]

print("Train:", X_train_bayes.shape, y_train_bayes.shape)
print("Test :", X_test_bayes.shape,  y_test_bayes.shape)
print("Classes:", classes_sorted)


# ==============================
# 1) CV config + Optuna config
# ==============================
n_splits = 5
random_state = 42
n_trials = 40  # change 20..150 depending on compute
timeout_sec = None  # or set like 3600

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ==============================
# 2) Objective (multi-metric)
#    - We track 3 objectives:
#       1) minimize logloss (good probability quality)
#       2) maximize f1_macro (balanced across classes)
#       3) maximize accuracy (simple business metric)
# ==============================
def objective(trial):
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",  # multi-class logloss-like metric
        "random_seed": random_state,
        "allow_writing_files": False,
        "verbose": False,

        # Main tuned params
        "iterations": trial.suggest_int("iterations", 600, 2600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),

        # Randomness / regularization
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),

        # Sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }

    fold_logloss = []
    fold_f1 = []
    fold_acc = []

    for tr_idx, va_idx in cv.split(X_train_bayes, y_train_bayes):
        X_tr = X_train_bayes.iloc[tr_idx].copy()
        y_tr = y_train_bayes[tr_idx].copy()
        X_va = X_train_bayes.iloc[va_idx].copy()
        y_va = y_train_bayes[va_idx].copy()

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols_cb)
        val_pool   = Pool(X_va, y_va, cat_features=cat_cols_cb)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=150)

        # Predict on validation
        pred_va = model.predict(X_va).astype(int).reshape(-1)
        proba_va = model.predict_proba(X_va)

        # Metrics
        acc = float(accuracy_score(y_va, pred_va))
        f1m = float(f1_score(y_va, pred_va, average="macro"))

        # CatBoost can provide its eval metric value at best iteration:
        # easiest robust way: compute multiclass logloss manually
        eps = 1e-12
        proba_va = np.clip(proba_va, eps, 1 - eps)
        # logloss = mean(-log(p_true))
        p_true = proba_va[np.arange(len(y_va)), y_va]
        logloss = float(np.mean(-np.log(p_true)))

        fold_acc.append(acc)
        fold_f1.append(f1m)
        fold_logloss.append(logloss)

    return float(np.mean(fold_logloss)), float(np.mean(fold_f1)), float(np.mean(fold_acc))


study = optuna.create_study(
    directions=["minimize", "maximize", "maximize"]  # logloss, f1_macro, accuracy
)

study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

best_trial = study.best_trials[0]  # “best” is Pareto-optimal; Optuna picks one representative

print("\n=== Optuna done ===")
print("Best trial number:", best_trial.number)
print("Best values [logloss, f1_macro, accuracy]:", best_trial.values)
print("Best params:", best_trial.params)


# ==============================
# 3) Train final model on full train (best params)
# ==============================
model_name = "catboost_cls_bayes_optuna_no_te_v1"

best_params = dict(best_trial.params)

cat_cls_bayes_best = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=random_state,
    allow_writing_files=False,
    verbose=200,
    **best_params
)

train_pool_full = Pool(X_train_bayes, y_train_bayes, cat_features=cat_cols_cb)
test_pool_full  = Pool(X_test_bayes,  y_test_bayes,  cat_features=cat_cols_cb)

cat_cls_bayes_best.fit(
    train_pool_full,
    eval_set=test_pool_full,
    use_best_model=True,
    early_stopping_rounds=150
)

print("Final model trained:", model_name)


# ==============================
# 4) Predict + metrics (train/test)
# ==============================
pred_train = cat_cls_bayes_best.predict(X_train_bayes).astype(int).reshape(-1)
pred_test  = cat_cls_bayes_best.predict(X_test_bayes).astype(int).reshape(-1)

proba_train = cat_cls_bayes_best.predict_proba(X_train_bayes)
proba_test  = cat_cls_bayes_best.predict_proba(X_test_bayes)

def metrics_classification(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

m_train = metrics_classification(y_train_bayes, pred_train)
m_test  = metrics_classification(y_test_bayes,  pred_test)

metrics_df = pd.DataFrame([
    {"Jeu": "Train", **m_train},
    {"Jeu": "Test",  **m_test},
])

print("\n=== Métriques (CatBoost classification — Bayesian Optuna) ===")
display(metrics_df)

print("\n=== Rapport détaillé (test) ===")
print(classification_report(
    y_test_bayes,
    pred_test,
    labels=classes_sorted,
    target_names=class_labels_fr,
    zero_division=0
))


# ==============================
# 5) Confusion matrix (test) with French labels
# ==============================
cm = confusion_matrix(y_test_bayes, pred_test, labels=classes_sorted)
cm_df = pd.DataFrame(cm, index=[f"Réel: {n}" for n in class_labels_fr], columns=[f"Prédit: {n}" for n in class_labels_fr])

print("\n=== Matrice de confusion (test) ===")
display(cm_df)


# ==============================
# 6) ROC curves OvR (multi-class)
# ==============================
y_test_bin = label_binarize(y_test_bayes, classes=classes_sorted)

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


# ==============================
# 7) SHAP (global + example)
# ==============================
# For classification, SHAP returns per-class values.
# We'll show SHAP for the predicted class of the first test row.

sample_size = min(300, len(X_test_bayes))
X_shap = X_test_bayes.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_cls_bayes_best)
shap_values_all = explainer.shap_values(X_shap)

# Example row
X_one = X_test_bayes.iloc[[0]].copy()
pred_one = int(cat_cls_bayes_best.predict(X_one)[0])
proba_one = cat_cls_bayes_best.predict_proba(X_one)[0]

print("\n=== Exemple (1 ligne) ===")
print("Classe prédite:", pred_one, "->", class_names_fr[pred_one])
print("Probabilités par classe:")
for c in classes_sorted:
    print(f" - {class_names_fr[c]}: {proba_one[c]:.4f}")

# Global SHAP for that class
print("\nSHAP global (classe):", pred_one, "->", class_names_fr[pred_one])

if isinstance(shap_values_all, list):
    shap_vals_for_class = shap_values_all[pred_one]
else:
    shap_vals_for_class = shap_values_all[:, :, pred_one]

shap.summary_plot(shap_vals_for_class, X_shap, show=True)
shap.summary_plot(shap_vals_for_class, X_shap, plot_type="bar", show=True)

# Local SHAP on the same example row
shap_one_all = explainer.shap_values(X_one)
if isinstance(shap_one_all, list):
    shap_one = shap_one_all[pred_one]
    base_val = explainer.expected_value[pred_one] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    shap_one = shap_one_all[:, :, pred_one]
    base_val = explainer.expected_value[pred_one] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

shap.plots.waterfall(
    shap.Explanation(
        values=shap_one[0],
        base_values=base_val,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()


# ==============================
# 8) Save model + artifacts
# ==============================
save_dir = os.path.join("models", "catboost", "classification", model_name)
os.makedirs(save_dir, exist_ok=True)

cat_cls_bayes_best.save_model(os.path.join(save_dir, "model.cbm"))
metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)
cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"))

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

with open(os.path.join(save_dir, "best_trial_values.json"), "w") as f:
    json.dump(
        {"logloss_cv": float(best_trial.values[0]), "f1_macro_cv": float(best_trial.values[1]), "accuracy_cv": float(best_trial.values[2])},
        f, indent=2
    )

with open(os.path.join(save_dir, "classes_fr.json"), "w") as f:
    json.dump({int(k): v for k, v in class_names_fr.items()}, f, indent=2, ensure_ascii=False)

print("Saved to:", save_dir)

```
