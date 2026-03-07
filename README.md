```
# ============================================================
# XGBoost Classification — BAYESIAN OPTIMIZATION (Optuna)
# Optimize MULTIPLE metrics (CV):
#   - maximize F1_macro_CV
#   - maximize Accuracy_CV
#   - minimize LogLoss_CV
#
# Then:
#   - pick best trial (by F1_macro_CV by default)
#   - train final model on full train
#   - evaluate Train/Test (accuracy, f1, precision/recall, logloss)
#   - ROC curves OvR (test) with class names
#   - Confusion matrix (test)
#   - SHAP global + SHAP local (example row)
#   - Save model: models/xgboost/classification/<model_name>/
#
# Uses SAME DATA (no target encoding):
#   X_train_xgb_cls_no_te, X_test_xgb_cls_no_te
#   y_train_xgb_cls_no_te, y_test_xgb_cls_no_te
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, log_loss,
    roc_curve, auc
)

import optuna
import shap


# ==============================
# 1) Data (same as before)
# ==============================
X_train_cls = X_train_xgb_cls_no_te.copy().astype(float)
X_test_cls  = X_test_xgb_cls_no_te.copy().astype(float)

y_train_cls = pd.to_numeric(y_train_xgb_cls_no_te, errors="coerce").astype(int)
y_test_cls  = pd.to_numeric(y_test_xgb_cls_no_te,  errors="coerce").astype(int)

classes_sorted = sorted(np.unique(y_train_cls))
num_classes = int(len(classes_sorted))

print("Train:", X_train_cls.shape, y_train_cls.shape)
print("Test :", X_test_cls.shape,  y_test_cls.shape)
print("Classes:", classes_sorted)

# (Optional) ensure classes are 0..K-1
# If you already have 0..4, keep it. Otherwise remap.
need_remap = (classes_sorted != list(range(num_classes)))
if need_remap:
    class_map = {c:i for i, c in enumerate(classes_sorted)}
    y_train_cls = y_train_cls.map(class_map).astype(int)
    y_test_cls  = y_test_cls.map(class_map).astype(int)
    classes_sorted = sorted(np.unique(y_train_cls))
    print("Remapped classes to:", classes_sorted)

# Class names (edit)
class_names = [
    "Extrêmement mauvais (0–2)",
    "Mauvais (3–6)",
    "Neutre (7–8)",
    "Bien (9)",
    "Très bien (10)",
]
if len(class_names) != num_classes:
    class_names = [f"Classe {i}" for i in range(num_classes)]


# ==============================
# 2) Optuna config (editable)
# ==============================
n_splits = 5
n_trials = 60           # increase if you have more compute (e.g. 100-300)
timeout_sec = None      # or set e.g. 3600
random_state = 42

# Which metric decides "best" trial for final training?
# Options: "f1_macro", "accuracy", "logloss"
best_metric = "f1_macro"


# ==============================
# 3) CV helper: compute multiple metrics per fold
# ==============================
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def cv_multi_metrics(params):
    f1s = []
    accs = []
    lls = []

    for tr_idx, va_idx in cv.split(X_train_cls, y_train_cls):
        X_tr, X_va = X_train_cls.iloc[tr_idx], X_train_cls.iloc[va_idx]
        y_tr, y_va = y_train_cls.iloc[tr_idx], y_train_cls.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        pred_va = model.predict(X_va)
        proba_va = model.predict_proba(X_va)

        f1s.append(f1_score(y_va, pred_va, average="macro"))
        accs.append(accuracy_score(y_va, pred_va))
        lls.append(log_loss(y_va, proba_va, labels=list(range(num_classes))))

    return float(np.mean(f1s)), float(np.mean(accs)), float(np.mean(lls))


# ==============================
# 4) Optuna objective (MULTI-OBJECTIVE)
#    We return:
#      (maximize f1_macro, maximize accuracy, minimize logloss)
# ==============================
def objective(trial):
    # Suggest params (keep reasonable ranges)
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,

        # Core
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),

        # Regularization / split control
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),

        # Sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    f1_cv, acc_cv, logloss_cv = cv_multi_metrics(params)

    # Optuna expects all objectives in the order of directions below
    return f1_cv, acc_cv, logloss_cv


study = optuna.create_study(
    directions=["maximize", "maximize", "minimize"]
)
study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

print("\nOptuna finished.")
print("Number of trials:", len(study.trials))


# ==============================
# 5) Pick best trial (by YOUR chosen metric)
#    Multi-objective => Pareto front.
#    We select one "best" trial using a simple rule.
# ==============================
def pick_best_trial(study, best_metric="f1_macro"):
    # values = (f1, acc, logloss)
    pareto = study.best_trials

    if best_metric == "f1_macro":
        return max(pareto, key=lambda t: t.values[0])
    if best_metric == "accuracy":
        return max(pareto, key=lambda t: t.values[1])
    if best_metric == "logloss":
        return min(pareto, key=lambda t: t.values[2])
    return max(pareto, key=lambda t: t.values[0])

best_trial = pick_best_trial(study, best_metric=best_metric)

best_params = best_trial.params
best_f1_cv, best_acc_cv, best_ll_cv = best_trial.values

print("\nChosen best trial metric:", best_metric)
print("Best CV F1_macro:", best_f1_cv)
print("Best CV Accuracy:", best_acc_cv)
print("Best CV LogLoss:", best_ll_cv)
print("Best params:", best_params)


# ==============================
# 6) Train final model on full train
# ==============================
final_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": random_state,
    **best_params
}

xgb_cls_bayes = XGBClassifier(**final_params)
xgb_cls_bayes.fit(X_train_cls, y_train_cls)

pred_train = xgb_cls_bayes.predict(X_train_cls)
pred_test  = xgb_cls_bayes.predict(X_test_cls)

proba_train = xgb_cls_bayes.predict_proba(X_train_cls)
proba_test  = xgb_cls_bayes.predict_proba(X_test_cls)


# ==============================
# 7) Metrics (train/test)
# ==============================
def metrics_table(y_true, y_pred, y_proba, split):
    return {
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "LogLoss": log_loss(y_true, y_proba, labels=list(range(num_classes))),
    }

df_metrics = pd.DataFrame([
    metrics_table(y_train_cls, pred_train, proba_train, "Train"),
    metrics_table(y_test_cls,  pred_test,  proba_test,  "Test"),
])

print("\nMetrics — XGBoost Bayesian (Optuna)")
display(df_metrics)

print("\nRapport de classification (test):")
print(classification_report(y_test_cls, pred_test, target_names=class_names, zero_division=0))


# ==============================
# 8) ROC curves OvR (test) with class names
# ==============================
plt.figure(figsize=(7, 6))
for c in range(num_classes):
    y_true_bin = (y_test_cls.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test (Optuna)")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 9) Confusion matrix (test) with class names
# ==============================
cm = confusion_matrix(y_test_cls, pred_test, labels=list(range(num_classes)))
cm_df = pd.DataFrame(
    cm,
    index=[f"Réel — {name}" for name in class_names],
    columns=[f"Prédit — {name}" for name in class_names],
)
print("\nMatrice de confusion (test)")
display(cm_df)


# ==============================
# 10) SHAP global + local (example row)
#    Fixes common shape mismatch:
#    - if SHAP returns (n, p+1) we drop last column
# ==============================
sample_size = min(300, len(X_test_cls))
X_shap = X_test_cls.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(xgb_cls_bayes)
shap_vals = explainer.shap_values(X_shap)

# Normalize shap to list by class
if isinstance(shap_vals, list):
    shap_list = shap_vals
else:
    # shap might be (n, p, K)
    if shap_vals.ndim == 3:
        shap_list = [shap_vals[:, :, k] for k in range(shap_vals.shape[2])]
    else:
        shap_list = [shap_vals]

class_for_global = 0
sv = shap_list[class_for_global]
if sv.shape[1] == X_shap.shape[1] + 1:
    sv = sv[:, :-1]

print("\nSHAP global — classe:", class_for_global, "-", class_names[class_for_global])
shap.summary_plot(sv, X_shap, show=True)
shap.summary_plot(sv, X_shap, plot_type="bar", show=True)


# ---- SHAP local on your example row
X_example = None
try:
    # You should already have a prepared example dataframe for classification
    # with SAME columns as X_train_cls (encoded no_te)
    X_example = X_example_xgb_cls_no_te.copy().astype(float)  # change variable name if needed
except Exception:
    pass

if X_example is None:
    print("\nNo example row dataframe found. Set: X_example = <your_example_df> with same columns.")
else:
    X_example = X_example.reindex(columns=X_train_cls.columns, fill_value=0.0)

    proba_ex = xgb_cls_bayes.predict_proba(X_example)[0]
    pred_ex = int(np.argmax(proba_ex))

    print("\nExample prediction:")
    print("Classe prédite:", pred_ex, "-", class_names[pred_ex])
    for i, p in enumerate(proba_ex):
        print(f"  {i} - {class_names[i]} : {p:.4f}")

    shap_one = explainer.shap_values(X_example)
    if isinstance(shap_one, list):
        shap_one_list = shap_one
    else:
        if shap_one.ndim == 3:
            shap_one_list = [shap_one[:, :, k] for k in range(shap_one.shape[2])]
        else:
            shap_one_list = [shap_one]

    sv_one = shap_one_list[pred_ex]
    if sv_one.shape[1] == X_example.shape[1] + 1:
        sv_one = sv_one[:, :-1]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[pred_ex]

    shap.plots.waterfall(
        shap.Explanation(
            values=sv_one[0],
            base_values=base_val,
            data=X_example.iloc[0],
            feature_names=X_example.columns
        ),
        max_display=20
    )
    plt.show()


# ==============================
# 11) Save model + metadata
# ==============================
model_name = "xgb_cls_optuna_multiobj_no_te_v1"
save_dir = os.path.join("models", "xgboost", "classification", model_name)
os.makedirs(save_dir, exist_ok=True)

joblib.dump(xgb_cls_bayes, os.path.join(save_dir, "model.joblib"))

meta = {
    "best_metric": best_metric,
    "best_trial_values_cv": {
        "f1_macro": best_f1_cv,
        "accuracy": best_acc_cv,
        "logloss": best_ll_cv
    },
    "best_params": best_params,
    "final_params": final_params,
    "n_trials": n_trials,
    "n_splits": n_splits
}
with open(os.path.join(save_dir, "optuna_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

df_metrics.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)
cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

print("\nSaved to:", save_dir)

```
