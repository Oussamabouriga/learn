```
# ============================================================
# XGBOOST CLASSIFICATION ONLY + BAYESIAN OPTIMIZATION (Optuna)
# - Works for binary or multiclass classification
# - Optimizes using MANY metrics computed in CV (not just one)
# - Uses multi-objective Optuna:
#     maximize: AUC (binary), F1_macro, BalancedAccuracy, Accuracy
#     minimize: LogLoss
# - Selects ONE best trial from Pareto front using an editable scoring rule
# - Trains final model on full train
# - Reports Train + Test metrics table
# - SHAP global + SHAP local for the example row
#
# REQUIREMENTS:
#   pip install xgboost optuna shap scikit-learn pandas numpy matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import optuna
import shap


# ============================================================
# 1) INPUTS YOU MUST HAVE
# ============================================================
# X_train_encoded, X_test_encoded: numeric DataFrames
# y_train_cls, y_test_cls: classification labels (int or str)
# X_new_encoded (optional): 1-row DataFrame encoded like training

# If you don't have y_train_cls yet, create it BEFORE running this code.

X_train_encoded = X_train_encoded.copy().astype(float)
X_test_encoded  = X_test_encoded.copy().astype(float)

y_train_cls = pd.Series(y_train_cls).copy()
y_test_cls  = pd.Series(y_test_cls).copy()

if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)

classes = np.unique(y_train_cls)
n_classes = len(classes)
is_binary = (n_classes == 2)

print("Train:", X_train_encoded.shape, y_train_cls.shape)
print("Test :", X_test_encoded.shape, y_test_cls.shape)
print("Classes:", classes, "| n_classes:", n_classes)


# ============================================================
# 2) CONFIG (editable)
# ============================================================
random_state = 42
n_splits = 5
n_trials = 60  # increase 60..150 if you have compute

# Pareto selection weights (editable)
# We want: high AUC (binary), high F1_macro, high balanced_acc, high acc, low logloss
w_auc = 1.0
w_f1 = 1.0
w_bal_acc = 1.0
w_acc = 0.3
w_logloss = 0.8


# ============================================================
# 3) CV
# ============================================================
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ============================================================
# 4) OPTUNA OBJECTIVE (multi-metrics, multi-objective)
# ============================================================
def objective(trial):
    # Select objective depending on binary vs multiclass
    if is_binary:
        objective_name = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective_name = "multi:softprob"
        eval_metric = "mlogloss"

    params = {
        "objective": objective_name,
        "eval_metric": eval_metric,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,

        # Hyperparameters (Bayesian)
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }

    if not is_binary:
        params["num_class"] = n_classes

    # Collect fold metrics
    aucs = []
    f1s = []
    baccs = []
    accs = []
    lls = []

    for tr_idx, va_idx in cv.split(X_train_encoded, y_train_cls):
        X_tr = X_train_encoded.iloc[tr_idx]
        y_tr = y_train_cls.iloc[tr_idx]

        X_va = X_train_encoded.iloc[va_idx]
        y_va = y_train_cls.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # proba for logloss and auc
        proba = model.predict_proba(X_va)
        pred = model.predict(X_va)

        # Metrics
        acc = accuracy_score(y_va, pred)
        bacc = balanced_accuracy_score(y_va, pred)
        f1m = f1_score(y_va, pred, average="macro")
        ll = log_loss(y_va, proba, labels=classes)

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1m)
        lls.append(ll)

        # AUC only reliable for binary; for multiclass use OVR macro
        if is_binary:
            aucs.append(roc_auc_score(y_va, proba[:, 1]))
        else:
            # multiclass AUC (OVR macro)
            try:
                aucs.append(roc_auc_score(y_va, proba, multi_class="ovr", average="macro"))
            except Exception:
                # if fails, skip fold (rare)
                pass

    auc_mean = float(np.mean(aucs)) if len(aucs) else np.nan
    f1_mean = float(np.mean(f1s))
    bacc_mean = float(np.mean(baccs))
    acc_mean = float(np.mean(accs))
    ll_mean = float(np.mean(lls))

    # Multi-objective returns:
    # maximize auc, maximize f1, maximize bacc, maximize acc, minimize logloss
    # Optuna expects numeric; AUC might be nan in edge cases -> replace with 0.0
    if np.isnan(auc_mean):
        auc_mean = 0.0

    return auc_mean, f1_mean, bacc_mean, acc_mean, ll_mean


# directions aligned with returned values
study = optuna.create_study(directions=["maximize", "maximize", "maximize", "maximize", "minimize"])
study.optimize(objective, n_trials=n_trials)

print("\n✅ Optuna done")
print("Pareto trials:", len(study.best_trials))


# ============================================================
# 5) PICK ONE BEST TRIAL from Pareto front (editable scoring)
# ============================================================
best_trial = None
best_score = -1e18

for t in study.best_trials:
    auc_cv, f1_cv, bacc_cv, acc_cv, ll_cv = t.values

    # Higher score = better
    score = (
        w_auc * auc_cv
        + w_f1 * f1_cv
        + w_bal_acc * bacc_cv
        + w_acc * acc_cv
        - w_logloss * ll_cv
    )

    if score > best_score:
        best_score = score
        best_trial = t

print("\n✅ Selected trial from Pareto front")
print("Score:", best_score)
print("CV metrics (AUC, F1_macro, BalAcc, Acc, LogLoss):", best_trial.values)
print("Best params:", best_trial.params)

best_params = best_trial.params


# ============================================================
# 6) TRAIN FINAL MODEL on FULL TRAIN with best params
# ============================================================
if is_binary:
    objective_name = "binary:logistic"
    eval_metric = "logloss"
else:
    objective_name = "multi:softprob"
    eval_metric = "mlogloss"

final_params = {
    "objective": objective_name,
    "eval_metric": eval_metric,
    "tree_method": "hist",
    "random_state": random_state,
    "n_jobs": -1,
    "verbosity": 0,
    **best_params
}
if not is_binary:
    final_params["num_class"] = n_classes

xgb_clf_best = XGBClassifier(**final_params)
xgb_clf_best.fit(X_train_encoded, y_train_cls)

print("\n✅ Final XGBoost classifier trained")


# ============================================================
# 7) REPORT MANY METRICS on TRAIN + TEST
# ============================================================
def compute_metrics(model, X, y, label=""):
    pred = model.predict(X)
    proba = model.predict_proba(X)

    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    ll = log_loss(y, proba, labels=classes)

    if is_binary:
        aucv = roc_auc_score(y, proba[:, 1])
    else:
        aucv = roc_auc_score(y, proba, multi_class="ovr", average="macro")

    return {
        "split": label,
        "AUC": float(aucv),
        "F1_macro": float(f1m),
        "BalancedAcc": float(bacc),
        "Accuracy": float(acc),
        "LogLoss": float(ll),
    }

train_metrics = compute_metrics(xgb_clf_best, X_train_encoded, y_train_cls, "Train")
test_metrics  = compute_metrics(xgb_clf_best, X_test_encoded,  y_test_cls,  "Test")

metrics_df = pd.DataFrame([train_metrics, test_metrics])
print("\n=== Metrics (Train/Test) ===")
display(metrics_df)

print("\n=== Confusion matrix (TEST) ===")
print(confusion_matrix(y_test_cls, xgb_clf_best.predict(X_test_encoded)))

print("\n=== Classification report (TEST) ===")
print(classification_report(y_test_cls, xgb_clf_best.predict(X_test_encoded), zero_division=0))


# ============================================================
# 8) SHAP (global + example local)
# ============================================================
print("\n=== SHAP: Global importance ===")
X_shap = X_test_encoded.sample(min(400, len(X_test_encoded)), random_state=random_state).copy()

explainer = shap.TreeExplainer(xgb_clf_best)
shap_values = explainer.shap_values(X_shap)

# Global SHAP importance
if isinstance(shap_values, list):
    # multiclass sometimes returns list[class] of (n, f)
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)  # (f,)
else:
    arr = np.array(shap_values)
    if arr.ndim == 3:
        mean_abs = np.abs(arr).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(arr).mean(axis=0)

global_shap = pd.DataFrame({
    "feature": X_shap.columns.astype(str),
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

display(global_shap.head(20))

plt.figure(figsize=(8, 6))
plt.barh(global_shap.head(20)["feature"][::-1], global_shap.head(20)["mean_abs_shap"][::-1])
plt.title("SHAP global importance (Top 20)")
plt.xlabel("mean(|SHAP|)")
plt.tight_layout()
plt.show()

# Local SHAP for example row (if exists)
if "X_new_encoded" in globals():
    print("\n=== SHAP: Local explanation (example) ===")

    # predict class for example
    pred_ex = xgb_clf_best.predict(X_new_encoded)[0]

    sv_ex = explainer.shap_values(X_new_encoded)

    if isinstance(sv_ex, list):
        # if multiclass list, select predicted class index if label is numeric 0..K-1
        # If labels are not 0..K-1, fallback to first class
        try:
            class_idx = int(pred_ex)
            sv_use = sv_ex[class_idx][0]
            base = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        except Exception:
            sv_use = sv_ex[0][0]
            base = explainer.expected_value[0]
    else:
        arr = np.array(sv_ex)
        if arr.ndim == 3:
            # (1,f,K) -> take the best proba class index
            class_idx = int(np.argmax(xgb_clf_best.predict_proba(X_new_encoded)[0]))
            sv_use = arr[0, :, class_idx]
            base = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            sv_use = arr[0]
            base = explainer.expected_value

    exp = shap.Explanation(
        values=sv_use,
        base_values=base,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )
    shap.plots.waterfall(exp, max_display=15, show=True)

    print("Example predicted class:", pred_ex)
else:
    print("\n[INFO] X_new_encoded not found -> skip SHAP local example")









What metrics are optimized during Bayesian search?

This Optuna search optimizes CV metrics (on validation folds), not test:
	•	AUC (binary or multiclass OVR macro)
	•	F1_macro
	•	BalancedAccuracy
	•	Accuracy
	•	LogLoss

Then it selects one configuration from the Pareto front using the weighted score.

If you tell me whether your target is binary or multiclass (and the label names), I can tailor:
	•	the exact AUC type (binary vs macro/weighted),
	•	add precision/recall for a specific class,
	•	or optimize directly for “recall on the bad class”.

```
