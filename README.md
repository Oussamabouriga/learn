```
# ============================================================
# TWO-STAGE ENSEMBLE:
#   Stage 1: XGBoost CLASSIFIER -> predicts satisfaction class
#   Stage 2: XGBoost REGRESSOR per class -> predicts note within class range
# + Bayesian optimization (Optuna) to tune both stages
# + Feature importance (XGBoost + SHAP)
# + Example row prediction + local SHAP
# ============================================================
# Requirements:
#   pip install xgboost optuna shap scikit-learn
# ============================================================

import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)

import optuna
import shap
import matplotlib.pyplot as plt


# ==============================
# 0) ASSUMPTIONS (you already have)
# ==============================
# X_train_encoded, X_test_encoded: encoded numeric DataFrames (same columns)
# y_train, y_test: numeric target 0..10 (float or int)
# X_new_encoded: 1-row encoded DataFrame (optional)

X_train_encoded = X_train_encoded.copy().astype(float)
X_test_encoded  = X_test_encoded.copy().astype(float)

y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test  = pd.to_numeric(y_test, errors="coerce").astype(float)

if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)


# ==============================
# 1) Define class mapping (0..10 -> 5 classes)
# ==============================
# Class ids:
# 0: extrêmement mauvais (0..2)
# 1: mauvais            (3..6)
# 2: neutre            (7..8)
# 3: bien              (9)
# 4: très bien         (10)

class_names = {
    0: "extrêmement mauvais",
    1: "mauvais",
    2: "neutre",
    3: "bien",
    4: "très bien"
}

# Ranges to clip regressor output per class
class_ranges = {
    0: (0.0, 2.0),
    1: (3.0, 6.0),
    2: (7.0, 8.0),
    3: (9.0, 9.0),
    4: (10.0, 10.0)
}

def to_class(y):
    y = np.asarray(y, dtype=float)
    # Round to nearest integer note (optional, but consistent with 9 and 10 exact classes)
    y_round = np.rint(y).astype(int)

    cls = np.zeros_like(y_round, dtype=int)
    cls[(y_round >= 0) & (y_round <= 2)] = 0
    cls[(y_round >= 3) & (y_round <= 6)] = 1
    cls[(y_round >= 7) & (y_round <= 8)] = 2
    cls[(y_round == 9)] = 3
    cls[(y_round >= 10)] = 4
    return cls

y_train_cls = to_class(y_train)
y_test_cls  = to_class(y_test)

print("Class distribution (train):")
display(pd.Series(y_train_cls).map(class_names).value_counts())


# ==============================
# 2) Optional: sample weights for imbalance (classifier)
#    (inverse class frequency)
# ==============================
class_counts = pd.Series(y_train_cls).value_counts().to_dict()
clf_weights = np.array([1.0 / class_counts[c] for c in y_train_cls], dtype=float)
clf_weights = clf_weights / clf_weights.mean()
clf_weights = np.clip(clf_weights, 0.5, 5.0)

print("\nClassifier sample weights summary:")
print(pd.Series(clf_weights).describe())


# ==============================
# 3) Optuna Bayesian Optimization config
# ==============================
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

n_trials = 40  # increase to 60-120 if you have compute

# penalty weight for classification accuracy inside objective
lambda_acc = 0.4  # editable: higher => prioritize class accuracy more


# ==============================
# 4) Optuna objective: optimize the ensemble end-to-end on CV
#    We optimize a single scalar:
#      objective = MAE_note_CV + lambda_acc*(1 - acc_class_CV)
# ==============================
def objective(trial):
    # ---- Classifier hyperparams (XGBClassifier)
    clf_params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,

        "n_estimators": trial.suggest_int("clf_n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("clf_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("clf_max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("clf_min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("clf_gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("clf_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("clf_colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("clf_reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("clf_reg_lambda", 0.5, 10.0, log=True),
    }

    # ---- Regressor hyperparams (shared across all class regressors)
    reg_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "missing": np.nan,

        "n_estimators": trial.suggest_int("reg_n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("reg_max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("reg_min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("reg_gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("reg_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("reg_colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_reg_lambda", 0.5, 10.0, log=True),
    }

    fold_mae_note = []
    fold_acc_cls = []

    for tr_idx, va_idx in cv.split(X_train_encoded):
        X_tr = X_train_encoded.iloc[tr_idx]
        y_tr_note = y_train.iloc[tr_idx]
        y_tr_cls = y_train_cls[tr_idx]
        w_tr_cls = clf_weights[tr_idx]

        X_va = X_train_encoded.iloc[va_idx]
        y_va_note = y_train.iloc[va_idx]
        y_va_cls_true = y_train_cls[va_idx]

        # -------- Stage 1: classifier
        clf = XGBClassifier(**clf_params)
        clf.fit(X_tr, y_tr_cls, sample_weight=w_tr_cls)

        y_va_cls_pred = clf.predict(X_va)
        acc = float(accuracy_score(y_va_cls_true, y_va_cls_pred))
        fold_acc_cls.append(acc)

        # -------- Stage 2: regressors per class (train on fold train only)
        # Train 5 regressors, each on samples belonging to that class in fold train
        regressors = {}
        for c in range(5):
            idx_c = np.where(y_tr_cls == c)[0]
            if len(idx_c) < 20:
                # too few samples -> fallback regressor that predicts mean of that class range
                regressors[c] = None
            else:
                reg = XGBRegressor(**reg_params)
                reg.fit(X_tr.iloc[idx_c], y_tr_note.iloc[idx_c])
                regressors[c] = reg

        # Predict note for validation samples:
        pred_note = np.zeros(len(X_va), dtype=float)

        for i, c_pred in enumerate(y_va_cls_pred):
            lo, hi = class_ranges[int(c_pred)]
            if regressors[int(c_pred)] is None:
                # fallback to middle of range (or exact if range is a point)
                pred = (lo + hi) / 2.0
            else:
                pred = float(regressors[int(c_pred)].predict(X_va.iloc[[i]])[0])
                pred = float(np.clip(pred, lo, hi))
            pred_note[i] = pred

        mae_note = float(mean_absolute_error(y_va_note.values, pred_note))
        fold_mae_note.append(mae_note)

    mae_cv = float(np.mean(fold_mae_note))
    acc_cv = float(np.mean(fold_acc_cls))

    # Single objective: minimize note error + penalty for poor class accuracy
    obj = mae_cv + lambda_acc * (1.0 - acc_cv)
    return obj


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_params
print("\n✅ Best Optuna objective:", study.best_value)
print("✅ Best params:", best_params)


# ==============================
# 5) Train final ensemble on FULL TRAIN using best params
# ==============================
# Rebuild clf/reg params
clf_params_best = {
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,

    "n_estimators": best_params["clf_n_estimators"],
    "learning_rate": best_params["clf_learning_rate"],
    "max_depth": best_params["clf_max_depth"],
    "min_child_weight": best_params["clf_min_child_weight"],
    "gamma": best_params["clf_gamma"],
    "subsample": best_params["clf_subsample"],
    "colsample_bytree": best_params["clf_colsample_bytree"],
    "reg_alpha": best_params["clf_reg_alpha"],
    "reg_lambda": best_params["clf_reg_lambda"],
}

reg_params_best = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "missing": np.nan,

    "n_estimators": best_params["reg_n_estimators"],
    "learning_rate": best_params["reg_learning_rate"],
    "max_depth": best_params["reg_max_depth"],
    "min_child_weight": best_params["reg_min_child_weight"],
    "gamma": best_params["reg_gamma"],
    "subsample": best_params["reg_subsample"],
    "colsample_bytree": best_params["reg_colsample_bytree"],
    "reg_alpha": best_params["reg_reg_alpha"],
    "reg_lambda": best_params["reg_reg_lambda"],
}

# Train classifier on full train
clf_final = XGBClassifier(**clf_params_best)
clf_final.fit(X_train_encoded, y_train_cls, sample_weight=clf_weights)

# Train regressors per class on full train
regressors_final = {}
for c in range(5):
    idx_c = np.where(y_train_cls == c)[0]
    if len(idx_c) < 20:
        regressors_final[c] = None
    else:
        reg = XGBRegressor(**reg_params_best)
        reg.fit(X_train_encoded.iloc[idx_c], y_train.iloc[idx_c])
        regressors_final[c] = reg

print("\n✅ Final ensemble trained (classifier + 5 regressors)")


# ==============================
# 6) Evaluate on TEST (end-to-end)
# ==============================
y_test_cls_pred = clf_final.predict(X_test_encoded)
cls_acc_test = accuracy_score(y_test_cls, y_test_cls_pred)

pred_test_note = np.zeros(len(X_test_encoded), dtype=float)
for i, c_pred in enumerate(y_test_cls_pred):
    lo, hi = class_ranges[int(c_pred)]
    reg = regressors_final[int(c_pred)]
    if reg is None:
        pred = (lo + hi) / 2.0
    else:
        pred = float(reg.predict(X_test_encoded.iloc[[i]])[0])
        pred = float(np.clip(pred, lo, hi))
    pred_test_note[i] = pred

mae_test = mean_absolute_error(y_test.values, pred_test_note)
rmse_test = float(np.sqrt(mean_squared_error(y_test.values, pred_test_note)))
r2_test = float(r2_score(y_test.values, pred_test_note))
acc_test_tol = float((np.abs(y_test.values - pred_test_note) <= 1.0).mean() * 100)

print("\n=== ENSEMBLE TEST RESULTS ===")
print("Classification accuracy (5 classes):", round(cls_acc_test, 4))
print("MAE :", round(mae_test, 4))
print("RMSE:", round(rmse_test, 4))
print("R2  :", round(r2_test, 4))
print("Accuracy@±1.0:", round(acc_test_tol, 2), "%")

print("\nConfusion matrix (classes):")
print(confusion_matrix(y_test_cls, y_test_cls_pred))


# ==============================
# 7) Predict on example row (X_new_encoded) + show chosen class
# ==============================
if "X_new_encoded" in globals():
    c_pred = int(clf_final.predict(X_new_encoded)[0])
    c_name = class_names[c_pred]
    lo, hi = class_ranges[c_pred]

    reg = regressors_final[c_pred]
    if reg is None:
        pred_note = (lo + hi) / 2.0
    else:
        pred_note = float(reg.predict(X_new_encoded)[0])
        pred_note = float(np.clip(pred_note, lo, hi))

    print("\n=== EXAMPLE PREDICTION ===")
    print("Predicted class:", c_pred, "-", c_name)
    print("Class range:", (lo, hi))
    print("Predicted note:", pred_note)
else:
    print("\nX_new_encoded not found -> skip example prediction")


# ==============================
# 8) Feature importance (XGBoost built-in) — classifier
# ==============================
imp_clf = pd.DataFrame({
    "feature": X_train_encoded.columns.astype(str),
    "importance": clf_final.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 classifier feature importances:")
display(imp_clf.head(20))


# ==============================
# 9) SHAP — classifier global + example local
# ==============================
# Global on a sample for speed
X_shap = X_test_encoded.sample(min(600, len(X_test_encoded)), random_state=42).copy()

explainer_clf = shap.TreeExplainer(clf_final)
shap_values_clf = explainer_clf.shap_values(X_shap)  # multiclass -> list or array

print("\nSHAP (classifier): global importance (bar)")
# For multiclass, shap returns list[ndarray] or ndarray; handle both:
if isinstance(shap_values_clf, list):
    # average absolute shap across classes
    shap_abs = np.mean([np.abs(sv) for sv in shap_values_clf], axis=0)
else:
    shap_abs = np.abs(shap_values_clf)

global_shap_clf = pd.DataFrame({
    "feature": X_shap.columns.astype(str),
    "mean_abs_shap": shap_abs.mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

display(global_shap_clf.head(20))

plt.figure()
plt.barh(global_shap_clf.head(20).iloc[::-1]["feature"], global_shap_clf.head(20).iloc[::-1]["mean_abs_shap"])
plt.title("Classifier SHAP (Top 20) — mean(|SHAP|) over classes")
plt.tight_layout()
plt.show()

# Local SHAP for example (classifier) if available
if "X_new_encoded" in globals():
    sv_row = explainer_clf.shap_values(X_new_encoded)
    if isinstance(sv_row, list):
        # pick the predicted class shap
        sv_use = sv_row[c_pred][0]
        base_value = explainer_clf.expected_value[c_pred] if isinstance(explainer_clf.expected_value, (list, np.ndarray)) else explainer_clf.expected_value
    else:
        sv_use = sv_row[0]
        base_value = explainer_clf.expected_value

    local_clf_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "value_in_row": X_new_encoded.iloc[0].values,
        "shap_value": sv_use,
        "abs_shap": np.abs(sv_use),
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("\nTop 15 SHAP drivers for classifier (example, predicted class):")
    display(local_clf_df.head(15))


# ==============================
# 10) Feature importance + SHAP for the chosen regressor
# ==============================
if "X_new_encoded" in globals():
    reg = regressors_final[c_pred]
    if reg is not None:
        imp_reg = pd.DataFrame({
            "feature": X_train_encoded.columns.astype(str),
            "importance": reg.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"\nTop 20 regressor feature importances (class={c_pred} {class_names[c_pred]}):")
        display(imp_reg.head(20))

        # SHAP for that regressor
        explainer_reg = shap.TreeExplainer(reg)
        shap_values_reg = explainer_reg.shap_values(X_shap)

        print("\nSHAP (regressor): global summary")
        plt.figure()
        shap.summary_plot(shap_values_reg, X_shap, show=False)
        plt.tight_layout()
        plt.show()

        # Local explanation for example
        shap_row_reg = explainer_reg.shap_values(X_new_encoded)[0]
        base_reg = explainer_reg.expected_value
        if isinstance(base_reg, (list, np.ndarray)):
            base_reg = float(np.array(base_reg).reshape(-1)[0])
        else:
            base_reg = float(base_reg)

        local_reg_df = pd.DataFrame({
            "feature": X_new_encoded.columns.astype(str),
            "value_in_row": X_new_encoded.iloc[0].values,
            "shap_value": shap_row_reg,
            "abs_shap": np.abs(shap_row_reg),
            "direction": np.where(shap_row_reg > 0, "↑ augmente", np.where(shap_row_reg < 0, "↓ diminue", "≈ neutre"))
        }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

        print("\nTop 15 SHAP drivers for regressor (example):")
        display(local_reg_df.head(15))

        # Waterfall (clean)
        exp = shap.Explanation(
            values=shap_row_reg,
            base_values=base_reg,
            data=X_new_encoded.iloc[0].values,
            feature_names=X_new_encoded.columns.astype(str).tolist()
        )
        plt.figure()
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.tight_layout()
        plt.show()
    else:
        print("\nRegressor for predicted class has too few samples -> no SHAP/reg importance.")
```
