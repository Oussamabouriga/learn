```

Notes (important)

1) Why your regression predicted outside the class before

Because the regressor was trained on all values (or not clipped), so it could output any number.
✅ Here we enforce:

pred = np.clip(pred, lo, hi)

both during CV and during final prediction, so it cannot escape the class interval.

2) Which “accuracy” is optimized in Optuna here?
	•	Classifier Optuna: maximizes CV classification accuracy (train folds → val fold).
	•	Regressor Optuna: minimizes CV RMSE inside its class.
	•	Then you evaluate on your held-out test set.

If you want the classifier objective to be macro-F1 (better for imbalance), tell me and I’ll switch it.





	•	Step A: build 5 classes from your note (0–10)
	•	Step B: Optuna tunes the classifier (5 classes)
	•	Step C: Optuna tunes one regressor per class (only for classes with a real interval)
	•	Step D: final prediction = classifier → choose regressor → predict → CLIP inside class interval
	•	Step E: metrics + Accuracy@±tol + confusion matrix
	•	Step F: SHAP global + SHAP local (for classifier + the regressor used for the example)

 Works with your encoded matrices:
	•	X_train_encoded (DataFrame)
	•	X_test_encoded (DataFrame)
	•	y_train (Series/array)
	•	y_test (Series/array)
and your example row already encoded as X_new_encoded (1 row DataFrame, same columns as X_train_encoded)

If you don’t have X_new_encoded yet, scroll down: I included a safe alignment block that makes it match columns.



# ============================================================
# ENSEMBLE: XGBoost (5-class classifier) + 5 regressors (per class)
# OPTUNA Bayesian optimization (classifier + regressors)
# + imbalance handling via sample_weight
# + CV (StratifiedKFold for classifier, KFold for regressors)
# + metrics (MAE, RMSE, R2, Accuracy@±tol) + confusion matrix
# + SHAP global + SHAP local (example)
# ============================================================

# Requirements:
#   pip install xgboost optuna shap scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

import optuna
import shap

# ------------------------------
# 0) Config (EDITABLE)
# ------------------------------
random_state = 42
n_trials_clf = 40         # classifier Optuna trials
n_trials_reg = 30         # per-regressor Optuna trials
n_splits_clf = 5
n_splits_reg = 5

tol = 1.0                 # Accuracy@±tol on note prediction
note_clip_low, note_clip_high = 0.0, 10.0

# Class definitions (your rule)
# 0: 0..2  (extrêmement mauvais)
# 1: 3..6  (mauvais)
# 2: 7..8  (neutre)
# 3: 9     (bien)
# 4: 10    (très bien)
class_names_fr = {
    0: "extrêmement mauvais (0–2)",
    1: "mauvais (3–6)",
    2: "neutre (7–8)",
    3: "bien (9)",
    4: "très bien (10)",
}
class_ranges = {
    0: (0.0, 2.0),
    1: (3.0, 6.0),
    2: (7.0, 8.0),
    3: (9.0, 9.0),
    4: (10.0, 10.0),
}

# ------------------------------------------------------------
# 1) Inputs expected (already prepared/encoded by you)
# ------------------------------------------------------------
# X_train_encoded : pd.DataFrame
# X_test_encoded  : pd.DataFrame
# y_train         : pd.Series or np.array (0..10)
# y_test          : pd.Series or np.array (0..10)

# Safety: ensure correct types
X_train_encoded = X_train_encoded.copy()
X_test_encoded  = X_test_encoded.copy()
y_train = pd.Series(y_train).astype(float).copy()
y_test  = pd.Series(y_test).astype(float).copy()

# Optional: cast X to float (XGBoost friendly)
X_train_encoded = X_train_encoded.astype(float)
X_test_encoded  = X_test_encoded.astype(float)

print("X_train_encoded:", X_train_encoded.shape)
print("X_test_encoded :", X_test_encoded.shape)
print("y_train:", y_train.shape, " | y_test:", y_test.shape)

# ------------------------------------------------------------
# 2) Build 5-class target from y (classification labels)
# ------------------------------------------------------------
def to_class_label(y_val: float) -> int:
    # robust for float notes
    if y_val <= 2.0:
        return 0
    elif y_val <= 6.0:
        return 1
    elif y_val <= 8.0:
        return 2
    elif y_val < 10.0:
        # this captures 9.x (if exists) as class 3
        # if you only have exact 9, it's still fine
        return 3
    else:
        return 4

y_train_cls = y_train.apply(to_class_label).astype(int)
y_test_cls  = y_test.apply(to_class_label).astype(int)

print("\nClass distribution (train):")
print(y_train_cls.value_counts().sort_index())

# ------------------------------------------------------------
# 3) Imbalance handling (sample weights)
#    - classifier: inverse frequency of class
#    - regressors: inverse frequency of target bins (within class)
# ------------------------------------------------------------
# 3a) Class weights for classifier
class_counts = y_train_cls.value_counts().to_dict()
clf_weights_train = y_train_cls.map(lambda c: 1.0 / class_counts[c]).astype(float).values
clf_weights_train = clf_weights_train / np.mean(clf_weights_train)
clf_weights_train = np.clip(clf_weights_train, 0.5, 5.0)

# ------------------------------------------------------------
# 4) OPTUNA — Classifier (maximize CV accuracy)
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True, random_state=random_state)

def objective_clf(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,

        # main search space (compute-friendly)
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }

    fold_scores = []

    for tr_idx, va_idx in skf.split(X_train_encoded, y_train_cls):
        X_tr = X_train_encoded.iloc[tr_idx]
        y_tr = y_train_cls.iloc[tr_idx]
        w_tr = clf_weights_train[tr_idx]

        X_va = X_train_encoded.iloc[va_idx]
        y_va = y_train_cls.iloc[va_idx]

        model = XGBClassifier(**params)

        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        pred_va = model.predict(X_va)
        acc = accuracy_score(y_va, pred_va)
        fold_scores.append(acc)

    return float(np.mean(fold_scores))  # maximize

study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=n_trials_clf)

best_clf_params = study_clf.best_params
best_clf_cv_acc = study_clf.best_value

print("\n✅ OPTUNA classifier done")
print("Best CV accuracy:", best_clf_cv_acc)
print("Best classifier params:", best_clf_params)

# Train final classifier on full train
clf_final = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=random_state,
    n_jobs=-1,
    verbosity=0,
    **best_clf_params
)
clf_final.fit(X_train_encoded, y_train_cls, sample_weight=clf_weights_train)

# ------------------------------------------------------------
# 5) OPTUNA — Regressors (one per class)
#     - For classes with fixed value (9, 10), we use constants.
# ------------------------------------------------------------
def make_regression_weights(y_sub, n_bins=6, clip_min=0.5, clip_max=5.0):
    y_sub = pd.Series(y_sub).astype(float).reset_index(drop=True)
    # bin for imbalance in regression
    try:
        bins = pd.qcut(y_sub, q=min(n_bins, y_sub.nunique()), duplicates="drop")
    except Exception:
        bins = pd.cut(y_sub, bins=min(n_bins, max(2, y_sub.nunique())))
    freq = bins.value_counts()
    w = bins.map(lambda b: 1.0 / freq[b]).astype(float).values
    w = w / np.mean(w)
    return np.clip(w, clip_min, clip_max)

reg_models = {}     # class_id -> model or float constant
reg_best_params = {}

for cls_id, (lo, hi) in class_ranges.items():
    width = hi - lo

    # If the class interval is a single point -> constant predictor
    if width == 0.0:
        reg_models[cls_id] = float(lo)
        reg_best_params[cls_id] = {"type": "constant", "value": float(lo)}
        continue

    # Subset data for this class
    idx = (y_train_cls == cls_id).values
    X_sub = X_train_encoded.loc[idx].copy()
    y_sub = y_train.loc[idx].copy()

    if len(X_sub) < 50:
        # too few samples: use mean as fallback
        const_val = float(np.clip(y_sub.mean(), lo, hi))
        reg_models[cls_id] = const_val
        reg_best_params[cls_id] = {"type": "fallback_mean", "value": const_val, "n": int(len(X_sub))}
        continue

    # weights inside class
    w_sub = make_regression_weights(y_sub, n_bins=6, clip_min=0.5, clip_max=5.0)

    # KFold for this class
    k = min(n_splits_reg, max(2, len(X_sub)//100))
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    def objective_reg(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,

            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        fold_rmse = []

        for tr_i, va_i in kf.split(X_sub):
            X_tr = X_sub.iloc[tr_i]
            y_tr = y_sub.iloc[tr_i]
            w_tr = w_sub[tr_i]

            X_va = X_sub.iloc[va_i]
            y_va = y_sub.iloc[va_i]

            m = XGBRegressor(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

            pred = m.predict(X_va)
            pred = np.clip(pred, lo, hi)  # VERY IMPORTANT: keep inside class interval
            rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            fold_rmse.append(rmse)

        return float(np.mean(fold_rmse))  # minimize

    study_reg = optuna.create_study(direction="minimize")
    study_reg.optimize(objective_reg, n_trials=n_trials_reg)

    best_params = study_reg.best_params
    reg_best_params[cls_id] = best_params

    # Train final regressor on full class subset
    reg_final = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **best_params
    )
    reg_final.fit(X_sub, y_sub, sample_weight=w_sub)
    reg_models[cls_id] = reg_final

    print(f"\n✅ Optuna regressor done for class {cls_id} ({class_names_fr[cls_id]})")
    print("Best CV RMSE:", study_reg.best_value)

# ------------------------------------------------------------
# 6) Ensemble prediction (classifier -> regressor -> clipped note)
# ------------------------------------------------------------
# Predict class
pred_test_cls = clf_final.predict(X_test_encoded)
pred_train_cls = clf_final.predict(X_train_encoded)

def ensemble_predict_note(X_df, pred_cls_array):
    preds = np.zeros(len(X_df), dtype=float)

    for i, cls_id in enumerate(pred_cls_array):
        lo, hi = class_ranges[int(cls_id)]
        m = reg_models[int(cls_id)]

        if isinstance(m, (float, int)):
            p = float(m)
        else:
            p = float(m.predict(X_df.iloc[i:i+1])[0])
            p = float(np.clip(p, lo, hi))  # enforce interval

        preds[i] = p

    return np.clip(preds, note_clip_low, note_clip_high)

pred_test_note = ensemble_predict_note(X_test_encoded, pred_test_cls)
pred_train_note = ensemble_predict_note(X_train_encoded, pred_train_cls)

# ------------------------------------------------------------
# 7) Metrics (classification + regression on final note)
# ------------------------------------------------------------
clf_acc_test = accuracy_score(y_test_cls, pred_test_cls)
clf_acc_train = accuracy_score(y_train_cls, pred_train_cls)

mae_test = mean_absolute_error(y_test, pred_test_note)
rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test_note)))
r2_test = r2_score(y_test, pred_test_note)
acc_tol_test = (np.abs(y_test.values - pred_test_note) <= tol).mean() * 100

mae_train = mean_absolute_error(y_train, pred_train_note)
rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train_note)))
r2_train = r2_score(y_train, pred_train_note)
acc_tol_train = (np.abs(y_train.values - pred_train_note) <= tol).mean() * 100

print("\n=== ENSEMBLE RESULTS ===")
print(f"Classifier accuracy (train): {clf_acc_train:.4f}")
print(f"Classifier accuracy (test) : {clf_acc_test:.4f}")

print("\nFinal note prediction metrics:")
print(f"MAE  (train/test): {mae_train:.4f} / {mae_test:.4f}")
print(f"RMSE (train/test): {rmse_train:.4f} / {rmse_test:.4f}")
print(f"R2   (train/test): {r2_train:.4f} / {r2_test:.4f}")
print(f"Accuracy@±{tol} (train/test): {acc_tol_train:.2f}% / {acc_tol_test:.2f}%")

cm = confusion_matrix(y_test_cls, pred_test_cls, labels=[0,1,2,3,4])
print("\nConfusion matrix (test classes):")
print(cm)

# ------------------------------------------------------------
# 8) Example row: predict class + note + SHAP local explanation
# ------------------------------------------------------------
# Expected: you already built X_new_encoded with SAME columns as X_train_encoded.
# If not, build it then ALIGN it like this:
#
# X_new_encoded = <your encoded one-row DataFrame>
# (ensure same columns order)
if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.copy().astype(float)

    # align columns exactly (safe)
    for c in X_train_encoded.columns:
        if c not in X_new_encoded.columns:
            X_new_encoded[c] = 0.0
    X_new_encoded = X_new_encoded[X_train_encoded.columns].copy()

    # Predict class
    ex_cls = int(clf_final.predict(X_new_encoded)[0])
    lo, hi = class_ranges[ex_cls]
    reg_m = reg_models[ex_cls]

    if isinstance(reg_m, (float, int)):
        ex_note = float(reg_m)
    else:
        ex_note = float(reg_m.predict(X_new_encoded)[0])
        ex_note = float(np.clip(ex_note, lo, hi))

    print("\n=== EXAMPLE PREDICTION ===")
    print(f"Predicted class: {ex_cls} -> {class_names_fr[ex_cls]}")
    print(f"Class interval: [{lo}, {hi}]")
    print(f"Predicted note (clipped): {ex_note:.4f}")

else:
    print("\n[WARN] X_new_encoded not found. Create it then re-run section 8.")

# ------------------------------------------------------------
# 9) SHAP (Classifier): global importance + local (example)
# ------------------------------------------------------------
print("\n=== SHAP: CLASSIFIER ===")
X_shap = X_test_encoded.sample(min(300, len(X_test_encoded)), random_state=random_state)

expl_clf = shap.TreeExplainer(clf_final)
shap_vals_clf = expl_clf.shap_values(X_shap)

# Multi-class: shap_vals_clf is list[ndarray] or ndarray depending on version
if isinstance(shap_vals_clf, list):
    # shape: [K] of (n_samples, n_features)
    sv = np.stack([np.abs(s) for s in shap_vals_clf], axis=0)   # (K, n, f)
    mean_abs = sv.mean(axis=(0,1))                               # (f,)
else:
    # possible shape (n_samples, n_features) or (n_samples, n_features, K)
    arr = np.array(shap_vals_clf)
    if arr.ndim == 3:
        mean_abs = np.abs(arr).mean(axis=(0,2))
    else:
        mean_abs = np.abs(arr).mean(axis=0)

global_shap_clf = pd.DataFrame({
    "feature": X_shap.columns.astype(str),
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 20 SHAP classifier (global):")
display(global_shap_clf.head(20))

# Optional SHAP plot (global)
plt.figure(figsize=(8,6))
plt.barh(global_shap_clf.head(20)["feature"][::-1], global_shap_clf.head(20)["mean_abs_shap"][::-1])
plt.title("SHAP global importance — Classifier (Top 20)")
plt.xlabel("mean(|SHAP|)")
plt.tight_layout()
plt.show()

# Local SHAP for example (classifier)
if "X_new_encoded" in globals():
    shap_vals_ex = expl_clf.shap_values(X_new_encoded)

    # For multiclass, show explanation for predicted class
    if isinstance(shap_vals_ex, list):
        sv_ex = shap_vals_ex[ex_cls][0]  # (features,)
        base = expl_clf.expected_value[ex_cls] if isinstance(expl_clf.expected_value, (list, np.ndarray)) else expl_clf.expected_value
    else:
        # fallback
        sv_ex = np.array(shap_vals_ex)[0]
        base = expl_clf.expected_value

    exp = shap.Explanation(
        values=sv_ex,
        base_values=base,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )
    shap.plots.waterfall(exp, max_display=15, show=True)

# ------------------------------------------------------------
# 10) SHAP (Regressor used for example): global + local
# ------------------------------------------------------------
print("\n=== SHAP: REGRESSOR (example class) ===")
if "X_new_encoded" in globals():
    if isinstance(reg_m, (float, int)):
        print("Regressor is a constant for this class -> no SHAP.")
    else:
        Xr_shap = X_test_encoded[pred_test_cls == ex_cls].copy()
        if len(Xr_shap) == 0:
            Xr_shap = X_test_encoded.sample(min(200, len(X_test_encoded)), random_state=random_state)

        Xr_shap = Xr_shap.sample(min(300, len(Xr_shap)), random_state=random_state)

        expl_reg = shap.TreeExplainer(reg_m)
        sv_reg = expl_reg.shap_values(Xr_shap)

        # Global
        mean_abs_reg = np.abs(np.array(sv_reg)).mean(axis=0)
        global_shap_reg = pd.DataFrame({
            "feature": Xr_shap.columns.astype(str),
            "mean_abs_shap": mean_abs_reg
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        print("\nTop 20 SHAP regressor (global):")
        display(global_shap_reg.head(20))

        plt.figure(figsize=(8,6))
        plt.barh(global_shap_reg.head(20)["feature"][::-1], global_shap_reg.head(20)["mean_abs_shap"][::-1])
        plt.title(f"SHAP global importance — Regressor for class {ex_cls} (Top 20)")
        plt.xlabel("mean(|SHAP|)")
        plt.tight_layout()
        plt.show()

        # Local for example
        sv_ex_reg = expl_reg.shap_values(X_new_encoded)[0]
        base_reg = expl_reg.expected_value

        exp_reg = shap.Explanation(
            values=sv_ex_reg,
            base_values=base_reg,
            data=X_new_encoded.iloc[0].values,
            feature_names=X_new_encoded.columns.tolist()
        )
        shap.plots.waterfall(exp_reg, max_display=15, show=True)

print("\n✅ Done.")

```
