```
# ============================================================
# GATED MODEL (Mixture-of-Experts)
# XGBoost Classifier (Optuna) + XGBoost Regressors per class (Optuna)
# - Configurable class intervals (editable table)
# - Train + CV Bayesian optimization (Optuna) for both classifier & regressors
# - Gated prediction: classify -> pick regressor -> predict inside interval (clipped)
# - Metrics (classification + regression gated) + Accuracy@±tol
# - SHAP global + SHAP local on example (classifier + predicted-class regressor)
# - Save everything under: models/assembled_models/<model_name>/
# ============================================================

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from xgboost import XGBClassifier, XGBRegressor

import optuna
import shap
import matplotlib.pyplot as plt

# ============================================================
# 0) CONFIG (EDITABLE)
# ============================================================

# ---- A) Model name + save path
model_name = "xgb_gated_optuna_v1"
save_dir = f"models/assembled_models/{model_name}"
os.makedirs(save_dir, exist_ok=True)

# ---- B) Accuracy tolerance (points)
tol = 1.0  # change to 0.5 if you want

# ---- C) Class intervals table (EDIT THIS)
# Rules:
# - Each row defines one class_id and note interval [low, high]
# - include_high means upper bound included (use True on last bin or discrete bins)
# - label is for display
class_config = pd.DataFrame([
    {"class_id": 0, "label_fr": "extrêmement mauvais", "low": 0.0, "high": 2.0, "include_high": True},
    {"class_id": 1, "label_fr": "mauvais",             "low": 3.0, "high": 6.0, "include_high": True},
    {"class_id": 2, "label_fr": "neutre",             "low": 7.0, "high": 8.0, "include_high": True},
    {"class_id": 3, "label_fr": "bien",               "low": 9.0, "high": 9.0, "include_high": True},
    {"class_id": 4, "label_fr": "très bien",          "low": 10.0,"high": 10.0,"include_high": True},
]).sort_values("class_id").reset_index(drop=True)

# Build helper dicts
class_names_fr = {int(r["class_id"]): str(r["label_fr"]) for _, r in class_config.iterrows()}
class_ranges = {int(r["class_id"]): (float(r["low"]), float(r["high"]), bool(r["include_high"])) for _, r in class_config.iterrows()}
num_classes = int(class_config["class_id"].nunique())

print("=== Class config (editable) ===")
display(class_config)

# ---- D) Optuna configs
# Classifier optimization
n_trials_cls = 40
n_splits_cls = 5
random_state = 42

# Regressor optimization per class
n_trials_reg_per_class = 30
n_splits_reg = 5

# Composite objective weights (tune if you want)
# Classifier: maximize f1_macro + accuracy (and optionally you can add auc_ovr if computed)
w_cls_f1 = 0.7
w_cls_acc = 0.3

# Regressor: minimize rmse + mae and maximize r2 + acc@tol
# We'll minimize: (rmse + 0.6*mae) - 0.6*r2 - 0.02*acc
w_reg_rmse = 1.0
w_reg_mae = 0.6
w_reg_r2 = 0.6
w_reg_acc = 0.02


# ============================================================
# 1) DATA (expects existing prepared variables)
# ============================================================

# Required:
# X_train_encoded_no_te, X_test_encoded_no_te
# y_train_no_te, y_test_no_te

X_train_all = X_train_encoded_no_te.copy().astype(float)
X_test_all  = X_test_encoded_no_te.copy().astype(float)

y_train_all = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_all  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

# Drop missing target rows if any (safety)
train_mask = y_train_all.notna()
test_mask = y_test_all.notna()

X_train_all = X_train_all.loc[train_mask].copy()
y_train_all = y_train_all.loc[train_mask].copy()

X_test_all = X_test_all.loc[test_mask].copy()
y_test_all = y_test_all.loc[test_mask].copy()

print("Train:", X_train_all.shape, y_train_all.shape)
print("Test :", X_test_all.shape,  y_test_all.shape)


# ============================================================
# 2) Build CLASS labels from y using class_config
# ============================================================

def y_to_class_id(y_val: float, config_df: pd.DataFrame) -> int:
    # Assign the first matching bin
    for _, r in config_df.iterrows():
        cid = int(r["class_id"])
        low = float(r["low"])
        high = float(r["high"])
        inc_high = bool(r["include_high"])
        if inc_high:
            if (y_val >= low) and (y_val <= high):
                return cid
        else:
            if (y_val >= low) and (y_val < high):
                return cid
    # fallback: if nothing matches (shouldn't happen if bins cover 0..10)
    # assign closest by clipping
    return int(config_df["class_id"].iloc[-1])

y_train_cls = y_train_all.apply(lambda v: y_to_class_id(float(v), class_config)).astype(int)
y_test_cls  = y_test_all.apply(lambda v: y_to_class_id(float(v), class_config)).astype(int)

print("\nTrain class distribution:")
print(y_train_cls.value_counts().sort_index().rename(index=class_names_fr))

print("\nTest class distribution:")
print(y_test_cls.value_counts().sort_index().rename(index=class_names_fr))


# ============================================================
# 3) OPTUNA — XGBoost CLASSIFIER (Bayesian) with Stratified CV
# ============================================================

cv_cls = StratifiedKFold(n_splits=n_splits_cls, shuffle=True, random_state=random_state)

def objective_xgb_classifier(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),

        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    }

    f1s = []
    accs = []

    for tr_idx, va_idx in cv_cls.split(X_train_all, y_train_cls):
        X_tr, X_va = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
        y_tr, y_va = y_train_cls.iloc[tr_idx], y_train_cls.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        pred_va = model.predict(X_va)
        f1 = f1_score(y_va, pred_va, average="macro")
        acc = accuracy_score(y_va, pred_va)

        f1s.append(float(f1))
        accs.append(float(acc))

    mean_f1 = float(np.mean(f1s))
    mean_acc = float(np.mean(accs))

    # We want to MAXIMIZE both => minimize negative weighted sum
    score = -(w_cls_f1 * mean_f1 + w_cls_acc * mean_acc)
    return score

study_cls = optuna.create_study(direction="minimize")
study_cls.optimize(objective_xgb_classifier, n_trials=n_trials_cls)

best_params_cls = study_cls.best_params
print("\n=== Best classifier params (Optuna) ===")
print(best_params_cls)

# Train final classifier on full train
xgb_cls_gated = XGBClassifier(
    **best_params_cls,
    objective="multi:softprob",
    num_class=num_classes,
    tree_method="hist",
    n_jobs=-1,
    random_state=random_state,
    verbosity=0,
    eval_metric="mlogloss"
)
xgb_cls_gated.fit(X_train_all, y_train_cls)

# Evaluate classifier
pred_train_cls = xgb_cls_gated.predict(X_train_all)
pred_test_cls  = xgb_cls_gated.predict(X_test_all)

acc_train_cls = accuracy_score(y_train_cls, pred_train_cls) * 100
acc_test_cls  = accuracy_score(y_test_cls,  pred_test_cls)  * 100

f1_train_cls = f1_score(y_train_cls, pred_train_cls, average="macro")
f1_test_cls  = f1_score(y_test_cls,  pred_test_cls,  average="macro")

print("\n=== CLASSIFIER metrics ===")
print(f"Accuracy train: {acc_train_cls:.2f}% | test: {acc_test_cls:.2f}%")
print(f"F1 macro train: {f1_train_cls:.4f} | test: {f1_test_cls:.4f}")

print("\nClassification report (test):")
print(classification_report(y_test_cls, pred_test_cls, target_names=[class_names_fr[i] for i in range(num_classes)]))

cm = confusion_matrix(y_test_cls, pred_test_cls)
cm_df = pd.DataFrame(cm,
    index=[f"Réel_{i}:{class_names_fr[i]}" for i in range(num_classes)],
    columns=[f"Prédit_{i}:{class_names_fr[i]}" for i in range(num_classes)]
)
print("\nMatrice de confusion (test):")
display(cm_df)


# ============================================================
# 4) OPTUNA — XGBoost REGRESSOR PER CLASS (trained only on that class)
# ============================================================

def clip_to_class_range(pred: np.ndarray, class_id: int) -> np.ndarray:
    low, high, _ = class_ranges[int(class_id)]
    return np.clip(pred, low, high)

def metrics_reg(y_true, y_pred, tol_points=1.0):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol_points).mean() * 100.0)
    return mae, rmse, r2, acc

# Store regressors per class
xgb_reg_by_class = {}
best_params_reg_by_class = {}
reg_train_sizes = {}

cv_reg = KFold(n_splits=n_splits_reg, shuffle=True, random_state=random_state)

for cid in range(num_classes):
    # Filter train subset for this class
    mask_c = (y_train_cls == cid)
    X_c = X_train_all.loc[mask_c].copy()
    y_c = y_train_all.loc[mask_c].copy()

    reg_train_sizes[cid] = int(len(X_c))
    print(f"\n--- Class {cid} | {class_names_fr[cid]} | train size: {len(X_c)} ---")

    # If too few samples, we do a safe small model without CV tuning
    if len(X_c) < max(50, n_splits_reg * 10):
        print("Too few samples for Optuna CV -> using default regressor params.")
        reg = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0
        )
        reg.fit(X_c, y_c)
        xgb_reg_by_class[cid] = reg
        best_params_reg_by_class[cid] = {"fallback": True}
        continue

    def objective_xgb_reg(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),

            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
            "verbosity": 0,
        }

        maes, rmses, r2s, accs = [], [], [], []

        for tr_idx, va_idx in cv_reg.split(X_c):
            X_tr, X_va = X_c.iloc[tr_idx], X_c.iloc[va_idx]
            y_tr, y_va = y_c.iloc[tr_idx], y_c.iloc[va_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            pred_va = model.predict(X_va)
            pred_va = clip_to_class_range(pred_va, cid)

            mae, rmse, r2, acc = metrics_reg(y_va, pred_va, tol_points=tol)
            maes.append(mae); rmses.append(rmse); r2s.append(r2); accs.append(acc)

        mae_m = float(np.mean(maes))
        rmse_m = float(np.mean(rmses))
        r2_m = float(np.mean(r2s))
        acc_m = float(np.mean(accs))

        # Minimize: rmse + w*mae - w*r2 - w*acc
        score = (w_reg_rmse * rmse_m) + (w_reg_mae * mae_m) - (w_reg_r2 * r2_m) - (w_reg_acc * acc_m)
        return float(score)

    study_reg = optuna.create_study(direction="minimize")
    study_reg.optimize(objective_xgb_reg, n_trials=n_trials_reg_per_class)

    best_params_reg = study_reg.best_params
    best_params_reg_by_class[cid] = best_params_reg

    reg = XGBRegressor(
        **best_params_reg,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0
    )
    reg.fit(X_c, y_c)

    xgb_reg_by_class[cid] = reg
    print("Best reg params:", best_params_reg)


# ============================================================
# 5) GATED PREDICTION (train & test)
# ============================================================

def gated_predict(X_df: pd.DataFrame, cls_model, reg_models: dict) -> (np.ndarray, np.ndarray):
    pred_class = cls_model.predict(X_df).astype(int)

    pred_value = np.zeros(len(X_df), dtype=float)

    for cid in range(num_classes):
        idx = np.where(pred_class == cid)[0]
        if len(idx) == 0:
            continue

        reg = reg_models.get(cid, None)
        if reg is None:
            # fallback: mid of interval
            low, high, _ = class_ranges[cid]
            pred_value[idx] = (low + high) / 2.0
            continue

        X_part = X_df.iloc[idx]
        pred_part = reg.predict(X_part)
        pred_part = clip_to_class_range(pred_part, cid)
        pred_value[idx] = pred_part

    # Final safety clip to [0,10]
    pred_value = np.clip(pred_value, 0, 10)
    return pred_class, pred_value

pred_train_gate_class, pred_train_gate_value = gated_predict(X_train_all, xgb_cls_gated, xgb_reg_by_class)
pred_test_gate_class,  pred_test_gate_value  = gated_predict(X_test_all,  xgb_cls_gated, xgb_reg_by_class)

# ============================================================
# 6) METRICS (classification + regression gated)
# ============================================================

# Classifier metrics (already computed) but also for gated predicted class:
acc_train_gate_cls = accuracy_score(y_train_cls, pred_train_gate_class) * 100
acc_test_gate_cls  = accuracy_score(y_test_cls,  pred_test_gate_class)  * 100
f1_train_gate_cls  = f1_score(y_train_cls, pred_train_gate_class, average="macro")
f1_test_gate_cls   = f1_score(y_test_cls,  pred_test_gate_class,  average="macro")

# Regression metrics on gated predicted value
mae_tr, rmse_tr, r2_tr, acc_tr = metrics_reg(y_train_all, pred_train_gate_value, tol_points=tol)
mae_te, rmse_te, r2_te, acc_te = metrics_reg(y_test_all,  pred_test_gate_value,  tol_points=tol)

results_table = pd.DataFrame({
    "Bloc": [
        "Classifier (global)",
        "Gated classifier",
        "Gated regression"
    ],
    "Train": [
        f"Acc={acc_train_cls:.2f}%, F1macro={f1_train_cls:.4f}",
        f"Acc={acc_train_gate_cls:.2f}%, F1macro={f1_train_gate_cls:.4f}",
        f"MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f}, Acc@±{tol}={acc_tr:.2f}%"
    ],
    "Test": [
        f"Acc={acc_test_cls:.2f}%, F1macro={f1_test_cls:.4f}",
        f"Acc={acc_test_gate_cls:.2f}%, F1macro={f1_test_gate_cls:.4f}",
        f"MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, R2={r2_te:.4f}, Acc@±{tol}={acc_te:.2f}%"
    ],
})

print("\n=== GATED MODEL RESULTS ===")
display(results_table)

# Sanity check: gated values always inside predicted class interval?
def check_inside_interval(pred_class_arr, pred_value_arr):
    ok = True
    for cid in range(num_classes):
        low, high, _ = class_ranges[cid]
        idx = np.where(pred_class_arr == cid)[0]
        if len(idx) == 0:
            continue
        v = pred_value_arr[idx]
        if np.any(v < low - 1e-9) or np.any(v > high + 1e-9):
            ok = False
            print("Interval violation for class", cid, "min/max:", float(v.min()), float(v.max()), "expected:", (low, high))
    return ok

print("Gated interval check (test):", check_inside_interval(pred_test_gate_class, pred_test_gate_value))


# ============================================================
# 7) EXAMPLE PREDICTION (encoded example)
# ============================================================

# Priority:
# - if X_new_encoded_no_te exists: use it
# - else if test_row_xgb_no_te exists: build df and align
X_example = None

if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()
elif "test_row_xgb_no_te" in globals():
    X_example = pd.DataFrame(test_row_xgb_no_te).copy()
    # align to training columns
    for c in X_train_all.columns:
        if c not in X_example.columns:
            X_example[c] = 0.0
    X_example = X_example[X_train_all.columns].copy()
else:
    raise ValueError("Provide X_new_encoded_no_te (encoded example) or test_row_xgb_no_te (raw).")

X_example = X_example.copy()
# Ensure numeric float
X_example = X_example[X_train_all.columns].astype(float)

ex_class, ex_value = gated_predict(X_example, xgb_cls_gated, xgb_reg_by_class)
ex_class = int(ex_class[0])
ex_value = float(ex_value[0])

low, high, _ = class_ranges[ex_class]

print("\n=== EXAMPLE PREDICTION (GATED) ===")
print("Classe prédite:", ex_class, "|", class_names_fr.get(ex_class, ex_class))
print("Intervalle:", (low, high))
print("Note prédite:", ex_value)


# ============================================================
# 8) SHAP (classifier global + example) and regressor (pred class) global + example
# ============================================================

# ---------- Helper: safe SHAP array fixing (bias column, multiclass handling)
def _fix_shap_matrix(sv, X_df):
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n, f, k) -> keep as is here (handled outside)
        return sv
    # bias column
    if sv.shape[1] == X_df.shape[1] + 1:
        sv = sv[:, :-1]
    if sv.shape[1] != X_df.shape[1]:
        m = min(sv.shape[1], X_df.shape[1])
        sv = sv[:, :m]
        X_df = X_df.iloc[:, :m].copy()
    return sv, X_df

# ---------- SHAP for classifier (global)
X_shap_cls = X_test_all.sample(min(300, len(X_test_all)), random_state=42).copy()
explainer_cls = shap.TreeExplainer(xgb_cls_gated)
sv_cls = explainer_cls.shap_values(X_shap_cls)

# multiclass: sv_cls can be list or (n,f,k)
class_for_global = ex_class  # show global for predicted class of example
if isinstance(sv_cls, list):
    svg = sv_cls[class_for_global]
else:
    sv_arr = np.asarray(sv_cls)
    if sv_arr.ndim == 3:
        svg = sv_arr[:, :, class_for_global]
    else:
        svg = sv_arr

svg, X_shap_plot = _fix_shap_matrix(svg, X_shap_cls)

print("\nSHAP global — CLASSIFIER — class:", class_for_global, "|", class_names_fr.get(class_for_global))
shap.summary_plot(svg, X_shap_plot, show=True)
shap.summary_plot(svg, X_shap_plot, plot_type="bar", show=True)

# ---------- SHAP for classifier (example local)
sv_one_cls = explainer_cls.shap_values(X_example)
if isinstance(sv_one_cls, list):
    svo = np.asarray(sv_one_cls[ex_class])
else:
    svo_arr = np.asarray(sv_one_cls)
    if svo_arr.ndim == 3:
        svo = svo_arr[:, :, ex_class]
    else:
        svo = svo_arr

# fix bias col
if svo.shape[1] == X_example.shape[1] + 1:
    svo = svo[:, :-1]

expected_cls = explainer_cls.expected_value
base_val_cls = expected_cls[ex_class] if isinstance(expected_cls, (list, np.ndarray)) else expected_cls

print("\nSHAP local — CLASSIFIER (example)")
shap.plots.waterfall(
    shap.Explanation(
        values=svo[0],
        base_values=base_val_cls,
        data=X_example.iloc[0],
        feature_names=X_example.columns
    ),
    max_display=20
)
plt.show()

# ---------- SHAP for regressor of predicted class (global + local)
reg_model = xgb_reg_by_class.get(ex_class, None)
if reg_model is not None:
    # pick X_shap within that class (true class subset) to interpret regressor properly
    mask_test_true_class = (y_test_cls == ex_class)
    if mask_test_true_class.sum() >= 30:
        X_shap_reg = X_test_all.loc[mask_test_true_class].sample(min(200, int(mask_test_true_class.sum())), random_state=42).copy()
    else:
        X_shap_reg = X_test_all.sample(min(200, len(X_test_all)), random_state=42).copy()

    explainer_reg = shap.TreeExplainer(reg_model)
    sv_reg = explainer_reg.shap_values(X_shap_reg)
    # sv_reg should be (n,f) or (n,f+1)
    sv_reg, X_shap_reg_plot = _fix_shap_matrix(sv_reg, X_shap_reg)

    print("\nSHAP global — REGRESSOR of class:", ex_class, "|", class_names_fr.get(ex_class))
    shap.summary_plot(sv_reg, X_shap_reg_plot, show=True)
    shap.summary_plot(sv_reg, X_shap_reg_plot, plot_type="bar", show=True)

    # local example shap
    sv_one_reg = explainer_reg.shap_values(X_example)
    sv_one_reg = np.asarray(sv_one_reg)
    if sv_one_reg.shape[1] == X_example.shape[1] + 1:
        sv_one_reg = sv_one_reg[:, :-1]

    expected_reg = explainer_reg.expected_value
    base_val_reg = expected_reg if not isinstance(expected_reg, (list, np.ndarray)) else expected_reg[0]

    print("\nSHAP local — REGRESSOR (example) for predicted class")
    shap.plots.waterfall(
        shap.Explanation(
            values=sv_one_reg[0],
            base_values=base_val_reg,
            data=X_example.iloc[0],
            feature_names=X_example.columns
        ),
        max_display=20
    )
    plt.show()
else:
    print("\nNo regressor found for predicted class (fallback used).")


# ============================================================
# 9) SAVE EVERYTHING (classifier + regressors + config + columns)
# ============================================================

# Save config + columns + optuna params
meta = {
    "model_name": model_name,
    "created_at": datetime.now().isoformat(),
    "tol": tol,
    "class_config": class_config.to_dict(orient="records"),
    "class_names_fr": class_names_fr,
    "classifier_optuna_best_params": best_params_cls,
    "regressors_optuna_best_params_by_class": best_params_reg_by_class,
    "reg_train_sizes_by_class": reg_train_sizes,
    "columns": list(X_train_all.columns),
    "num_classes": num_classes,
}
with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# Save XGBoost models as JSON (booster format)
cls_path = os.path.join(save_dir, "xgb_classifier.json")
xgb_cls_gated.get_booster().save_model(cls_path)

reg_dir = os.path.join(save_dir, "regressors_by_class")
os.makedirs(reg_dir, exist_ok=True)

for cid, reg in xgb_reg_by_class.items():
    reg_path = os.path.join(reg_dir, f"xgb_regressor_class_{cid}.json")
    reg.get_booster().save_model(reg_path)

print("\nSaved:")
print("-", os.path.join(save_dir, "meta.json"))
print("-", cls_path)
print("-", reg_dir)

```
