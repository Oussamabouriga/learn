```
# ============================================================
# CATBOOST — BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna)
# + Imbalanced regression sample weights
# + K-Fold CV
# + Train best model
# + Metrics + Accuracy@±tol
# + Predict your example row
# + SHAP global + SHAP local (example)
# ============================================================
# Requirements (run once if needed):
#   pip install catboost optuna shap
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

import optuna
import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Config (editable)
# ==============================
target_col = "evaluate_note"
test_size = 0.2
random_state = 42

# Categorical columns (EXPLICIT) — edit if needed
cat_cols = [
    "PARCOURS_FINAL",
    "PARCOURS_INITIAL",
    "operating_system",
    "marque",
    "model",
    "garantie",
    "list_prest",
    "code_postal",   # forced categorical
]
cat_cols = [c for c in cat_cols if c in df.columns]

# Accuracy tolerance (in target points)
tol = 1.0  # change to 0.5 if you want

# Sample weights config for imbalance
n_bins = 8
clip_min_w, clip_max_w = 0.5, 5.0

# Optuna / CV config
n_splits = 5
n_trials = 40  # adjust 20..80 depending on compute budget


# ==============================
# 2) Build X / y
# ==============================
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].astype(float).copy()

# Convert pd.NA -> np.nan
X = X.replace({pd.NA: np.nan})

# Force categorical cols to string + fill missing
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("__MISSING__")

# Convert numeric columns to numeric
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

print("X shape:", X.shape, " | y shape:", y.shape)
print("Categorical cols:", cat_cols)


# ==============================
# 3) Train/Test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)


# ==============================
# 4) Build sample weights on y_train (inverse-frequency bins)
# ==============================
try:
    y_bins = pd.qcut(y_train, q=min(n_bins, y_train.nunique()), duplicates="drop")
except Exception:
    y_bins = pd.cut(y_train, bins=min(n_bins, max(2, y_train.nunique())))

bin_counts = y_bins.value_counts()
weights_train = y_bins.map(lambda b: 1.0 / bin_counts[b]).astype(float).values
weights_train = weights_train / np.mean(weights_train)
weights_train = np.clip(weights_train, clip_min_w, clip_max_w)

print("\nWeights summary:")
print(pd.Series(weights_train).describe())


# ==============================
# 5) Bayesian optimization with Optuna (KFold CV, weighted RMSE)
# ==============================
cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def objective(trial):
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,

        # Tune main params (Bayesian)
        "iterations": trial.suggest_int("iterations", 600, 2600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),

        # Regularization / randomness
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),

        # Sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

        # Tree mode (CatBoost)
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
    }

    fold_rmses = []

    for tr_idx, va_idx in cv.split(X_train):
        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        w_tr = weights_train[tr_idx]

        X_va = X_train.iloc[va_idx].copy()
        y_va = y_train.iloc[va_idx].copy()

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=w_tr)
        val_pool   = Pool(X_va, y_va, cat_features=cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=200)

        pred_va = np.clip(model.predict(X_va), 0, 10)
        rmse_va = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        fold_rmses.append(rmse_va)

    return float(np.mean(fold_rmses))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)

best_bayes_params = study.best_params
best_bayes_rmse_cv = study.best_value

print("\n✅ Bayesian Optimization done")
print("Best CV RMSE:", best_bayes_rmse_cv)
print("Best params:", best_bayes_params)


# ==============================
# 6) Train final best Bayesian model on full train (with weights)
# ==============================
train_pool_full = Pool(X_train, y_train, cat_features=cat_cols, weight=weights_train)
test_pool       = Pool(X_test,  y_test,  cat_features=cat_cols)

cat_model_bayes_best = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=random_state,
    verbose=200,
    allow_writing_files=False,
    **best_bayes_params
)

cat_model_bayes_best.fit(
    train_pool_full,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("\n✅ Final Bayesian CatBoost trained")


# ==============================
# 7) Evaluate (Train/Test) + Accuracy@±tol
# ==============================
pred_train_b = np.clip(cat_model_bayes_best.predict(X_train), 0, 10)
pred_test_b  = np.clip(cat_model_bayes_best.predict(X_test),  0, 10)

mae_train_b  = mean_absolute_error(y_train, pred_train_b)
rmse_train_b = float(np.sqrt(mean_squared_error(y_train, pred_train_b)))
r2_train_b   = r2_score(y_train, pred_train_b)

mae_test_b   = mean_absolute_error(y_test, pred_test_b)
rmse_test_b  = float(np.sqrt(mean_squared_error(y_test, pred_test_b)))
r2_test_b    = r2_score(y_test, pred_test_b)

acc_train_b = (np.abs(y_train.values - pred_train_b) <= tol).mean() * 100
acc_test_b  = (np.abs(y_test.values  - pred_test_b)  <= tol).mean() * 100

metrics_bayes = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train":  [mae_train_b, rmse_train_b, r2_train_b, acc_train_b],
    "Test":   [mae_test_b,  rmse_test_b,  r2_test_b,  acc_test_b],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== Bayesian Best Model Metrics ===")
display(metrics_bayes)


# ==============================
# 8) Predict on your example row (same preprocessing rules)
# ==============================
new_rows = [{
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "tarif": 19.99,
    "Nombre_sisnitre_client": 1,
    "Nombre_sisnitre_accepte_client": 1,
    "Nombre_sisnitre_refuse_client": np.nan,
    "Nombre_sisnitre_sans_suite_client": np.nan,
    "code_postal": 59700,
    "operating_system": "Android",
    "marque": "Google",
    "model": "Pixel 7 Pro ",
    "ancienneté_de_contrat": 509555,
    "garantie": "Dommage",
    "Age": 43,
    "dossier_complet": 1,
    "decision_ai": 0,
    "nombre_prestation_ko": 0,
    "Nbr_ticket_pieces": 0,
    "Nbr_ticket_information": 4,
    "list_prest": "ADVANCED_SWAP",
    "delai_declaration": 279000,
    "delai_de_completude": np.nan,
    "delai_decision": 13090,
    "delai_reparation": 4,
    "delai_indemnisation": 4,
    "montant_indem": np.nan,
    "delai_Sinistre": 602000,
}]

X_new = pd.DataFrame(new_rows)

# add missing cols
for c in X_train.columns:
    if c not in X_new.columns:
        X_new[c] = np.nan

# align order
X_new = X_new[X_train.columns].copy()

# pd.NA -> np.nan
X_new = X_new.replace({pd.NA: np.nan})

# categorical to string + fill missing
for c in cat_cols:
    X_new[c] = X_new[c].astype("string").fillna("__MISSING__")

# numeric conversion
num_cols_new = [c for c in X_new.columns if c not in cat_cols]
for c in num_cols_new:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

pred_new_bayes = float(np.clip(cat_model_bayes_best.predict(X_new)[0], 0, 10))
print("\n=== Example prediction (Bayesian best) ===")
print("Predicted note:", pred_new_bayes)


# ==============================
# 9) SHAP — Global importance + Example explanation
# ==============================
sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()

explainer_bayes = shap.TreeExplainer(cat_model_bayes_best)
shap_values_bayes = explainer_bayes.shap_values(X_shap)

print("\nSHAP: global feature importance (Bayesian best)")
shap.summary_plot(shap_values_bayes, X_shap, show=True)

print("\nSHAP: explanation for the example row (Bayesian best)")
shap_values_one = explainer_bayes.shap_values(X_new)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[0],
        base_values=explainer_bayes.expected_value,
        data=X_new.iloc[0],
        feature_names=X_new.columns
    )
)
plt.show()

```
