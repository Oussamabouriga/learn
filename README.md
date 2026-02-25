```
pip install optuna



import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool

# ------------------------------------------------------------
# Assumes you already have:
# X_train, y_train, cat_cols, weights_train
# ------------------------------------------------------------

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,

        # Core params to tune
        "iterations": trial.suggest_int("iterations", 400, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),

        # Regularization / randomness
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

        # Speed/robust
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
    }

    fold_rmses = []

    for tr_idx, va_idx in cv.split(X_train):
        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        w_tr = np.asarray(weights_train)[tr_idx]

        X_va = X_train.iloc[va_idx].copy()
        y_va = y_train.iloc[va_idx].copy()

        # Build Pools (important for cat + weights)
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=w_tr)
        val_pool   = Pool(X_va, y_va, cat_features=cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=200)

        pred_va = np.clip(model.predict(X_va), 0, 10)
        rmse_va = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        fold_rmses.append(rmse_va)

    return float(np.mean(fold_rmses))


# -------------------------
# Run Bayesian optimization
# -------------------------
study = optuna.create_study(direction="minimize")  # minimize RMSE
study.optimize(objective, n_trials=40)  # try 30-80 typically

print("✅ Best RMSE:", study.best_value)
print("✅ Best params:", study.best_params)

best_bayes_params = study.best_params
```
