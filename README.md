```
Below is a complete, practical documentation for using Random Forest Regression in Python on a pandas DataFrame: from data loading → cleaning → splitting → training → hyperparameters → evaluation (many metrics + cross-validation) → interpretation → saving/loading.

⸻

1) What is Random Forest Regression?

RandomForestRegressor is an ensemble model that trains many decision trees on different random subsets of:
	•	rows (bootstrap samples),
	•	and features (random feature selection at each split),

Then it predicts by taking the average of all tree predictions.

Why it’s useful
	•	Works well on tabular data
	•	Handles non-linear relationships
	•	Robust to outliers/noise
	•	Minimal feature scaling needed

⸻

2) Installation & Imports

pip install pandas numpy scikit-learn matplotlib joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error
)

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib


⸻

3) Load Data into a DataFrame

Example (CSV)

df = pd.read_csv("data.csv")
print(df.shape)
print(df.head())


⸻

4) Decide your target and features

Let’s assume your target column is named "target".

TARGET = "target"

X = df.drop(columns=[TARGET])
y = df[TARGET]


⸻

5) Cleaning & Preprocessing (DataFrame-friendly)

5.1 Handle missing values (simple baseline)

Random Forest in scikit-learn does not accept NaNs (in most versions/configs), so fill or impute:

# numeric columns
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# categorical columns (if any)
cat_cols = X.select_dtypes(exclude=[np.number]).columns
X[cat_cols] = X[cat_cols].fillna("missing")

5.2 Encode categorical columns

RandomForestRegressor needs numeric inputs.

Option A: One-hot encoding (recommended baseline)

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

Note: scaling (StandardScaler) is usually not needed for random forests.

⸻

6) Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape, X_test.shape)


⸻

7) Train a Random Forest Regressor

7.1 Basic model

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

7.2 Predict

y_pred = rf.predict(X_test)


⸻

8) Training Hyperparameters (ALL important ones explained)

Below are the most used parameters and what they do:

Forest-level
	•	n_estimators: number of trees. More trees → usually better but slower.
	•	random_state: reproducibility.
	•	n_jobs: CPU parallelism (-1 = use all cores).
	•	bootstrap: sample rows with replacement (default True). If False, each tree uses all rows (less randomness).
	•	oob_score: “out-of-bag” score; internal validation if bootstrap=True.

Tree complexity / overfitting control
	•	max_depth: maximum tree depth. Smaller → less overfit.
	•	min_samples_split: minimum samples required to split a node.
	•	min_samples_leaf: minimum samples in a leaf. Increasing it smooths predictions.
	•	max_leaf_nodes: limit leaf count.
	•	min_impurity_decrease: split only if it improves impurity enough.

Feature randomness
	•	max_features: number of features considered at each split.
	•	1.0 or None: all features
	•	"sqrt": sqrt(num_features)
	•	"log2": log2(num_features)
	•	or a float (fraction) / int

Split criterion
	•	criterion: regression objective:
	•	"squared_error" (default) → MSE
	•	"absolute_error" → MAE (more robust to outliers, slower)
	•	"friedman_mse" often good for boosting-like splits
	•	"poisson" for count-like targets (only if y >= 0)

⸻

9) Evaluation Metrics (and how to apply)

9.1 Core regression metrics

def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "ExplainedVariance": evs,
        "MAPE": mape,
        "MedianAE": medae,
    }

report = regression_report(y_test, y_pred)
print(pd.Series(report).sort_index())

9.2 What each metric means
	•	MAE: average absolute error. Easy to interpret (same unit as target).
	•	MSE: squares errors, punishes large errors strongly.
	•	RMSE: sqrt(MSE), same unit as target, sensitive to big errors.
	•	R²: how much variance is explained (1.0 best, can be negative).
	•	Explained Variance: like R² but focuses on variance of errors.
	•	MAPE: percentage error; can explode if y is near 0.
	•	MedianAE: robust to outliers (median absolute error).

⸻

10) Cross-Validation (Better evaluation)

Instead of one split, use K-fold CV.

10.1 Cross-validated R²

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(
    rf, X, y,
    cv=cv,
    scoring="r2",
    n_jobs=-1
)
print("CV R2 mean:", cv_r2.mean())
print("CV R2 std :", cv_r2.std())

10.2 Cross-validated MAE (note: sklearn uses NEGATIVE for errors)

cv_neg_mae = cross_val_score(
    rf, X, y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
cv_mae = -cv_neg_mae
print("CV MAE mean:", cv_mae.mean())
print("CV MAE std :", cv_mae.std())

10.3 Cross-validated predictions (for diagnostics plots)

y_cv_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)


⸻

11) Hyperparameter Tuning (RandomizedSearch + GridSearch)

11.1 RandomizedSearchCV (recommended first)

param_dist = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 5, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
    "bootstrap": [True, False],
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best params:", search.best_params_)

11.2 Evaluate tuned model

y_pred_best = best_model.predict(X_test)
print(pd.Series(regression_report(y_test, y_pred_best)))


⸻

12) Feature Importance (2 ways)

12.1 Built-in importance (fast, can be biased)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(20))

12.2 Permutation importance (more reliable)

perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print(perm_imp.head(20))


⸻

13) Diagnostics Plots (Actual vs Predicted + Residuals)

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals"):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_actual_vs_pred(y_test, y_pred, "RF: Actual vs Predicted (Test)")
plot_residuals(y_test, y_pred, "RF: Residuals (Test)")


⸻

14) Save & Load the model

joblib.dump(rf, "random_forest_model.joblib")

rf_loaded = joblib.load("random_forest_model.joblib")
pred = rf_loaded.predict(X_test)


⸻

15) Full End-to-End Template (Copy/Paste)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1) Load
df = pd.read_csv("data.csv")

# 2) Target + features
TARGET = "target"
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# 3) Missing values
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("missing")

# 4) Encode
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 5) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Train baseline
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 7) Evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print({"MAE": mae, "RMSE": rmse, "R2": r2})

# 8) Cross-val
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=-1)
print("CV R2 mean:", cv_r2.mean(), "std:", cv_r2.std())

# 9) Save
joblib.dump(rf, "rf.joblib")


⸻

If you paste a small sample of your DataFrame columns (like df.head() + target name), I can tailor:
	•	the preprocessing (encoding/imputation),
	•	the tuning space (ranges that make sense),
	•	and the evaluation report (your preferred metrics like MAE vs RMSE vs MAPE).

```
