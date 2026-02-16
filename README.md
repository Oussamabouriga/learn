```

Below is the same end-to-end documentation, but for XGBoost Regressor (predicting satisfaction from 0 to 10).
It includes:
	•	Encoding categorical + numeric data (OneHotEncoder + notes on alternatives)
	•	Handling imbalanced regression target (rare satisfaction scores) using:
	•	Stratified split via target bins
	•	Sample weights (best practical technique)
	•	Optional upsampling (if you want)
	•	Training XGBRegressor with all key hyperparameters explained
	•	Evaluation methods (metrics + plots + error-by-bins + CV) and what each tells you
	•	Predict from a dictionary (you type values manually)

I’m keeping it no custom helper functions — everything inline.

Install: pip install xgboost scikit-learn pandas numpy matplotlib joblib
XGBoost docs for parameters + sklearn interface:  ￼

⸻

Part A — Encoding & Preprocessing (DataFrame → model-ready)

1) Imports + load data

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

TARGET = "satisfaction"  # 0..10
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print(df.shape)
print(df.head())
print(y.describe())


⸻

2) Identify numeric vs categorical columns

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)


⸻

3) Preprocessing with ColumnTransformer + Pipeline (best practice)

Why this is best:
	•	prevents leakage (fit transforms only on train folds)
	•	ensures train/test have identical one-hot columns
	•	works with CV and hyperparameter tuning cleanly

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

3.1 Numeric pipeline
	•	median imputation (robust)
	•	scaling optional (XGBoost doesn’t require it; leave it False unless you want consistent preprocessing)

scale_numeric = False

numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
if scale_numeric:
    numeric_steps.append(("scaler", StandardScaler()))

num_pipe = Pipeline(steps=numeric_steps)

3.2 Categorical pipeline (OneHotEncoding)

For XGBoost, sparse one-hot is usually more memory-efficient, so we set sparse_output=True.

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

3.3 Combine

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

Key preprocessing hyperparameters (quick meaning)
	•	SimpleImputer(strategy="median"): numeric missing → median (robust to outliers)
	•	SimpleImputer(strategy="most_frequent"): categorical missing → most common label
	•	OneHotEncoder(handle_unknown="ignore"): unseen categories at prediction won’t crash
	•	OneHotEncoder(sparse_output=True): keeps one-hot as sparse matrix (faster/less RAM on many categories)

Note: XGBoost also has native categorical support via enable_categorical=True, but one-hot remains the most general and predictable baseline.  ￼

⸻

Part B — “Imbalanced” Target for Regression (0..10)

Imbalance in regression = some score ranges are rare (few 0–2, few 10).
If you ignore this, the model tends to “play safe” and predict near the mean.

We’ll do:
	1.	Stratified split using target bins
	2.	Sample weights by inverse bin frequency (recommended)

⸻

4) Stratified train/test split using target bins

from sklearn.model_selection import train_test_split

# Create bins from y (quantiles). If dataset is small, try q=4.
y_bins = pd.qcut(y, q=6, duplicates="drop")
print("Bin counts:\n", y_bins.value_counts())

X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X, y, y_bins,
    test_size=0.2,
    random_state=42,
    stratify=y_bins
)

print("Train bin distribution:\n", bins_train.value_counts(normalize=True))
print("Test bin distribution:\n", bins_test.value_counts(normalize=True))

What this shows / why it matters
	•	Your test set contains rare satisfaction ranges → evaluation is realistic.

⸻

5) Sample weights (best practical method)

# Compute bin frequencies on TRAIN
bin_freq = bins_train.value_counts(normalize=True)

# Weight each row by 1 / frequency of its bin
train_weights = bins_train.map(lambda b: 1.0 / bin_freq[b]).astype(float).values

# Normalize weights so mean ~ 1 (optional)
train_weights = train_weights / train_weights.mean()

print("weights min/mean/max:", train_weights.min(), train_weights.mean(), train_weights.max())

What it does
	•	rare target ranges get larger weight → model pays more attention to them during training.

⸻

6) Optional: upsampling rare bins (alternative)

Use only if weighting isn’t enough:

train_df = X_train.copy()
train_df[TARGET] = y_train.values
train_df["bin"] = bins_train.values

max_count = train_df["bin"].value_counts().max()

parts = []
for b, part in train_df.groupby("bin"):
    parts.append(part.sample(n=max_count, replace=True, random_state=42))

train_bal = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

X_train_bal = train_bal.drop(columns=[TARGET, "bin"])
y_train_bal = train_bal[TARGET]


⸻

Part C — XGBoost Regressor (training + hyperparameters)

7) Install + import

# pip install xgboost
from xgboost import XGBRegressor

XGBoost parameters reference:  ￼

⸻

8) XGBRegressor hyperparameters explained (what they control)

XGBoost builds trees sequentially (boosting). Each new tree corrects errors of the previous ensemble.

8.1 Main “learning dynamics”
	•	n_estimators: number of boosting rounds (trees). More = potentially better, but can overfit if too high.
	•	learning_rate (eta): shrinkage applied to each tree’s contribution.
	•	smaller learning_rate → need more trees but often generalizes better.
	•	early_stopping_rounds: stop training when validation metric stops improving (prevents overfit, saves time). Uses eval_set.  ￼

8.2 Tree structure / complexity
	•	max_depth: max depth of each tree.
	•	higher = more complex, more overfit risk.
	•	min_child_weight: minimum “weight” in a leaf (roughly minimum samples / hessian).
	•	higher = more conservative splits (less overfit).
	•	gamma: minimum loss reduction required to make a split.
	•	higher = fewer splits (regularization).

8.3 Randomness (helps generalization)
	•	subsample: fraction of rows used per tree.
	•	e.g. 0.8 reduces overfitting.
	•	colsample_bytree: fraction of features used per tree.
	•	colsample_bylevel / colsample_bynode: feature subsampling at level or split.

8.4 Regularization
	•	reg_lambda (lambda): L2 regularization.
	•	reg_alpha (alpha): L1 regularization.
	•	max_delta_step: limits the step size (rarely needed for regression; more common in imbalanced classification).

8.5 Objective & eval metric
	•	objective: regression loss
	•	"reg:squarederror" is standard MSE loss.  ￼
	•	eval_metric: metric monitored on eval_set
	•	"rmse", "mae" are common.

8.6 Speed / hardware
	•	tree_method
	•	"hist" is fast on CPU and recommended for most tabular problems.
	•	"gpu_hist" if you have GPU support.
	•	n_jobs: CPU threads.

⸻

9) Build the Pipeline (preprocess + XGBRegressor)

Here’s a strong baseline model for satisfaction 0..10:

xgb = XGBRegressor(
    objective="reg:squarederror",  # standard regression loss
    eval_metric="rmse",

    n_estimators=4000,        # large, because we'll use early stopping
    learning_rate=0.03,       # small lr => needs more trees, often better generalization

    max_depth=6,              # tree complexity
    min_child_weight=5,       # higher => more conservative (less overfit)
    gamma=0.0,                # >0 makes splitting harder (regularization)

    subsample=0.8,            # row sampling per tree
    colsample_bytree=0.8,     # feature sampling per tree

    reg_alpha=0.0,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization

    tree_method="hist",       # fast CPU method
    n_jobs=-1,
    random_state=42
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", xgb)
])

Quick tuning rules
	•	Overfitting → decrease max_depth, increase min_child_weight, set gamma>0, lower subsample/colsample, increase reg_alpha/reg_lambda
	•	Underfitting → increase max_depth, decrease min_child_weight, increase n_estimators (with early stopping), slightly increase learning_rate

⸻

10) Early stopping (recommended) + sample weights (imbalance)

We need a validation set for early stopping.
We’ll split train into train/valid (still stratified via bins if you want).

# Create bins from y_train for stratified train/valid split
train_bins2 = pd.qcut(y_train, q=6, duplicates="drop")

X_tr, X_val, y_tr, y_val, bins_tr, bins_val = train_test_split(
    X_train, y_train, train_bins2,
    test_size=0.2,
    random_state=42,
    stratify=train_bins2
)

# Compute weights on X_tr only (same idea)
freq_tr = bins_tr.value_counts(normalize=True)
w_tr = bins_tr.map(lambda b: 1.0 / freq_tr[b]).astype(float).values
w_tr = w_tr / w_tr.mean()

Now fit with:
	•	model__sample_weight=w_tr (weights for imbalance)
	•	model__eval_set=[(X_val, y_val)] (early stopping eval set)
	•	model__early_stopping_rounds=100 (stop if no improvement)

model.fit(
    X_tr, y_tr,
    model__sample_weight=w_tr,
    model__eval_set=[(X_val, y_val)],
    model__early_stopping_rounds=100,
    model__verbose=False
)

Early stopping behavior (what it shows):
	•	training continues until validation RMSE stops improving for 100 rounds, then stops and keeps best iteration.  ￼

⸻

11) Predict on test set

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 10)


⸻

Part D — Evaluation for Regression (what each method shows)

12) Metrics

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error,
    median_absolute_error
)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAE :", mae)      # avg absolute error in satisfaction points
print("RMSE:", rmse)     # penalizes big errors more
print("R2  :", r2)       # variance explained (can be negative)
print("ExplainedVariance:", evs)
print("MedianAE:", medae)
print("MAPE:", mape)

How to interpret
	•	MAE ~ “on average, I’m off by X points”
	•	RMSE >> MAE means “sometimes I make big mistakes”
	•	R²: overall quality; closer to 1 is better

⸻

13) Plots for understanding model behavior

A) Actual vs Predicted

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual satisfaction")
plt.ylabel("Predicted satisfaction")
plt.title("XGBoost: Actual vs Predicted")
plt.grid(True)
plt.show()

Shows:
	•	diagonal = perfect
	•	compression toward middle = model ignores extremes

B) Residual plot

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted satisfaction")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("XGBoost: Residuals vs Predicted")
plt.grid(True)
plt.show()

Shows:
	•	patterns = missing relationships / feature issues
	•	widening spread = errors larger in some ranges

⸻

14) Error by target range (checks imbalance fix worked)

df_eval = pd.DataFrame({"y": y_test.values, "pred": y_pred})
df_eval["bin"] = pd.qcut(df_eval["y"], q=6, duplicates="drop")
df_eval["abs_err"] = (df_eval["y"] - df_eval["pred"]).abs()

summary = df_eval.groupby("bin").agg(
    count=("y", "size"),
    mae=("abs_err", "mean")
).reset_index()

print(summary)

If rare bins have much bigger MAE → imbalance still hurting; increase weighting strength (more bins) or use upsampling.

⸻

15) Cross-validation (more reliable than one split)

from sklearn.model_selection import KFold, cross_val_predict

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
y_cv_pred = np.clip(y_cv_pred, 0, 10)

print("CV MAE :", mean_absolute_error(y, y_cv_pred))
print("CV RMSE:", np.sqrt(mean_squared_error(y, y_cv_pred)))
print("CV R2  :", r2_score(y, y_cv_pred))


⸻

16) Permutation importance (works with OneHot)

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# feature names (after one-hot)
feature_names = model.named_steps["prep"].get_feature_names_out()
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

print(imp.head(30))


⸻

Part E — Save/Load + Predict from a dictionary

17) Save pipeline

import joblib
joblib.dump(model, "satisfaction_xgb_pipeline.joblib")

18) Load and predict from dictionary (manual input)

loaded = joblib.load("satisfaction_xgb_pipeline.joblib")

sample = {
    # Use YOUR real column names:
    "age": 29,
    "city": "Tunis",
    "subscription_type": "premium",
    "rides_last_month": 10,
    "salary": 2500
}

sample_df = pd.DataFrame([sample])

pred = loaded.predict(sample_df)[0]
pred = float(np.clip(pred, 0, 10))

print("Predicted satisfaction:", pred)
print("Rounded 0..10:", int(np.round(pred)))

Because we used handle_unknown="ignore", new categories won’t crash prediction.

⸻

If you want, paste:
	•	df.dtypes
	•	and tell me roughly how many rows you have (1k / 10k / 1M)

…and I’ll give you a recommended XGBoost config (best max_depth/min_child_weight/subsample/colsample/learning_rate for that scale) and a ready-made sample dictionary with your real columns.

```
