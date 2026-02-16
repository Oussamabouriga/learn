```
Perfect — 0 to 10 means you’re doing REGRESSION, not classification.
So I’m remaking the documentation for:
	1.	Encoding categorical + numeric data (OneHotEncoder + alternatives)
	2.	Handling an “imbalanced” target in regression (rare satisfaction scores) so the model doesn’t ignore them
	3.	Training a model end-to-end (Pipeline + RandomForestRegressor with hyperparameters)
	4.	Evaluation techniques for regression (what each shows + how to interpret)
	5.	Predicting from a dictionary (you type the data manually)

And as you asked: no custom helper functions — all code is inline.

⸻

Part A — Encoding & Preprocessing (DataFrame → model-ready)

1) Load data and define target

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

TARGET = "satisfaction"   # 0..10
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print(df.shape)
print(df.head())
print(y.describe())


⸻

2) Identify numeric vs categorical columns

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric:", num_cols)
print("Categorical:", cat_cols)


⸻

3) Best practice preprocessing: ColumnTransformer + Pipeline

Why:
	•	avoids leakage (fit preprocessing only on train folds)
	•	keeps same columns between train/test (critical with one-hot)
	•	works with CV and hyperparameter tuning

3.1 Imports

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

3.2 Build preprocessing (OneHot for categorical)

✅ Numeric: median impute (and optional scaling)
✅ Categorical: most_frequent impute + OneHotEncoder

scale_numeric = False  # For RandomForest: usually False. True if you later try linear/SVM/kNN.

numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
if scale_numeric:
    numeric_steps.append(("scaler", StandardScaler()))
num_pipe = Pipeline(steps=numeric_steps)

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

Key preprocessing hyperparameters explained

SimpleImputer
	•	strategy="median" (numeric): robust to outliers (best default)
	•	strategy="most_frequent" (categorical): fills missing with most common category

OneHotEncoder
	•	handle_unknown="ignore": if you later type a category never seen → no crash
	•	sparse_output=False: easier to inspect; set True for huge data

StandardScaler
	•	needed for distance/linear models
	•	not required for trees (RandomForest)

⸻

Part B — Handling “Imbalanced” Target in Regression (0..10)

With satisfaction 0..10, “imbalance” means:
	•	you may have many 7/8/9 and very few 0/1/2
	•	the model learns to predict the common scores and ignores rare ones

We solve this with (1) stratified split using bins and (2) sample weighting.
Optionally, (3) upsampling rare bins.

⸻

4) Stratified train/test split for regression (using target bins)

train_test_split(stratify=...) expects categories → we create bins from y.

# Create bins of y using quantiles
# If dataset is small, use q=4; if large, q=8 is fine
y_bins = pd.qcut(y, q=6, duplicates="drop")
print("Bin counts:\n", y_bins.value_counts())

X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X, y, y_bins,
    test_size=0.2,
    random_state=42,
    stratify=y_bins
)

print("Train bins:\n", bins_train.value_counts(normalize=True))
print("Test bins:\n", bins_test.value_counts(normalize=True))

What this achieves
	•	rare satisfaction ranges appear in both train and test
	•	evaluation becomes fair (you’ll see if model fails on rare scores)

⸻

5) Sample weights (best technique for imbalanced regression)

We weight examples from rare bins higher.

# Frequency of each bin in TRAIN
bin_freq = bins_train.value_counts(normalize=True)

# Weight per row = 1 / freq(bin)
train_weights = bins_train.map(lambda b: 1.0 / bin_freq[b]).astype(float).values

# Normalize (optional): keep mean weight around 1
train_weights = train_weights / train_weights.mean()

print("weights: min/mean/max =", train_weights.min(), train_weights.mean(), train_weights.max())

Interpretation
	•	if a bin is rare, its samples get larger weights
	•	the model “cares” more about those rare ranges

⸻

6) Optional: Upsample rare target bins (alternative or additional)

This duplicates rare samples so the learner sees them more often.

Use this only if needed (weights usually enough).
Here’s a simple upsampling approach in pure pandas:

train_df = X_train.copy()
train_df[TARGET] = y_train.values
train_df["bin"] = bins_train.values

# Target count = max bin size (fully balanced)
max_count = train_df["bin"].value_counts().max()

balanced_parts = []
for b, part in train_df.groupby("bin"):
    # sample with replacement to reach max_count
    part_up = part.sample(n=max_count, replace=True, random_state=42)
    balanced_parts.append(part_up)

train_bal = pd.concat(balanced_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

X_train_bal = train_bal.drop(columns=[TARGET, "bin"])
y_train_bal = train_bal[TARGET]

If you use upsampling, you can train without sample_weight (or still use weights lightly).

⸻

Part C — Train a model (Random Forest Regression) with hyperparameters explained

7) Imports

from sklearn.ensemble import RandomForestRegressor

8) RandomForestRegressor hyperparameters (detailed)

Here is a full model with key hyperparameters shown:

rf = RandomForestRegressor(
    # --- Core / stability ---
    n_estimators=600,          # number of trees. More = more stable, slower.
    random_state=42,           # reproducibility
    n_jobs=-1,                 # parallel CPU usage

    # --- Split criterion ---
    criterion="squared_error", # MSE objective. Alternative: "absolute_error" (more robust, slower)

    # --- Control overfitting (tree complexity) ---
    max_depth=None,            # None allows deep trees (can overfit)
    min_samples_split=2,       # min samples required to split a node (bigger => simpler)
    min_samples_leaf=1,        # min samples in leaf (bigger => smoother predictions)
    max_leaf_nodes=None,       # limit leaf nodes (int). Helps regularize.
    min_impurity_decrease=0.0, # require impurity improvement to split (regularize)

    # --- Randomness (features) ---
    max_features="sqrt",       # features considered per split: "sqrt", "log2", float, int
                               # with many one-hot columns, "sqrt" often generalizes better

    # --- Randomness (rows) ---
    bootstrap=True,            # sample rows with replacement
    max_samples=None,          # if bootstrap=True, set like 0.8 for more diversity

    # --- Advanced regularization ---
    ccp_alpha=0.0,             # pruning strength (0 means no pruning)
    oob_score=False            # out-of-bag score (only if bootstrap=True)
)

How to tune quickly (rules of thumb)

If you overfit (train great, test bad):
	•	set min_samples_leaf=3 or 5
	•	set max_depth=10 or 20
	•	set max_samples=0.8
	•	consider ccp_alpha > 0

If you underfit (both bad):
	•	increase n_estimators
	•	allow more depth (max_depth=None)
	•	reduce min_samples_leaf

⸻

9) Build the full pipeline (preprocessing + model)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", rf)
])


⸻

10) Train (with sample weights) — recommended

model.fit(X_train, y_train, model__sample_weight=train_weights)

If you use upsampled training set instead:

# model.fit(X_train_bal, y_train_bal)


⸻

11) Predict

y_pred = model.predict(X_test)

# Keep inside 0..10 (optional, but recommended)
y_pred = np.clip(y_pred, 0, 10)


⸻

Part D — Evaluation techniques for Regression (what each shows + interpretation)

12) Metrics

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("ExplainedVariance:", evs)
print("MedianAE:", medae)
print("MAPE:", mape)

What each metric tells you
	•	MAE: average error in satisfaction points (easy to explain)
	•	RMSE: punishes large errors more (sensitive to big misses)
	•	R²: how much variance explained (1 best; can be negative if bad)
	•	Explained Variance: similar to R²; checks variance capture
	•	MedianAE: “typical” error ignoring extreme outliers
	•	MAPE: percent error (can be weird when y≈0)

⸻

13) Plots that explain model results

A) Actual vs Predicted

Shows bias and whether predictions are compressed (not predicting extremes).

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual satisfaction")
plt.ylabel("Predicted satisfaction")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

How to interpret
	•	Points near diagonal → good
	•	If model avoids 0/10 → predictions cluster around middle

⸻

B) Residuals vs Predicted

Detects patterns: missing features, nonlinearity, heteroscedasticity.

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted satisfaction")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()

How to interpret
	•	random cloud around 0 → good
	•	curve pattern → model missing relationship
	•	spread increases in some range → errors larger there

⸻

14) Evaluate fairness across target ranges (MOST IMPORTANT for imbalance)

This checks whether rare ranges still have worse error.

df_eval = pd.DataFrame({"y": y_test.values, "pred": y_pred})
df_eval["bin"] = pd.qcut(df_eval["y"], q=6, duplicates="drop")
df_eval["abs_err"] = (df_eval["y"] - df_eval["pred"]).abs()

bin_summary = df_eval.groupby("bin").agg(
    count=("y", "size"),
    mae=("abs_err", "mean")
).reset_index()

print(bin_summary)

Interpretation
	•	If low/high bins have much larger MAE → imbalance still hurts
	•	Weighting/upsampling should reduce this gap

⸻

15) Cross-validation (more reliable)

from sklearn.model_selection import KFold, cross_val_predict

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
y_cv_pred = np.clip(y_cv_pred, 0, 10)

print("CV MAE :", mean_absolute_error(y, y_cv_pred))
print("CV RMSE:", np.sqrt(mean_squared_error(y, y_cv_pred)))
print("CV R2  :", r2_score(y, y_cv_pred))

Interpretation
	•	CV performance is more trustworthy than a single split

⸻

16) Feature importance (Permutation importance)

Best for one-hot encoded features.

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_names = model.named_steps["prep"].get_feature_names_out()
importances = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

print(importances.head(30))

What it shows
	•	If shuffling a feature increases error a lot → feature is important
	•	Correlated features share importance (may reduce each other’s score)

⸻

Part E — Save/Load + Predict from Dictionary (your manual test)

17) Save the pipeline

import joblib
joblib.dump(model, "satisfaction_rf_pipeline.joblib")

18) Load and predict from a dictionary

loaded = joblib.load("satisfaction_rf_pipeline.joblib")

sample = {
    # Put YOUR real column names here:
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

Why it won’t crash with new categories
	•	OneHotEncoder has handle_unknown="ignore"

⸻

If you want it “perfect” for your dataset

Send me:
	•	df.dtypes (just the output)
	•	and the exact TARGET column name

Then I’ll rewrite the sample = {...} dictionary with your real columns + recommend the best RF hyperparameters for your dataset size.


```
