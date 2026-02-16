

```


Below is the same full documentation, but no custom functions at all. It’s written as a step-by-step notebook style script so you can read it line by line and understand exactly what happens.

The example goal: predict satisfaction (0..10) from a DataFrame with numeric + categorical features, using:
	•	OneHotEncoding for categorical data
	•	Imputation for missing values
	•	RandomForestRegressor with many hyperparameters shown
	•	Imbalanced target handling (regression) via target binning + sample weights
	•	Evaluation metrics + plots + CV
	•	Prediction from a dictionary

⸻

1) Imports

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

2) Load your data

df = pd.read_csv("data.csv")

# Example target column
TARGET = "satisfaction"  # should be 0..10

print(df.shape)
print(df.head())


⸻

3) Separate X and y (features / target)

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()


⸻

4) Detect numeric vs categorical columns

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)


⸻

5) Handle “imbalanced regression target” BEFORE split (stratified split using bins)

Why?

If satisfaction values are not uniformly distributed (ex: many 7–9, few 0–2), a random split might put almost all rare values in train OR test, which makes evaluation misleading.

We create bins from the target using quantiles.

# Create target bins for stratified splitting
# q = number of bins; 6 is a good start
y_bins = pd.qcut(y, q=6, duplicates="drop")
print(y_bins.value_counts())

Now do a stratified split using these bins:

X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X, y, y_bins,
    test_size=0.2,
    random_state=42,
    stratify=y_bins
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train bin distribution:\n", bins_train.value_counts(normalize=True))
print("Test bin distribution:\n", bins_test.value_counts(normalize=True))


⸻

6) Compute sample weights on training set (to help rare target ranges)

Idea

We want rare bins to matter more. Weight = inverse frequency of the bin.

# Frequency of each bin in TRAIN
bin_freq = bins_train.value_counts(normalize=True)

# Weight each row by 1 / freq(bin)
train_weights = bins_train.map(lambda b: 1.0 / bin_freq[b]).astype(float).values

# Normalize weights so average weight ~ 1 (optional)
train_weights = train_weights / train_weights.mean()

print("Weights summary:", np.min(train_weights), np.mean(train_weights), np.max(train_weights))

What this does
	•	Rare satisfaction ranges (ex: low scores) get higher weights.
	•	The forest tries harder to reduce error on them.

⸻

7) Build preprocessing (no functions): numeric + categorical pipelines

7.1 Numeric pipeline
	•	Fill missing numeric values with median
	•	Scaling is optional for forests (not needed).
I will not scale to keep it simple and correct for RF.

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

7.2 Categorical pipeline
	•	Fill missing categories with most frequent
	•	One-hot encode categories
	•	handle_unknown=“ignore” lets you predict later even if you type a new category in a dictionary

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

7.3 Combine them with ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)


⸻

8) RandomForestRegressor — put a lot of hyperparameters (and understand each)

rf = RandomForestRegressor(
    # --- Forest size / stability ---
    n_estimators=600,        # number of trees (more = stable but slower)
    random_state=42,         # reproducibility
    n_jobs=-1,               # all CPU cores

    # --- Split criterion ---
    criterion="squared_error",  # MSE objective (default). Alternative: "absolute_error"

    # --- Tree complexity (controls overfitting) ---
    max_depth=None,          # None = trees grow deep (can overfit)
    min_samples_split=2,     # min samples to split a node (bigger => less complex)
    min_samples_leaf=1,      # min samples in leaf (bigger => smoother predictions)
    max_leaf_nodes=None,     # limit leaf nodes (set int to simplify trees)
    min_impurity_decrease=0.0, # require improvement to split (regularization)

    # --- Feature randomness ---
    max_features="sqrt",     # features considered at each split
                             # "sqrt" often works well with many features (esp. one-hot)

    # --- Row randomness ---
    bootstrap=True,          # sample rows with replacement per tree
    max_samples=None,        # if bootstrap=True, you can set e.g. 0.8 to use 80% per tree

    # --- Other regularization ---
    ccp_alpha=0.0,           # pruning strength (0 = no pruning)

    # --- OOB (optional internal validation) ---
    oob_score=False,         # works only if bootstrap=True
)

How to interpret these hyperparameters quickly
	•	If model overfits (train great, test bad):
increase min_samples_leaf, reduce max_depth, set max_samples=0.8, or use ccp_alpha>0.
	•	If model underfits (both train & test bad):
increase n_estimators, allow deeper trees (max_depth=None), reduce min_samples_leaf.

⸻

9) Build the full Pipeline (preprocess → model)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", rf)
])


⸻

10) Train the model using sample weights (important)

In Pipeline, you pass weights like this: model__sample_weight.

model.fit(X_train, y_train, model__sample_weight=train_weights)


⸻

11) Predict on test set

y_pred = model.predict(X_test)

Optional: keep predictions in [0,10]

y_pred_clipped = np.clip(y_pred, 0, 10)


⸻

12) Evaluation metrics (what each shows)

mae = mean_absolute_error(y_test, y_pred_clipped)
mse = mean_squared_error(y_test, y_pred_clipped)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_clipped)
evs = explained_variance_score(y_test, y_pred_clipped)
mape = mean_absolute_percentage_error(y_test, y_pred_clipped)
medae = median_absolute_error(y_test, y_pred_clipped)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)
print("ExplainedVariance:", evs)
print("MAPE:", mape)
print("MedianAE:", medae)

Interpretation
	•	MAE: average error in satisfaction points. (best for “human scale”)
	•	RMSE: punishes big errors. If RMSE >> MAE, you have some large mistakes.
	•	R²: percent of variance explained. Negative means worse than predicting mean.
	•	Explained Variance: similar to R²; helps detect offset vs variance capture.
	•	MAPE: % error; not great near y=0.
	•	MedianAE: typical error ignoring big outliers.

⸻

13) Plots to understand behavior

13.1 Actual vs Predicted

Shows if model is biased or compressed (not predicting extremes).

plt.figure()
plt.scatter(y_test, y_pred_clipped)
plt.xlabel("Actual satisfaction")
plt.ylabel("Predicted satisfaction")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

Interpretation
	•	Good model → points near diagonal
	•	If it never predicts 0 or 10 → “compressed band” in the middle

13.2 Residuals vs Predicted

Residual = Actual − Predicted. Shows patterns and where it fails.

residuals = y_test - y_pred_clipped

plt.figure()
plt.scatter(y_pred_clipped, residuals)
plt.axhline(0)
plt.xlabel("Predicted satisfaction")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()

Interpretation
	•	Random cloud around 0 is good
	•	Pattern/curve means missing relationship/features

⸻

14) Error by target range (check imbalance fix)

This tells you if rare satisfaction values still have high error.

df_err = pd.DataFrame({
    "y": y_test.values,
    "pred": y_pred_clipped
})

df_err["bin"] = pd.qcut(df_err["y"], q=6, duplicates="drop")
df_err["abs_err"] = (df_err["y"] - df_err["pred"]).abs()

summary = df_err.groupby("bin").agg(
    count=("y", "size"),
    mae=("abs_err", "mean")
).reset_index()

print(summary)

Interpretation
	•	If low/high bins have much higher MAE → model struggles on rare ranges
	•	If weights worked → MAE becomes more uniform across bins

⸻

15) Cross-validation evaluation (more reliable than one split)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated predictions (for overall report)
y_cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
y_cv_pred = np.clip(y_cv_pred, 0, 10)

print("CV MAE :", mean_absolute_error(y, y_cv_pred))
print("CV RMSE:", np.sqrt(mean_squared_error(y, y_cv_pred)))
print("CV R2  :", r2_score(y, y_cv_pred))

Interpretation
	•	More stable estimate of performance on unseen data
	•	If CV results are much worse than test split → your split was “lucky” or data has drift

⸻

16) Feature importance (Permutation Importance)

Best way to interpret features with one-hot encoding.

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_names = model.named_steps["prep"].get_feature_names_out()
importances = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

print(importances.head(30))

Interpretation
	•	Higher importance = shuffling this feature harms predictions more
	•	Correlated features share importance

⸻

17) Save & load model (pipeline includes preprocessing!)

joblib.dump(model, "satisfaction_model.joblib")
loaded_model = joblib.load("satisfaction_model.joblib")


⸻

18) Predict from a dictionary (you type values yourself)

Example dictionary (YOU must use your real column names)

sample_input = {
    # numeric columns examples
    "age": 29,
    "rides_last_month": 12,
    "salary": 2500,

    # categorical columns examples
    "city": "Tunis",
    "subscription_type": "premium",
    "car_owner": "yes"
}

sample_df = pd.DataFrame([sample_input])

pred = loaded_model.predict(sample_df)[0]
pred = float(np.clip(pred, 0, 10))

print("Predicted satisfaction:", pred)
print("Rounded (0..10 integer):", int(np.round(pred)))

Why this works
	•	OneHotEncoder handles categories
	•	handle_unknown="ignore" prevents crash if category is new
	•	Pipeline applies the exact same preprocessing as training

⸻

19) What to change first if results are not good

If predictions are too “average” (always around 6–8)
	•	Increase importance of rare ranges:
	•	increase bins (q=8)
	•	keep weights
	•	Increase model capacity:
	•	max_features=1.0 (or 0.7)
	•	more trees: n_estimators=1000

If overfitting (good train, bad test)
	•	regularize:
	•	min_samples_leaf=3 or 5
	•	max_depth=10 or 20
	•	max_samples=0.8

⸻

If you paste your DataFrame columns (just df.dtypes output), I can rewrite the last “dictionary example” with your real feature names so you can copy/paste and test immediately.
```
