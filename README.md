```

Below is a full, detailed documentation (practical + copy-paste code) for training a model that predicts Satisfaction (0→10) from a pandas DataFrame using:
	•	Categorical encoding (OneHotEncoder + alternatives)
	•	Numeric preprocessing (missing values, optional scaling)
	•	Random Forest Regression (with all key hyperparameters explained)
	•	Imbalanced target handling for regression (rare scores don’t get ignored)
	•	Evaluation techniques (what each metric/plot shows + how to interpret)
	•	Predicting from a dictionary (you type the input yourself and test results)

Everything is built as a single Pipeline so preprocessing is consistent at training + prediction.

⸻

0) Important note: “imbalance” for regression

Imbalance for regression means: some target values are rare (e.g., very few 0/1 or 10), so the model might learn mostly the common values and perform poorly on rare extremes.

Good solutions:
	1.	Stratified split via target binning (so train/test both contain rare ranges)
	2.	Sample weights (give rare target ranges more importance during training)
	3.	(Optional) Upsampling rare target ranges in training data

Random Forest supports sample_weight in .fit() → very useful.

⸻

1) Setup (install + imports)

pip install pandas numpy scikit-learn matplotlib joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
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

2) Choose target + split columns (DataFrame → X, y)

TARGET = "satisfaction"  # 0..10

def split_columns(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return X, y, num_cols, cat_cols


⸻

3) Encoding categorical + preprocessing numeric (BEST PRACTICE)

3.1 Recommended: OneHotEncoder + imputers (pipeline-safe)

def make_preprocessor(num_cols, cat_cols, scale_numeric=False):
    # Numeric features: impute missing; optionally scale (not needed for forests)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    # Categorical: impute missing + OneHot encode
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

Hyperparameters explained (preprocessing)

SimpleImputer
	•	strategy="median": robust for numeric (outliers won’t distort like mean)
	•	strategy="most_frequent": best simple default for categories

OneHotEncoder
	•	handle_unknown="ignore": if you type a new category later (not seen in training), prediction still works (it becomes all-zeros for that category column)
	•	sparse_output=False: easier to debug; for huge datasets you can set it to True (saves memory)

StandardScaler
	•	useful for linear models / SVM / kNN
	•	not required for tree models (RandomForest), but you can keep it off

⸻

3.2 Alternative encodings (when OneHot isn’t ideal)

A) Ordinal encoding (ONLY if categories are truly ordered)

Example: ["low","medium","high"].
For unordered categories (city names), ordinal encoding is usually wrong.

B) Target encoding (best for high-cardinality like 5k cities)

Replaces each category by a smoothed mean target. Powerful but must be done carefully (avoid leakage).
If you want this, tell me and I’ll give you a leak-safe CV version.

For your request, we’ll use OneHotEncoder because it’s safest and standard.

⸻

4) Handling imbalanced target (regression) properly

4.1 Stratified split using target bins (recommended)

train_test_split has stratify= only for classification, so we bin the target.

def make_target_bins(y, n_bins=6):
    # Quantile bins: each bin has (roughly) same number of samples
    # Good when you have imbalance across ranges.
    bins = pd.qcut(y, q=n_bins, duplicates="drop")
    return bins

Then:

X, y, num_cols, cat_cols = split_columns(df, TARGET)
y_bins = make_target_bins(y, n_bins=6)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y_bins
)

What it shows / why it matters
	•	Ensures your test set contains examples from rare satisfaction ranges (0–2, 9–10, etc.)
	•	Prevents “easy” splits where test contains only common values

⸻

4.2 Sample weights to “rebalance” rare target ranges (recommended)

We compute bin frequencies and give higher weight to rare bins.

def compute_sample_weights(y, n_bins=6):
    bins = pd.qcut(y, q=n_bins, duplicates="drop")
    freq = bins.value_counts(normalize=True)
    weights = bins.map(lambda b: 1.0 / freq[b]).astype(float)
    # Normalize weights so average weight ~ 1 (optional)
    weights = weights / weights.mean()
    return weights.values

Usage:

train_weights = compute_sample_weights(y_train, n_bins=6)

What it does
	•	Rare target ranges contribute more to the loss reduction decisions in trees
	•	Helps model not ignore rare satisfaction extremes

⸻

4.3 Optional: Upsampling rare target ranges (alternative)

This duplicates training rows from rare bins to balance counts.

Sample weights is usually cleaner than duplication, but if you want upsampling too, tell me.

⸻

5) Random Forest Regression — hyperparameters explained (in detail)

RandomForestRegressor is a collection of many decision trees. Hyperparameters control:

5.1 “How many trees and randomness”
	•	n_estimators: number of trees
	•	More trees → better stability, but slower
	•	Typical: 200–1000
	•	random_state: reproducibility
	•	n_jobs: CPU usage (-1 = all cores)
	•	bootstrap: whether each tree trains on a bootstrap sample (default True)
	•	True increases diversity and reduces overfitting
	•	oob_score: “out-of-bag” scoring (only works if bootstrap=True)
	•	Gives a built-in validation estimate without separate CV (still, CV is better)

5.2 “Tree size / overfitting control”
	•	max_depth: max depth of each tree
	•	None = trees can grow deep (risk overfit)
	•	smaller values generalize better
	•	min_samples_split: minimum samples required to split a node
	•	larger → fewer splits → less overfit
	•	min_samples_leaf: minimum samples in a leaf
	•	larger → smoother predictions, better generalization
	•	max_leaf_nodes: limit number of leaf nodes
	•	forces simpler trees
	•	min_impurity_decrease: only split if it improves impurity by at least this amount
	•	stronger regularization, good to reduce overfitting

5.3 “Feature randomness”
	•	max_features: number of features considered at each split
	•	"sqrt", "log2", float fraction, integer count
	•	More randomness often generalizes better, especially with many features from one-hot encoding

5.4 “Split criterion”
	•	criterion
	•	"squared_error": optimizes MSE (common default)
	•	"absolute_error": optimizes MAE (more robust to outliers, slower)
	•	"friedman_mse": variant often used for boosting-like gains

5.5 “Other”
	•	ccp_alpha: cost-complexity pruning strength
	•	0 prunes the tree; helps reduce overfit (advanced)
	•	max_samples (only if bootstrap=True): fraction/number of samples used per tree
	•	smaller → more diversity, sometimes better generalization

⸻

6) Full training code (Pipeline + weights + full hyperparameters)

def build_rf_model(num_cols, cat_cols):
    preprocessor = make_preprocessor(num_cols, cat_cols, scale_numeric=False)

    rf = RandomForestRegressor(
        n_estimators=600,
        criterion="squared_error",
        max_depth=None,

        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,

        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        bootstrap=True,
        oob_score=False,      # set True if you want OOB estimate
        max_samples=None,     # set e.g. 0.8 for more randomness

        random_state=42,
        n_jobs=-1,

        ccp_alpha=0.0,
        verbose=0
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", rf)
    ])
    return pipe

Train:

df = pd.read_csv("data.csv")

X, y, num_cols, cat_cols = split_columns(df, TARGET)

y_bins = make_target_bins(y, n_bins=6)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

model = build_rf_model(num_cols, cat_cols)

train_weights = compute_sample_weights(y_train, n_bins=6)

# IMPORTANT: in sklearn Pipeline, pass fit params with step name:
model.fit(X_train, y_train, model__sample_weight=train_weights)

y_pred = model.predict(X_test)


⸻

7) Evaluation techniques (what they show + how to interpret)

7.1 Metrics (numbers)

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
        "MedianAE": medae
    }

print(pd.Series(regression_report(y_test, y_pred)).sort_index())

What each metric shows

MAE (Mean Absolute Error)
	•	Average absolute error in the target unit (here: satisfaction points).
	•	If MAE=0.6 → on average the model is off by 0.6 points.

MSE (Mean Squared Error)
	•	Squares errors → punishes big mistakes strongly.
	•	Good when you really want to penalize large errors.

RMSE
	•	sqrt(MSE) in the same unit (satisfaction points).
	•	Often easier to interpret than MSE and still punishes big errors.

R²
	•	How much variance in satisfaction is explained (1 is best).
	•	R² can be negative if the model is worse than predicting the mean.

Explained Variance
	•	Similar to R², focuses on variance captured by the prediction.
	•	If high but R² lower, it can indicate bias/offset.

MAPE
	•	Error as percentage.
	•	Not great if target has zeros or near-zero values (can blow up). For 0..10 it can be unstable when y≈0.

MedianAE
	•	Median absolute error: robust to outliers.
	•	If MAE is high but MedianAE is low → errors are usually small but sometimes very large.

⸻

7.2 Plots that explain model behavior

A) Actual vs Predicted (fit quality + bias)

def plot_actual_vs_pred(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual satisfaction")
    plt.ylabel("Predicted satisfaction")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()

plot_actual_vs_pred(y_test, y_pred)

How to interpret
	•	Points should align around the diagonal.
	•	If predictions are compressed (e.g., never predicts 0 or 10), you’ll see “flattening”.
	•	Systematic shift up/down = bias.

⸻

B) Residual plot (detect patterns, nonlinearity, missing features)

Residual = Actual − Predicted

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted satisfaction")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    plt.show()

plot_residuals(y_test, y_pred)

How to interpret
	•	Residuals should be randomly scattered around 0.
	•	A curve pattern means model is missing structure (or needs different features).
	•	Increasing spread for high predictions = heteroscedasticity (errors grow in some ranges).

⸻

C) Error by target range (checks “imbalance” fix worked)

This is the most important for imbalanced regression.

def error_by_bins(y_true, y_pred, n_bins=6):
    df_err = pd.DataFrame({"y": y_true, "pred": y_pred})
    df_err["bin"] = pd.qcut(df_err["y"], q=n_bins, duplicates="drop")
    df_err["abs_err"] = (df_err["y"] - df_err["pred"]).abs()

    summary = df_err.groupby("bin").agg(
        count=("y", "size"),
        mae=("abs_err", "mean")
    ).reset_index()

    print(summary)

error_by_bins(y_test, y_pred, n_bins=6)

How to interpret
	•	If rare bins (like very low satisfaction) have huge MAE, imbalance is still hurting.
	•	Sample weights usually reduce those gaps.

⸻

7.3 Cross-validation (stronger evaluation than one split)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated predictions for analysis
y_cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

print(pd.Series(regression_report(y, y_cv_pred)).sort_index())

What it shows
	•	More realistic estimate of performance on unseen data.
	•	Less dependent on a single train/test split.

⸻

8) Feature importance (interpretation)

8.1 Permutation importance (recommended)

Works even after OneHot encoding.

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Extract feature names
feature_names = model.named_steps["prep"].get_feature_names_out()

imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
print(imp.head(30))

What it shows
	•	How much the metric worsens when you randomly shuffle a feature.
	•	If shuffling a feature destroys performance → it was important.

Caution
	•	Correlated features share importance; one may look less important because the other carries similar info.

⸻

9) Save and load the whole pipeline

joblib.dump(model, "satisfaction_rf_pipeline.joblib")

loaded = joblib.load("satisfaction_rf_pipeline.joblib")


⸻

10) Predict by typing the input yourself (dictionary → prediction)

This is exactly what you asked: you enter a dictionary, model predicts satisfaction 0..10.

# Example: YOU will replace keys/values with your real columns
sample = {
    "age": 29,
    "salary": 2500,
    "city": "Tunis",
    "subscription_type": "premium",
    "car_owner": "yes",
    "rides_last_month": 12
}

sample_df = pd.DataFrame([sample])

pred_value = loaded.predict(sample_df)[0]
print("Predicted satisfaction:", pred_value)

Important rules so dict prediction works
	1.	Your dict keys must match the original DataFrame column names (except target).
	2.	If a categorical value is unseen, OneHotEncoder with handle_unknown="ignore" will still work.
	3.	Missing columns will cause error. If you want, I can add a “schema guard” that fills missing columns automatically.

⸻

11) (Optional but recommended) keep predictions inside [0, 10]

Random forests can predict outside range slightly.

pred_value = np.clip(pred_value, 0, 10)

If your satisfaction is strictly integer 0..10, you can round:

pred_int = int(np.round(np.clip(pred_value, 0, 10)))


⸻

12) One complete script (end-to-end)

If you want everything in one runnable file, combine sections 2→10 exactly.
If you paste your real column names (or df.dtypes) I can output a final script that is 100% tailored to your dataset.

⸻

If you answer just these 2 details, I’ll tailor it perfectly (no back-and-forth)
	•	What is the exact target column name (satisfaction column)?
	•	Is satisfaction integer 0..10 or can it be float (like 7.3)?

Even without that, the code above is ready: replace "data.csv" + TARGET, and update the sample dictionary keys to match your columns.

```
