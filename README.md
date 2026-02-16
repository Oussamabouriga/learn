```
Below I’m adding more regression evaluation metrics for your XGBoost Regressor (0→10 satisfaction), with:
	•	code (no helper functions) to compute everything
	•	for each metric: what it measures, how to interpret it, when to prefer it, common pitfalls
	•	best practices that make results trustworthy (no leakage, good splits, robust validation, etc.)

I’ll assume you already have:
	•	model (your Pipeline(preprocessor + XGBRegressor))
	•	X_test, y_test
	•	and you computed y_pred = np.clip(model.predict(X_test), 0, 10)

⸻

1) A richer set of regression metrics (code)

Imports

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error
)

Compute metrics

# Ensure numpy arrays
y_true = np.asarray(y_test)
y_hat  = np.asarray(y_pred)

# Basic errors
err  = y_true - y_hat
ae   = np.abs(err)
se   = err**2

# Core metrics
MAE   = mean_absolute_error(y_true, y_hat)
MSE   = mean_squared_error(y_true, y_hat)
RMSE  = np.sqrt(MSE)
R2    = r2_score(y_true, y_hat)
EVS   = explained_variance_score(y_true, y_hat)
MedAE = median_absolute_error(y_true, y_hat)
MaxE  = max_error(y_true, y_hat)

# Percentage metrics (careful near zero)
MAPE  = mean_absolute_percentage_error(y_true, y_hat)

# sMAPE (safer than MAPE when y can be small)
eps = 1e-9
sMAPE = np.mean(2.0 * ae / (np.abs(y_true) + np.abs(y_hat) + eps))

# Normalized errors (scale-free comparisons)
y_range = (y_true.max() - y_true.min()) + eps
NRMSE_range = RMSE / y_range
NMAE_range  = MAE  / y_range

y_mean = np.mean(y_true) + eps
NRMSE_mean = RMSE / y_mean
NMAE_mean  = MAE  / y_mean

# Bias (systematic over/under prediction)
Bias = np.mean(err)  # >0 means you under-predict on average (actual - pred positive)

# Correlations (ranking/association)
pearson_corr = np.corrcoef(y_true, y_hat)[0, 1]

# Spearman correlation without scipy (rank correlation)
y_true_rank = pd.Series(y_true).rank(method="average").to_numpy()
y_hat_rank  = pd.Series(y_hat).rank(method="average").to_numpy()
spearman_corr = np.corrcoef(y_true_rank, y_hat_rank)[0, 1]

# Adjusted R2 (penalizes too many features) - needs number of features after preprocessing
# Use encoded feature count:
n = len(y_true)
p = len(model.named_steps["prep"].get_feature_names_out())
Adj_R2 = 1 - (1 - R2) * (n - 1) / max(n - p - 1, 1)

# Put everything in a table
metrics = pd.Series({
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "R2": R2,
    "Adj_R2": Adj_R2,
    "ExplainedVariance": EVS,
    "MedianAE": MedAE,
    "MaxError": MaxE,
    "MAPE": MAPE,
    "sMAPE": sMAPE,
    "Bias(mean(y - pred))": Bias,
    "PearsonCorr": pearson_corr,
    "SpearmanCorr": spearman_corr,
    "NRMSE_range": NRMSE_range,
    "NMAE_range": NMAE_range,
    "NRMSE_mean": NRMSE_mean,
    "NMAE_mean": NMAE_mean,
}).sort_index()

print(metrics)


⸻

2) Explanation: what each metric tells you (and how to interpret it)

A) Error magnitude metrics (most important)

MAE (Mean Absolute Error)
	•	What it measures: average absolute difference |y - ŷ|
	•	Interpretation: “On average, I’m off by MAE points of satisfaction.”
	•	Best when: you want a human-readable error in the same unit (0..10 scale).
	•	Pitfall: MAE treats all errors equally (a 2-point miss is exactly twice a 1-point miss).

MSE (Mean Squared Error)
	•	What it measures: average squared error (y - ŷ)^2
	•	Interpretation: punishes large mistakes more than small ones.
	•	Best when: you care a lot about big errors.
	•	Pitfall: harder to interpret because it’s in “squared units”.

RMSE
	•	What it measures: sqrt(MSE) (back to original unit)
	•	Interpretation: “typical” size of error, but more sensitive to big misses than MAE.
	•	Best when: large errors are costly (bad user experience, critical scores).
	•	Diagnostic hint: if RMSE ≫ MAE, you likely have some big outliers/misses.

MedianAE
	•	What it measures: median absolute error (robust)
	•	Interpretation: “For a typical example, I’m off by about MedianAE.”
	•	Best when: data has outliers or some extreme bad cases that you don’t want to dominate the evaluation.

MaxError
	•	What it measures: largest absolute error on test set.
	•	Interpretation: worst-case miss.
	•	Best when: you need guarantees on worst-case behavior.
	•	Pitfall: very sensitive to a single outlier/test anomaly.

⸻

B) “Goodness of fit” metrics (overall explanatory power)

R² (Coefficient of determination)
	•	What it measures: improvement over predicting the mean; proportion of variance explained.
	•	Interpretation:
	•	R² = 1 perfect
	•	R² = 0 same as always predicting mean
	•	R² < 0 worse than predicting mean
	•	Best when: comparing models on same dataset/split.
	•	Pitfall: can look OK while still being bad on rare target ranges (so always pair with bin-wise errors).

Adjusted R²
	•	What it measures: R² penalized for too many features.
	•	Interpretation: helps detect “feature explosion” (like huge one-hot space) that doesn’t truly help.
	•	Pitfall: with one-hot, p (features) can be large; Adjusted R² can drop even when performance is decent.

Explained Variance (EVS)
	•	What it measures: how well predictions explain variance of the target (similar to R² but slightly different in how it accounts for bias).
	•	Interpretation: if EVS is high but R² lower, you may have systematic bias/offset issues.

⸻

C) Percentage metrics (use carefully for 0..10)

MAPE
	•	What it measures: average absolute percentage error.
	•	Interpretation: “average % error”
	•	Pitfall: If your true values include 0 or near 0, MAPE can explode or become misleading.

sMAPE
	•	What it measures: symmetric version using (abs(y)+abs(ŷ)) in denominator.
	•	Best when: your target can be small or include zeros.
	•	Still a pitfall: percentage metrics are less intuitive when your scale is small (0..10). Often MAE is better.

⸻

D) Bias and correlation (model behavior insights)

Bias = mean(y - ŷ)
	•	Interpretation:
	•	positive bias ⇒ model under-predicts on average
	•	negative bias ⇒ model over-predicts on average
	•	Best practice: bias close to 0 is generally good, but you can accept some bias if it reduces MAE/RMSE.

Pearson correlation
	•	What it measures: linear association between y and ŷ.
	•	Interpretation: higher = predictions track the target changes well (linearly).
	•	Pitfall: you can have high correlation but still be systematically off (bias).

Spearman correlation
	•	What it measures: ranking agreement between y and ŷ.
	•	Interpretation: good if you mainly want correct ordering (who is more satisfied), even if absolute values are slightly off.

⸻

E) Normalized errors (compare across datasets)

NRMSE_range / NMAE_range
	•	divides error by (max(y)-min(y))
	•	useful to compare across targets with different ranges.

NRMSE_mean / NMAE_mean
	•	divides error by mean(y)
	•	can be unstable when mean is small.

⸻

3) Extra evaluation you SHOULD do for satisfaction 0..10

A) Error by target ranges (checks imbalance handling)

This is one of the most important checks for “imbalanced regression”.

df_eval = pd.DataFrame({"y": y_true, "pred": y_hat})
df_eval["bin"] = pd.qcut(df_eval["y"], q=6, duplicates="drop")
df_eval["abs_err"] = np.abs(df_eval["y"] - df_eval["pred"])

bin_report = df_eval.groupby("bin").agg(
    count=("y", "size"),
    MAE=("abs_err", "mean"),
    MedianAE=("abs_err", "median"),
).reset_index()

print(bin_report)

How to interpret
	•	If bins near 0 or 10 have much larger MAE → model still ignores rare extremes.
	•	Your sample weights / binning strategy should reduce this.

⸻

B) Residual diagnostics (pattern detection)

import matplotlib.pyplot as plt

residuals = y_true - y_hat

plt.figure()
plt.scatter(y_hat, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()

Interpretation
	•	random cloud around 0 = good
	•	curve/pattern = missing features or wrong model capacity
	•	more spread at some predictions = unstable region

⸻

C) Error distribution (are most errors small?)

plt.figure()
plt.hist(np.abs(residuals), bins=30)
plt.xlabel("|Error|")
plt.ylabel("Count")
plt.title("Distribution of Absolute Errors")
plt.grid(True)
plt.show()

Interpretation
	•	If many errors near 0 with a long tail → MedianAE will look good but RMSE may be high.

⸻

D) “Within tolerance” accuracy (very useful for 0..10)

This is often the most business-friendly metric:
	•	% predictions within ±0.5
	•	within ±1.0
	•	within ±2.0

within_05 = np.mean(np.abs(y_true - y_hat) <= 0.5)
within_10 = np.mean(np.abs(y_true - y_hat) <= 1.0)
within_20 = np.mean(np.abs(y_true - y_hat) <= 2.0)

print("Within ±0.5:", within_05)
print("Within ±1.0:", within_10)
print("Within ±2.0:", within_20)

Interpretation
	•	Great for stakeholders: “70% of predictions are within 1 satisfaction point.”

⸻

4) Best practices (XGBoost regression 0..10) — what to do and why

1) Always use a Pipeline

✅ prevents leakage
✅ ensures dictionary prediction works
✅ ensures one-hot columns stay consistent

2) Use 3-way split for XGBoost: train / valid / test
	•	train: fit model
	•	valid: early stopping
	•	test: final unbiased evaluation

Early stopping is one of the biggest best practices for XGBoost.

3) Prefer MAE + RMSE + bin-wise MAE

For satisfaction 0..10:
	•	MAE = easiest to interpret
	•	RMSE = punishes big mistakes
	•	bin-wise MAE = checks rare targets

4) Always check imbalance impact

Do:
	•	stratified split using qcut bins
	•	sample weights based on bin frequency
	•	evaluate bin-wise MAE

5) Clip predictions to [0,10]

It’s reasonable for satisfaction:

y_pred = np.clip(y_pred, 0, 10)

If labels are integers, you may round at the end:

y_pred_int = np.round(y_pred).astype(int)

(Keep evaluation on float predictions too; rounding can hide issues.)

6) Start simple, then tune gradually

Start baseline:
	•	max_depth=4..8
	•	min_child_weight=1..10
	•	subsample=0.7..1.0
	•	colsample_bytree=0.7..1.0
	•	learning_rate=0.01..0.1
	•	large n_estimators + early stopping

7) Prefer robust validation
	•	If data has time order: split by time (don’t shuffle)
	•	Otherwise: KFold CV (but note early stopping complicates CV; you can still do it with fixed n_estimators or custom CV loops)

8) Use feature importance carefully
	•	One-hot creates many dummy columns; interpret importance at:
	•	dummy level (fine)
	•	or group by original feature (sum importance of all one-hot columns for one feature)

9) Set random seeds for reproducibility
	•	random_state=42
	•	fixed split

10) Keep a baseline model

Always compare to:
	•	predict mean
	•	or simple linear regression
If XGBoost barely beats baseline, features may be weak or noisy.

⸻

If you want, paste:
	•	df.shape
	•	df.dtypes
	•	and whether you want “good on extremes” (0/10) more than middle

…and I’ll recommend a strong XGBoost configuration + the best evaluation set for your exact dataset (including bin count and weighting strength).


```
