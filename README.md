```

Perfect — now that the baseline XGBoost training works, let’s add exactly the two things you asked for:
	1.	SHAP (feature importance + explanation)
	2.	Predict your new_rows example with the trained model (xgb_reg)

I’ll keep it clean, notebook-style, no functions ✅

⸻

12) SHAP for feature importance (global + local)

12.1 Install/import SHAP (if needed)

# If SHAP is not installed, run once:
# !pip install shap

# ==============================
# 12) Imports for SHAP
# ==============================
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


⸻

12.2 Build SHAP explainer on your trained baseline model

# ==============================
# 12.2) Build SHAP explainer
# ==============================
# xgb_reg is your trained baseline model
explainer = shap.TreeExplainer(xgb_reg)

print("SHAP explainer created ✅")


⸻

12.3 SHAP values on a sample of train data (faster + enough for global importance)

# ==============================
# 12.3) Compute SHAP values on train sample
# ==============================
sample_n = 500 if len(X_train_encoded) > 500 else len(X_train_encoded)

X_shap_sample = X_train_encoded.sample(sample_n, random_state=42).copy()

# Make sure columns are strings (safe)
X_shap_sample.columns = X_shap_sample.columns.astype(str)

shap_values_sample = explainer.shap_values(X_shap_sample)

print("X_shap_sample shape:", X_shap_sample.shape)
print("SHAP values shape:", np.array(shap_values_sample).shape)
print("SHAP values computed ✅")


⸻

12.4 SHAP summary plot (best global feature importance view)

# ==============================
# 12.4) SHAP summary plot (global importance)
# ==============================
shap.summary_plot(shap_values_sample, X_shap_sample, show=True)

How to read it (quick)
	•	Top rows = most important features globally
	•	Right (positive SHAP) = pushes prediction higher
	•	Left (negative SHAP) = pushes prediction lower
	•	Red = high feature value, Blue = low feature value

⸻

12.5 SHAP bar plot (simple ranked feature importance)

# ==============================
# 12.5) SHAP bar plot (mean |SHAP|)
# ==============================
shap.summary_plot(shap_values_sample, X_shap_sample, plot_type="bar", show=True)

This gives a simple ranking like feature importance, but based on SHAP (usually better than raw tree importance).

⸻

12.6 SHAP feature importance table (top features in dataframe)

# ==============================
# 12.6) SHAP importance table (dataframe)
# ==============================
shap_importance_df = pd.DataFrame({
    "feature": X_shap_sample.columns.astype(str),
    "mean_abs_shap": np.abs(shap_values_sample).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("Top 20 features by SHAP importance:")
display(shap_importance_df.head(20))


⸻

13) Predict your new_rows example with the trained model

Since you already said your new_rows block works, I’ll give you the prediction block + SHAP explanation for that example.

This assumes you already transformed it into X_new_encoded using the same encoding steps and aligned columns to X_train_encoded.columns.

⸻

13.1 Final checks + predict new example

# ==============================
# 13.1) Predict on new example row(s)
# ==============================
print("X_new_encoded shape:", X_new_encoded.shape)
print("Expected train shape (#features):", X_train_encoded.shape[1])

# Ensure exact same columns/order as train (important)
X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).copy()

# Predict
new_pred = xgb_reg.predict(X_new_encoded)

# Optional: clip to business range (0..10) if your target is evaluate_note
new_pred_clipped = np.clip(new_pred, 0, 10)

print("Raw prediction(s):", new_pred)
print("Clipped prediction(s) [0,10]:", new_pred_clipped)


⸻

13.2 Show prediction nicely beside your input row

# ==============================
# 13.2) Display input + prediction together
# ==============================
new_rows_preview = pd.DataFrame(new_rows).copy()

new_rows_preview["predicted_evaluate_note_raw"] = new_pred
new_rows_preview["predicted_evaluate_note_clipped_0_10"] = new_pred_clipped

display(new_rows_preview)


⸻

14) SHAP explanation for your new example (why this prediction?)

This explains which features pushed the prediction up/down for that specific row.

14.1 Compute SHAP values for X_new_encoded

# ==============================
# 14.1) SHAP for the new example row(s)
# ==============================
shap_values_new = explainer.shap_values(X_new_encoded)

print("Prediction (raw):", float(new_pred[0]))
print("Prediction (clipped):", float(new_pred_clipped[0]))
print("SHAP values for new row computed ✅")


⸻

14.2 SHAP contributions table for the first new row (best guaranteed output)

# ==============================
# 14.2) SHAP contribution table for first new row
# ==============================
shap_new_df = pd.DataFrame({
    "feature": X_new_encoded.columns.astype(str),
    "feature_value": X_new_encoded.iloc[0].values,
    "shap_value": shap_values_new[0]
})

shap_new_df["abs_shap"] = shap_new_df["shap_value"].abs()

# Sort by strongest effect
shap_new_df = shap_new_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

print("Top 20 feature contributions for this prediction:")
display(shap_new_df.head(20))

How to read this table
	•	shap_value > 0 → feature pushes prediction up
	•	shap_value < 0 → feature pushes prediction down
	•	larger abs_shap → stronger impact

⸻

14.3 Waterfall plot for your example (visual explanation)

# ==============================
# 14.3) SHAP waterfall plot (single row explanation)
# ==============================
# Some SHAP versions differ; this block tries the modern plot first
try:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    single_explanation = shap.Explanation(
        values=shap_values_new[0],
        base_values=base_value,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.astype(str).tolist()
    )

    shap.plots.waterfall(single_explanation, max_display=20)

except Exception as e:
    print("Waterfall plot not available in this SHAP version:", e)
    print("Use the contribution table above (14.2), it gives the same logic.")


⸻

15) (Optional) Compare XGBoost built-in importance vs SHAP importance

This helps you see the difference between model split importance and actual prediction contribution importance.

# ==============================
# 15) Compare built-in importance vs SHAP importance
# ==============================
xgb_importance_df = pd.DataFrame({
    "feature": X_train_encoded.columns.astype(str),
    "xgb_importance": xgb_reg.feature_importances_
}).sort_values("xgb_importance", ascending=False)

compare_importance_df = (
    xgb_importance_df.merge(shap_importance_df, on="feature", how="outer")
    .fillna(0)
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

print("Top features by SHAP (with XGBoost built-in importance side by side):")
display(compare_importance_df.head(20))


⸻

16) Quick notes (important for your workflow)
	•	XGBoost feature_importances_ = how often/usefully features split trees
	•	SHAP = how much features actually contribute to predictions
➡️ For explanation, SHAP is better
	•	For predicting new_rows, always keep:
	•	same encodings
	•	same one-hot fitted encoder (ohe)
	•	same frequency/target maps
	•	same column cleaning
	•	same column order as X_train_encoded

⸻

If you want next, I can give you the weighted baseline (imbalanced regression target) block in the same style (no functions), then a small Random Search block (fast and practical).
```
