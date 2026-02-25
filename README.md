```
Perfect — let’s extend the weighted model with:
	1.	SHAP feature importance (global)
	2.	SHAP explanation for your exact new example row
	3.	Predict the value for that example
	4.	(Optional) compare baseline vs weighted prediction on that same row

Since you already have the example row and encoding flow, I’ll give you the blocks after weighted training (xgb_reg_weighted) and using your same variables.

⸻

1) SHAP for weighted XGBoost (global feature importance)

This shows which features influence the weighted model most overall.

# ==============================
# 8) SHAP - Global feature importance (weighted model)
# ==============================
# Install once if needed:
# !pip install shap

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure feature names are strings and safe for SHAP/XGBoost
X_train_shap = X_train_encoded.copy()
X_test_shap = X_test_encoded.copy()

X_train_shap.columns = X_train_shap.columns.astype(str)
X_test_shap.columns = X_test_shap.columns.astype(str)

# (Optional but recommended) use a sample for speed if dataset is large
# Change 1000 if you want more/less
shap_sample_size = min(1000, len(X_train_shap))
X_shap_sample = X_train_shap.sample(shap_sample_size, random_state=42)

# Build explainer
explainer_w = shap.TreeExplainer(xgb_reg_weighted)

# Compute SHAP values on sample
shap_values_w_sample = explainer_w.shap_values(X_shap_sample)

print("SHAP values shape (sample):", np.array(shap_values_w_sample).shape)
print("SHAP sample shape:", X_shap_sample.shape)

SHAP summary plot (global importance)

# ==============================
# 9) SHAP summary plot (global)
# ==============================
plt.figure()
shap.summary_plot(shap_values_w_sample, X_shap_sample, show=False)
plt.tight_layout()
plt.show()

SHAP bar plot (global top features)

# ==============================
# 10) SHAP bar plot (global top features)
# ==============================
plt.figure()
shap.summary_plot(shap_values_w_sample, X_shap_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

Global SHAP importance table (clean dataframe)

# ==============================
# 11) Global SHAP importance table (weighted model)
# ==============================
mean_abs_shap_w = np.abs(shap_values_w_sample).mean(axis=0)

shap_importance_weighted_df = pd.DataFrame({
    "feature": X_shap_sample.columns.astype(str),
    "mean_abs_shap": mean_abs_shap_w
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("Top 20 global SHAP features (weighted model):")
display(shap_importance_weighted_df.head(20))


⸻

2) Encode your exact example row (same encoding logic as training) and predict with weighted model

You asked to use the exact row from your image.
Below is a clean block that:
	•	builds X_new from your row
	•	applies the same transforms
	•	aligns columns with X_train_encoded
	•	predicts with xgb_reg_weighted

A) Create your example row exactly (from your image)

I copied your example as shown. Adjust only if one value differs in your notebook.

# ==============================
# 12) Your exact example row (from image)
# ==============================
import numpy as np
import pandas as pd

new_rows = [
    {
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
        "delai_Sinistre": 602000
    }
]

X_new = pd.DataFrame(new_rows)
print("X_new shape:", X_new.shape)
display(X_new)


⸻

B) Apply the same preprocessing/encoding to X_new

This block assumes you still have these variables from training:
	•	onehot_cols
	•	target_encode_cols
	•	freq_encode_cols
	•	force_categorical_cols
	•	encoding_artifacts
	•	optionally log_transform_cols (if you created it before)

It also handles your previous error (missing onehot_encoder) by using the stored OHE if available.
If OHE wasn’t stored, I also added a safe fallback using the stored created OHE columns.

# ==============================
# 13) Encode X_new exactly like training (no functions)
# ==============================

X_new_encoded_work = X_new.copy()

# ------------------------------
# A) Make sure expected columns exist (based on original df columns)
# ------------------------------
# If you still have df from training, use it as reference:
for c in df.columns:
    if c == target_col:
        continue
    if c not in X_new_encoded_work.columns:
        X_new_encoded_work[c] = np.nan

# Keep only training feature columns before encoding (same raw order if possible)
raw_feature_cols = [c for c in df.columns if c != target_col]
X_new_encoded_work = X_new_encoded_work[raw_feature_cols].copy()

# ------------------------------
# B) Force selected categorical cols to string
# ------------------------------
for c in force_categorical_cols:
    if c in X_new_encoded_work.columns:
        X_new_encoded_work[c] = X_new_encoded_work[c].astype("string")

# ------------------------------
# C) Convert numeric-like columns (except selected categorical columns)
# ------------------------------
categorical_selected = set(onehot_cols + target_encode_cols + freq_encode_cols + force_categorical_cols)

for c in X_new_encoded_work.columns:
    if c not in categorical_selected:
        X_new_encoded_work[c] = pd.to_numeric(X_new_encoded_work[c], errors="coerce")

# ------------------------------
# D) Apply same "0 -> np.nan" business rule if you used it on specific columns
#    (edit this list to match your training)
# ------------------------------
zero_to_nan_cols = [
    "Nombre_sisnitre_refuse_client",
    "Nombre_sisnitre_sans_suite_client",
    "delai_de_completude",
    "montant_indem"
]

for c in zero_to_nan_cols:
    if c in X_new_encoded_work.columns:
        X_new_encoded_work[c] = pd.to_numeric(X_new_encoded_work[c], errors="coerce")
        X_new_encoded_work[c] = X_new_encoded_work[c].replace(0, np.nan)

# ------------------------------
# E) Log transform on selected numeric columns (same as training, if used)
# ------------------------------
# If you used a list before, keep it. If not, this safely skips.
if "log_transform_cols" in globals():
    for c in log_transform_cols:
        if c in X_new_encoded_work.columns:
            X_new_encoded_work[c] = pd.to_numeric(X_new_encoded_work[c], errors="coerce")
            # log1p only for non-negative values
            X_new_encoded_work[c] = np.where(
                X_new_encoded_work[c].notna() & (X_new_encoded_work[c] >= 0),
                np.log1p(X_new_encoded_work[c]),
                X_new_encoded_work[c]
            )

# ------------------------------
# F) Frequency / Count Encoding (using train maps)
# ------------------------------
freq_maps = encoding_artifacts.get("frequency_encoding_maps", {})

for c in freq_encode_cols:
    if c in X_new_encoded_work.columns:
        X_new_encoded_work[c] = X_new_encoded_work[c].astype("string")
        freq_map_c = freq_maps.get(c, {})
        X_new_encoded_work[c] = X_new_encoded_work[c].map(freq_map_c)

        # Fill unseen categories with 0 (or np.nan if you prefer)
        X_new_encoded_work[c] = X_new_encoded_work[c].fillna(0)

# ------------------------------
# G) Target Encoding (using train maps + global mean)
# ------------------------------
target_maps = encoding_artifacts.get("target_encoding_maps", {})
global_target_mean = encoding_artifacts.get("target_encoding_global_mean", float(pd.to_numeric(y_train, errors="coerce").mean()))

for c in target_encode_cols:
    if c in X_new_encoded_work.columns:
        X_new_encoded_work[c] = X_new_encoded_work[c].astype("string")
        te_map_c = target_maps.get(c, {})
        X_new_encoded_work[c] = X_new_encoded_work[c].map(te_map_c).fillna(global_target_mean)

# ------------------------------
# H) One-Hot Encoding (use fitted encoder if stored)
# ------------------------------
# Try common keys (depending on how you stored it)
ohe = None
for k in ["onehot_encoder", "ohe", "fitted_onehot_encoder"]:
    if k in encoding_artifacts:
        ohe = encoding_artifacts[k]
        break

if len(onehot_cols) > 0:
    # Ensure columns exist and are strings
    for c in onehot_cols:
        if c not in X_new_encoded_work.columns:
            X_new_encoded_work[c] = pd.Series([pd.NA] * len(X_new_encoded_work), dtype="string")
        else:
            X_new_encoded_work[c] = X_new_encoded_work[c].astype("string")

    if ohe is not None:
        X_new_ohe_arr = ohe.transform(X_new_encoded_work[onehot_cols])

        if hasattr(X_new_ohe_arr, "toarray"):
            X_new_ohe_arr = X_new_ohe_arr.toarray()

        ohe_feature_names = ohe.get_feature_names_out(onehot_cols).tolist()

        X_new_ohe = pd.DataFrame(
            X_new_ohe_arr,
            columns=ohe_feature_names,
            index=X_new_encoded_work.index
        )

        X_new_encoded_work = X_new_encoded_work.drop(columns=[c for c in onehot_cols if c in X_new_encoded_work.columns])
        X_new_encoded_work = pd.concat([X_new_encoded_work, X_new_ohe], axis=1)

    else:
        # Fallback if encoder object was not saved:
        # Use the training-created OHE columns list and manually create 0/1 columns only for matching categories
        print("⚠️ onehot_encoder not found in encoding_artifacts. Using fallback alignment by final columns only.")
        print("   This works for prediction only if your row categories match existing OHE columns in X_train_encoded.")

# ------------------------------
# I) Final align to training encoded columns (MOST IMPORTANT)
# ------------------------------
# Add missing encoded columns as 0, drop extras, reorder exactly like train
for c in X_train_encoded.columns:
    if c not in X_new_encoded_work.columns:
        X_new_encoded_work[c] = 0

extra_cols = [c for c in X_new_encoded_work.columns if c not in X_train_encoded.columns]
if len(extra_cols) > 0:
    X_new_encoded_work = X_new_encoded_work.drop(columns=extra_cols)

X_new_encoded = X_new_encoded_work[X_train_encoded.columns].copy()

# Convert all to numeric float
for c in X_new_encoded.columns:
    X_new_encoded[c] = pd.to_numeric(X_new_encoded[c], errors="coerce")

X_new_encoded = X_new_encoded.astype(float)

# Clean feature names (same rule as training)
X_new_encoded.columns = (
    X_new_encoded.columns.astype(str)
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "_lt_", regex=False)
)

# Also ensure training/test columns are same cleaned names if not already done
X_train_encoded.columns = (
    X_train_encoded.columns.astype(str)
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "_lt_", regex=False)
)
X_test_encoded.columns = (
    X_test_encoded.columns.astype(str)
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "_lt_", regex=False)
)

# Re-align again after cleaning names
X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print("X_new_encoded.shape:", X_new_encoded.shape)
print("Expected shape      :", (len(X_new_encoded), X_train_encoded.shape[1]))
display(X_new_encoded.head())


⸻

3) Predict your example row with the weighted model

# ==============================
# 14) Predict your example row (weighted model)
# ==============================
pred_new_weighted = xgb_reg_weighted.predict(X_new_encoded)

# Optional clip to business range
pred_new_weighted_clipped = np.clip(pred_new_weighted, 0, 10)

print("Weighted model prediction (raw):", pred_new_weighted[0])
print("Weighted model prediction (clipped 0..10):", pred_new_weighted_clipped[0])

Optional: compare with baseline model on the same row

# ==============================
# 15) Compare baseline vs weighted prediction on same row (optional)
# ==============================
if "baseline_xgb_model" in globals():
    pred_new_baseline = baseline_xgb_model.predict(X_new_encoded)
    pred_new_baseline_clipped = np.clip(pred_new_baseline, 0, 10)

    print("Baseline prediction (clipped):", pred_new_baseline_clipped[0])
    print("Weighted prediction (clipped):", pred_new_weighted_clipped[0])
else:
    print("baseline_xgb_model not found (skip baseline comparison)")


⸻

4) SHAP explanation for this specific example row (why this prediction)

This is exactly what you asked: show which features pushed the prediction up/down for that one row.

# ==============================
# 16) SHAP for this specific example row (weighted model)
# ==============================
# Use same explainer if already created; otherwise create it
if "explainer_w" not in globals():
    import shap
    explainer_w = shap.TreeExplainer(xgb_reg_weighted)

# SHAP values for the single row
shap_values_new_w = explainer_w.shap_values(X_new_encoded)

print("SHAP values shape for new row:", np.array(shap_values_new_w).shape)

Contribution table (best for debugging and understanding)

# ==============================
# 17) SHAP contribution table for the new row
# ==============================
# Base value (expected model output)
base_value_w = explainer_w.expected_value
if isinstance(base_value_w, (list, np.ndarray)):
    base_value_w = np.array(base_value_w).reshape(-1)[0]

# Single row arrays
row_values = X_new_encoded.iloc[0]
row_shap_values = np.array(shap_values_new_w)[0]

shap_row_contrib_df = pd.DataFrame({
    "feature": X_new_encoded.columns.astype(str),
    "feature_value": row_values.values,
    "shap_value": row_shap_values,
    "abs_shap_value": np.abs(row_shap_values)
}).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

# Add direction label
shap_row_contrib_df["effect_on_prediction"] = np.where(
    shap_row_contrib_df["shap_value"] > 0, "pushes UP",
    np.where(shap_row_contrib_df["shap_value"] < 0, "pushes DOWN", "neutral")
)

print("Top features that influenced THIS prediction:")
display(shap_row_contrib_df.head(25))

# Sanity check: base + sum(shap) ~= prediction (raw)
pred_check = float(pred_new_weighted[0])
reconstructed_pred = float(base_value_w + row_shap_values.sum())

print("Base value (expected):", base_value_w)
print("Sum SHAP values      :", row_shap_values.sum())
print("Reconstructed pred   :", reconstructed_pred)
print("Model raw prediction :", pred_check)
print("Difference           :", abs(reconstructed_pred - pred_check))

SHAP waterfall plot (single prediction explanation)

# ==============================
# 18) SHAP waterfall plot for this example row
# ==============================
import shap
import matplotlib.pyplot as plt

# Build explanation object
shap_exp_new = shap.Explanation(
    values=row_shap_values,
    base_values=base_value_w,
    data=X_new_encoded.iloc[0].values,
    feature_names=X_new_encoded.columns.tolist()
)

plt.figure()
shap.plots.waterfall(shap_exp_new, max_display=20, show=False)
plt.tight_layout()
plt.show()

SHAP force plot (optional, interactive in notebook)

# ==============================
# 19) SHAP force plot (optional)
# ==============================
shap.initjs()
shap.force_plot(
    base_value_w,
    row_shap_values,
    X_new_encoded.iloc[0],
    matplotlib=True
)


⸻

5) Small explanation of what you’ll see (quick)
	•	Global SHAP (summary/bar): which features are important across many rows
	•	Row SHAP (waterfall/table): why this specific row got this prediction
	•	positive SHAP → pushes prediction up
	•	negative SHAP → pushes prediction down

⸻

Important fix for your earlier onehot_encoder error

You got:

Missing encoding_artifacts['onehot_encoder']

That means during training you saved:
	•	maps (frequency_encoding_maps, target_encoding_maps)
	•	but not the fitted OneHotEncoder object

Save it during training (must do once)

When you fit OHE in training, add this line:

encoding_artifacts["onehot_encoder"] = ohe

Also keep:

encoding_artifacts["onehot_columns_created"] = X_train_ohe.columns.tolist()  # if you already create this

If you want, I can give you the exact training encoding block patch (1–2 lines in the right place) so prediction on new rows always works without errors.

⸻

Bonus: quick “prediction summary” print for your example

# ==============================
# 20) Clean summary for your example row
# ==============================
print("=== New Example Prediction Summary ===")
print("Weighted model predicted evaluate_note (raw):", float(pred_new_weighted[0]))
print("Weighted model predicted evaluate_note (clipped 0..10):", float(pred_new_weighted_clipped[0]))

print("\nTop 10 drivers of this prediction (absolute SHAP):")
display(shap_row_contrib_df[["feature", "feature_value", "shap_value", "effect_on_prediction"]].head(10))


⸻

If you want next, I can give you the same SHAP + prediction block for the baseline model too, so you compare:
	•	baseline prediction + explanation
	•	weighted prediction + explanation

```
