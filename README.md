```
Yes — the error is clear from your screenshot ✅

Why it failed

You are doing:

ohe = encoding_artifacts.get("onehot_encoder", None)

But in your encoding_artifacts, the fitted encoder object was not saved under "onehot_encoder" (or not saved at all).

So we’ll fix it in a robust way:
	•	Try multiple possible keys for the fitted OHE object
	•	If not found, use a safe fallback with pd.get_dummies(...) + align to X_train_encoded.columns
	•	Then continue with:
	•	X_new_encoded
	•	prediction
	•	SHAP local explanation for this exact row

⸻

✅ Replace your broken OHE block with this (fixed) + continue after it

# ============================================================
# One-Hot Encoding (same fitted OHE from training) - FIXED
# Robust version: tries to find saved OHE object, otherwise fallback
# ============================================================

if len(onehot_cols) > 0:
    # Make sure all onehot columns exist in X_new
    for c in onehot_cols:
        if c not in X_new.columns:
            X_new[c] = pd.Series([pd.NA] * len(X_new), dtype="string")
        else:
            X_new[c] = X_new[c].astype("string")

    # Try multiple possible keys (depending on what you saved before)
    ohe = None
    for possible_key in [
        "onehot_encoder",          # what we expected
        "ohe",                     # common name
        "one_hot_encoder",         # possible variant
        "fitted_onehot_encoder"    # possible variant
    ]:
        if possible_key in encoding_artifacts and encoding_artifacts[possible_key] is not None:
            ohe = encoding_artifacts[possible_key]
            print(f"Using fitted OHE from encoding_artifacts['{possible_key}'] ✅")
            break

    # -----------------------------
    # CASE A: fitted OHE exists
    # -----------------------------
    if ohe is not None:
        X_new_ohe_arr = ohe.transform(X_new[onehot_cols])

        if hasattr(X_new_ohe_arr, "toarray"):
            X_new_ohe_arr = X_new_ohe_arr.toarray()

        ohe_feature_names = ohe.get_feature_names_out(onehot_cols).tolist()

        X_new_ohe = pd.DataFrame(
            X_new_ohe_arr,
            columns=ohe_feature_names,
            index=X_new.index
        )

        # Drop original onehot cols, append encoded cols
        X_new = X_new.drop(columns=[c for c in onehot_cols if c in X_new.columns])
        X_new = pd.concat([X_new, X_new_ohe], axis=1)

    # -----------------------------
    # CASE B: fitted OHE NOT saved -> fallback with pd.get_dummies
    # (works if we align later to X_train_encoded columns)
    # -----------------------------
    else:
        print("⚠️ Fitted OneHotEncoder not found in encoding_artifacts.")
        print("⚠️ Using fallback: pd.get_dummies on new row + align to training columns.")

        X_new = pd.get_dummies(
            X_new,
            columns=[c for c in onehot_cols if c in X_new.columns],
            dummy_na=False
        )

        # IMPORTANT:
        # We will align to X_train_encoded columns later, so missing OHE columns become 0.
        # Unknown new categories (not seen in train) will not create matching train columns, so they effectively become all-zeros.


⸻

✅ Add this code AFTER the OHE block (encode finish → align → predict → SHAP)

# ============================================================
# Continue encoding after OHE
# ============================================================

# 1) Optional log transforms (ONLY if you used them during training)
# Put the EXACT same columns you transformed during training
log_transform_cols = [
    # "delai_declaration",
    # "delai_decision",
    # "delai_Sinistre",
    # "ancienneté_de_contrat",
    # "montant_indem"
]

for c in log_transform_cols:
    if c in X_new.columns:
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
        X_new.loc[X_new[c] <= -1, c] = np.nan
        X_new[c] = np.log1p(X_new[c])


# 2) Clean feature names (XGBoost cannot accept some special chars)
X_new.columns = (
    X_new.columns.astype(str)
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "lt_", regex=False)
    .str.replace(">", "gt_", regex=False)
)

# 3) Convert to numeric
for c in X_new.columns:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
X_new = X_new.astype(float)

print("X_new shape BEFORE align:", X_new.shape)


# ============================================================
# Align EXACTLY with training columns (VERY IMPORTANT)
# ============================================================
# This is what makes the fallback OHE strategy work too.
X_new_encoded = X_new.reindex(columns=X_train_encoded.columns, fill_value=0).copy()

# ensure types are correct
X_new_encoded.columns = X_new_encoded.columns.astype(str)
X_new_encoded = X_new_encoded.astype(float)

print("X_new_encoded.shape:", X_new_encoded.shape)
print("X_train_encoded.shape:", X_train_encoded.shape)
print("Same number of features? ->", X_new_encoded.shape[1] == X_train_encoded.shape[1])

display(X_new_encoded.head(1))


# ============================================================
# Predict with your trained model
# ============================================================
pred_new = xgb_reg.predict(X_new_encoded)
pred_new_clipped = np.clip(pred_new, 0, 10)

print("\n✅ Predicted value (raw):", float(pred_new[0]))
print("✅ Predicted value (clipped 0..10):", float(pred_new_clipped[0]))


# ============================================================
# SHAP local explanation for THIS row (why this prediction?)
# ============================================================
import shap
import matplotlib.pyplot as plt

# TreeExplainer for XGBoost tree model
explainer = shap.TreeExplainer(xgb_reg)

# shap values for single row
shap_values_single = explainer.shap_values(X_new_encoded)

# base value
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = float(np.array(base_value).reshape(-1)[0])
else:
    base_value = float(base_value)

# Build local feature-importance table for this row
local_shap_df = pd.DataFrame({
    "feature": X_new_encoded.columns.astype(str),
    "feature_value": X_new_encoded.iloc[0].values,
    "shap_value": shap_values_single[0]
})

local_shap_df["abs_shap"] = local_shap_df["shap_value"].abs()
local_shap_df = local_shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

print("\n=== Top features influencing THIS prediction (local SHAP) ===")
print("Base value (average prediction):", base_value)
print("Model prediction (raw):", float(pred_new[0]))
print("Model prediction (clipped):", float(pred_new_clipped[0]))
display(local_shap_df.head(20))


# ============================================================
# SHAP waterfall plot (best for one row)
# ============================================================
try:
    shap_explanation = shap.Explanation(
        values=shap_values_single[0],
        base_values=base_value,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )
    shap.plots.waterfall(shap_explanation, max_display=20)
    plt.show()
except Exception as e:
    print("Waterfall plot skipped:", e)


# ============================================================
# Optional: show top positive and top negative contributions
# ============================================================
top_positive = local_shap_df.sort_values("shap_value", ascending=False).head(10)
top_negative = local_shap_df.sort_values("shap_value", ascending=True).head(10)

print("\nTop positive contributors (push prediction UP):")
display(top_positive[["feature", "feature_value", "shap_value"]])

print("\nTop negative contributors (push prediction DOWN):")
display(top_negative[["feature", "feature_value", "shap_value"]])


⸻

Important note (so you understand exactly)

If fitted OHE was not saved

The fallback (pd.get_dummies) can still work because we align to X_train_encoded.columns.

That means:
	•	known one-hot columns from training are matched ✅
	•	missing ones are filled with 0 ✅
	•	unseen categories in new row become “no known category activated” (all zeros for that feature family) ✅

This is acceptable for prediction, but best practice is still to save the fitted OneHotEncoder.

⸻

(Optional but recommended) Save the encoder during training next time

In your training encoding block, after fitting OHE, add:

encoding_artifacts["onehot_encoder"] = ohe

Then future new-row encoding will be exact and easier.

⸻

If you want, next I can give you one more small block to print a human-readable explanation like:
	•	“Prediction increased mostly because delai_Sinistre, tarif, …”
	•	“Prediction decreased because garantie__freq, decision_ai, …”


```
