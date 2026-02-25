```
# ============================================================
# 0) Imports (prediction + local explanation with SHAP)
# ============================================================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# (optional, nicer display in notebooks)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# 1) Your exact new row (copied from your image)
#    Keep values exactly as you want the model to see them
# ============================================================
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
        "model": "Pixel 7 Pro",
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

X_new = pd.DataFrame(new_rows).copy()

print("Raw X_new shape:", X_new.shape)
display(X_new)


# ============================================================
# 2) Apply the SAME preprocessing logic as training
#    (zero -> np.nan only for selected business columns)
#    Edit this list to match what you used during training
# ============================================================
zero_to_nan_cols = [
    "Nombre_sisnitre_refuse_client",
    "Nombre_sisnitre_sans_suite_client",
    "delai_de_completude",
    "montant_indem"
]

for c in zero_to_nan_cols:
    if c in X_new.columns:
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
        X_new.loc[X_new[c] == 0, c] = np.nan


# ============================================================
# 3) Force categorical columns (same as training)
# ============================================================
if "force_categorical_cols" in globals():
    for c in force_categorical_cols:
        if c in X_new.columns:
            X_new[c] = X_new[c].astype("string")


# ============================================================
# 4) Frequency / Count Encoding (same maps from TRAIN only)
#    Requires: encoding_artifacts["frequency_encoding_maps"]
# ============================================================
freq_maps = encoding_artifacts.get("frequency_encoding_maps", {})

for c in freq_encode_cols:
    if c in X_new.columns:
        X_new[c] = X_new[c].astype("string")
        X_new[f"{c}__freq"] = X_new[c].map(freq_maps.get(c, {})).fillna(0).astype(float)

# Drop original frequency-encoded columns
freq_drop_cols = [c for c in freq_encode_cols if c in X_new.columns]
if len(freq_drop_cols) > 0:
    X_new = X_new.drop(columns=freq_drop_cols)


# ============================================================
# 5) Target Encoding (same maps from TRAIN only)
#    Requires: encoding_artifacts["target_encoding_maps"]
# ============================================================
te_maps = encoding_artifacts.get("target_encoding_maps", {})
global_te_mean = encoding_artifacts.get("target_encoding_global_mean", np.nan)

for c in target_encode_cols:
    if c in X_new.columns:
        X_new[c] = X_new[c].astype("string")
        X_new[f"{c}__te"] = X_new[c].map(te_maps.get(c, {})).fillna(global_te_mean).astype(float)

# Drop original target-encoded columns
te_drop_cols = [c for c in target_encode_cols if c in X_new.columns]
if len(te_drop_cols) > 0:
    X_new = X_new.drop(columns=te_drop_cols)


# ============================================================
# 6) One-Hot Encoding (same fitted OHE from training)
#    Requires: encoding_artifacts["onehot_encoder"]
# ============================================================
if len(onehot_cols) > 0:
    # Make sure all onehot columns exist
    for c in onehot_cols:
        if c not in X_new.columns:
            X_new[c] = pd.Series([pd.NA] * len(X_new), dtype="string")
        else:
            X_new[c] = X_new[c].astype("string")

    ohe = encoding_artifacts.get("onehot_encoder", None)
    if ohe is None:
        raise ValueError("Missing encoding_artifacts['onehot_encoder']. Save the fitted OneHotEncoder from training first.")

    X_new_ohe_arr = ohe.transform(X_new[onehot_cols])

    if hasattr(X_new_ohe_arr, "toarray"):
        X_new_ohe_arr = X_new_ohe_arr.toarray()

    ohe_feature_names = ohe.get_feature_names_out(onehot_cols).tolist()

    X_new_ohe = pd.DataFrame(
        X_new_ohe_arr,
        columns=ohe_feature_names,
        index=X_new.index
    )

    X_new = X_new.drop(columns=[c for c in onehot_cols if c in X_new.columns])
    X_new = pd.concat([X_new, X_new_ohe], axis=1)


# ============================================================
# 7) Apply SAME numeric transforms as training (if used)
#    IMPORTANT: Put the exact list you used before
# ============================================================
# Example: if you used log1p on some columns during training, list them here.
# If you did NOT use log transforms, keep this list empty.
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


# ============================================================
# 8) Clean feature names (XGBoost requirement)
#    Avoid chars like [, ], <
# ============================================================
X_new.columns = (
    X_new.columns.astype(str)
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "lt_", regex=False)
    .str.replace(">", "gt_", regex=False)
)

# Convert all to numeric float
for c in X_new.columns:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
X_new = X_new.astype(float)


# ============================================================
# 9) Align EXACTLY to training matrix columns (same order)
# ============================================================
X_new_encoded = X_new.reindex(columns=X_train_encoded.columns, fill_value=0).copy()
X_new_encoded.columns = X_new_encoded.columns.astype(str)
X_new_encoded = X_new_encoded.astype(float)

print("X_new_encoded.shape:", X_new_encoded.shape)
print("Expected same n_features as train:", X_train_encoded.shape[1])
display(X_new_encoded.head())


# ============================================================
# 10) Predict with trained model
# ============================================================
pred_new = xgb_reg.predict(X_new_encoded)

# If your target is evaluate_note (0..10), clipping is often useful
pred_new_clipped = np.clip(pred_new, 0, 10)

print("\n✅ Predicted value (raw):", pred_new[0])
print("✅ Predicted value (clipped 0..10):", pred_new_clipped[0])


# ============================================================
# 11) SHAP local explanation for THIS prediction
#    This shows which features pushed prediction up/down
# ============================================================
# TreeExplainer works well with XGBoost tree models
explainer = shap.TreeExplainer(xgb_reg)

# Compute SHAP values for the single row
shap_values_single = explainer.shap_values(X_new_encoded)

# Expected value (base value)
base_value = explainer.expected_value

# If output is array-like (sometimes happens), take scalar
if isinstance(base_value, (list, np.ndarray)):
    # for single-output regression this is usually length-1
    base_value_scalar = float(np.array(base_value).reshape(-1)[0])
else:
    base_value_scalar = float(base_value)

# Build a local importance table (absolute contribution ranking)
local_shap_df = pd.DataFrame({
    "feature": X_new_encoded.columns,
    "feature_value": X_new_encoded.iloc[0].values,
    "shap_value": shap_values_single[0]
})

local_shap_df["abs_shap"] = local_shap_df["shap_value"].abs()
local_shap_df = local_shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

print("\n=== Local explanation for this row (top features that influenced prediction) ===")
print("Base value (average model output):", base_value_scalar)
print("Prediction from model:", float(pred_new[0]))
display(local_shap_df.head(20))


# ============================================================
# 12) SHAP waterfall plot (best for one-row explanation)
#    This visually shows why the model chose this prediction
# ============================================================
# Preferred modern API
try:
    shap_explanation = shap.Explanation(
        values=shap_values_single[0],
        base_values=base_value_scalar,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )
    shap.plots.waterfall(shap_explanation, max_display=20)
    plt.show()
except Exception as e:
    print("Waterfall plot failed, falling back to summary text.")
    print("Reason:", e)


# ============================================================
# 13) Optional: SHAP force plot (interactive in notebook)
# ============================================================
try:
    shap.initjs()
    force_plot = shap.force_plot(
        base_value_scalar,
        shap_values_single[0],
        X_new_encoded.iloc[0],
        matplotlib=True,   # static image in notebook
        show=True
    )
except Exception as e:
    print("Force plot skipped:", e)

```
