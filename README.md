```

Perfect — let’s restart cleanly from the beginning and do it correctly (no functions, only code blocks), using your columns and your idea:
	•	convert selected 0 values to real np.nan
	•	split target / features
	•	force some columns as categorical (like code_postal)
	•	train/test split
	•	apply One-Hot on selected columns
	•	apply Frequency/Count Encoding on selected columns
	•	(optional) apply Target Encoding on selected columns (you currently set it to [])
	•	apply log transform on selected numeric columns you choose
	•	make final encoded datasets ready for XGBoost

⸻

1) Imports (start here)

# ==============================
# 1) Imports
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


⸻

2) Define what you want to encode / transform (your setup)

# ==============================
# 2) Your configuration (EDIT here)
# ==============================

# Target
target_col = "evaluate_note"

# Categorical encoding choices
onehot_cols = ['PARCOURS_FINAL', 'PARCOURS_INITIAL', 'operating_system']
target_encode_cols = []   # keep empty for now (you can add later)
freq_encode_cols = ['marque', 'model', 'garantie', 'list_prest']

# Columns to force as categorical even if they look numeric
force_categorical_cols = ['code_postal']

# Columns where 0 means "missing" (replace 0 -> np.nan)
zero_to_nan_cols = [
    'Nombre_sisnistre_refuse_client',
    'delai_de_completude',
    'delai_reparation',
    'delai_indemnisation',
    # add more if in your business logic 0 = no value / not applicable
]

# Numeric columns to apply log transform on (only positive-skewed variables)
# IMPORTANT: use columns where log makes sense (delays, amounts, counts...)
log_transform_cols = [
    'delai_declaration',
    'delai_de_completude',
    'delai_decision',
    'delai_reparation',
    'delai_indemnisation',
    'montant_indem',
    'delai_Sinistre'
]

# Split config
test_size = 0.2
random_state = 42


⸻

3) Copy data + basic cleaning + convert selected 0 to np.nan

# ==============================
# 3) Copy dataframe + basic cleaning
# ==============================
df_work = df.copy()

# Optional: strip spaces from column names (safe)
df_work.columns = df_work.columns.astype(str).str.strip()

# Make sure target exists
if target_col not in df_work.columns:
    raise ValueError(f"Target column '{target_col}' not found in df")

# ==============================
# 3.1) Convert selected 0 values to np.nan (REAL NaN)
# ==============================
for col in zero_to_nan_cols:
    if col in df_work.columns:
        # Convert to numeric first (invalid strings -> NaN)
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        # Replace business-zero with real numpy NaN
        df_work[col] = df_work[col].replace(0, np.nan)

# Quick check
print("Zero->NaN conversion done. Missing counts in selected columns:")
for col in zero_to_nan_cols:
    if col in df_work.columns:
        print(col, ":", df_work[col].isna().sum())


⸻

4) Force some columns to categorical + split X / y

# ==============================
# 4) Force selected columns as categorical (string)
# ==============================
for col in force_categorical_cols:
    if col in df_work.columns:
        df_work[col] = df_work[col].astype("string")

# Also make sure chosen categorical encoding columns are treated as string
for col in (onehot_cols + target_encode_cols + freq_encode_cols):
    if col in df_work.columns:
        df_work[col] = df_work[col].astype("string")

# ==============================
# 4.1) Split X and y
# ==============================
X = df_work.drop(columns=[target_col]).copy()
y = pd.to_numeric(df_work[target_col], errors="coerce").copy()

# Drop rows where target is missing
mask_target_ok = y.notna()
X = X.loc[mask_target_ok].copy()
y = y.loc[mask_target_ok].astype(float).copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Target missing removed:", (~mask_target_ok).sum())


⸻

5) Train/Test split (before encoding to avoid leakage)

# ==============================
# 5) Train / Test split (BEFORE encoding)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)


⸻

6) Frequency / Count Encoding (fit on train only, apply to train/test)

Explanation (simple)

Frequency encoding replaces each category by how often it appears in the training data.

Example:
	•	marque = "Samsung" appears 1200 times → encoded as 1200
	•	marque = "Apple" appears 300 times → encoded as 300

Why useful:
	•	good for high-cardinality columns (many categories)
	•	keeps feature count small (unlike one-hot explosion)

⸻


# ==============================
# 6) Frequency / Count Encoding (fit on TRAIN only)
# ==============================
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

encoding_artifacts = {
    "onehot_columns_created": [],
    "target_encoding_maps": {},
    "target_encoding_global_mean": None,
    "frequency_encoding_maps": {}
}

for col in freq_encode_cols:
    if col in X_train_enc.columns:
        # Use string category values (including missing as text label)
        train_col_str = X_train_enc[col].astype("string").fillna("__MISSING__")
        test_col_str  = X_test_enc[col].astype("string").fillna("__MISSING__")

        # Count encoding map (you can switch to normalized freq if you want)
        freq_map = train_col_str.value_counts(dropna=False).to_dict()

        # Save map for future prediction on new data
        encoding_artifacts["frequency_encoding_maps"][col] = freq_map

        # Apply map
        X_train_enc[col] = train_col_str.map(freq_map).astype(float)
        X_test_enc[col]  = test_col_str.map(freq_map).fillna(0).astype(float)  # unseen categories -> 0

print("Frequency encoding done for:", list(encoding_artifacts["frequency_encoding_maps"].keys()))


⸻

7) Target Encoding (optional — fit on train only)

You currently set target_encode_cols = [], so this block will just skip.

# ==============================
# 7) Target Encoding (optional, fit on TRAIN only)
# ==============================
# Target encoding = replace each category with mean(target) in train
# Important: fit on TRAIN only to avoid leakage

global_target_mean = float(y_train.mean())
encoding_artifacts["target_encoding_global_mean"] = global_target_mean

for col in target_encode_cols:
    if col in X_train_enc.columns:
        train_col_str = X_train_enc[col].astype("string").fillna("__MISSING__")
        test_col_str  = X_test_enc[col].astype("string").fillna("__MISSING__")

        te_map = pd.DataFrame({
            "cat": train_col_str,
            "target": y_train.values
        }).groupby("cat")["target"].mean().to_dict()

        encoding_artifacts["target_encoding_maps"][col] = te_map

        X_train_enc[col] = train_col_str.map(te_map).fillna(global_target_mean).astype(float)
        X_test_enc[col]  = test_col_str.map(te_map).fillna(global_target_mean).astype(float)

print("Target encoding done for:", list(encoding_artifacts["target_encoding_maps"].keys()))


⸻

8) One-Hot Encoding (fit on train only, apply to test)

# ==============================
# 8) One-Hot Encoding (fit on TRAIN only)
# ==============================
# We'll one-hot only the columns you selected in onehot_cols

onehot_cols_existing = [c for c in onehot_cols if c in X_train_enc.columns]
print("OneHot columns found:", onehot_cols_existing)

if len(onehot_cols_existing) > 0:
    # Prepare as string + explicit missing token
    X_train_ohe_input = X_train_enc[onehot_cols_existing].astype("string").fillna("__MISSING__")
    X_test_ohe_input  = X_test_enc[onehot_cols_existing].astype("string").fillna("__MISSING__")

    # sklearn compatibility across versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train_ohe = ohe.fit_transform(X_train_ohe_input)
    X_test_ohe  = ohe.transform(X_test_ohe_input)

    # Get generated column names
    ohe_feature_names = ohe.get_feature_names_out(onehot_cols_existing).tolist()

    # Make DataFrames with same index
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe_feature_names, index=X_train_enc.index)
    X_test_ohe_df  = pd.DataFrame(X_test_ohe, columns=ohe_feature_names, index=X_test_enc.index)

    # Drop original one-hot columns, then concat encoded columns
    X_train_enc = X_train_enc.drop(columns=onehot_cols_existing)
    X_test_enc  = X_test_enc.drop(columns=onehot_cols_existing)

    X_train_enc = pd.concat([X_train_enc, X_train_ohe_df], axis=1)
    X_test_enc  = pd.concat([X_test_enc, X_test_ohe_df], axis=1)

    encoding_artifacts["onehot_columns_created"] = ohe_feature_names
else:
    print("No one-hot columns to encode.")


⸻

9) Log transform selected numeric columns (safe way)

Why log transform?

Useful for skewed variables like:
	•	delays
	•	amounts
	•	counts

It compresses very large values and helps the model learn patterns better.

We use np.log1p(x) = log(1 + x) because it handles zero safely.

# ==============================
# 9) Log transform selected numerical columns (safe)
# ==============================
log_cols_existing = [c for c in log_transform_cols if c in X_train_enc.columns]
print("Log-transform columns found:", log_cols_existing)

for col in log_cols_existing:
    # Force numeric (invalid values -> NaN)
    X_train_enc[col] = pd.to_numeric(X_train_enc[col], errors="coerce")
    X_test_enc[col]  = pd.to_numeric(X_test_enc[col], errors="coerce")

    # If negatives exist, skip log OR clip depending on business meaning
    train_has_negative = (X_train_enc[col].dropna() < 0).any()
    test_has_negative  = (X_test_enc[col].dropna() < 0).any()

    if train_has_negative or test_has_negative:
        print(f"Skipping log for '{col}' (contains negative values)")
        continue

    # Create NEW columns (recommended, keeps original too)
    X_train_enc[f"{col}__log1p"] = np.log1p(X_train_enc[col])
    X_test_enc[f"{col}__log1p"]  = np.log1p(X_test_enc[col])

print("Log transform done.")


⸻

10) Convert remaining object/string columns (if any)

At this point, XGBoost needs numeric values.
If any categorical columns remain (not encoded), we either encode them or drop them.

This block helps you detect them.

# ==============================
# 10) Check remaining non-numeric columns
# ==============================
remaining_non_numeric_train = X_train_enc.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
remaining_non_numeric_test  = X_test_enc.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

print("Remaining non-numeric columns in TRAIN:", remaining_non_numeric_train)
print("Remaining non-numeric columns in TEST :", remaining_non_numeric_test)

If you still have columns there, it means:
	•	they were not included in onehot / target / frequency encoding
	•	and they are still strings

You can either:
	•	add them to one of your encoding lists, or
	•	drop them now (temporary)

⸻

11) Final numeric conversion for XGBoost (and keep np.nan)

# ==============================
# 11) Final conversion to numeric (XGBoost-ready)
# ==============================
# Convert booleans to int (optional)
bool_cols_train = X_train_enc.select_dtypes(include=["bool"]).columns.tolist()
bool_cols_test = X_test_enc.select_dtypes(include=["bool"]).columns.tolist()

if len(bool_cols_train) > 0:
    X_train_enc[bool_cols_train] = X_train_enc[bool_cols_train].astype(int)
if len(bool_cols_test) > 0:
    X_test_enc[bool_cols_test] = X_test_enc[bool_cols_test].astype(int)

# Convert every column to numeric (invalid -> NaN)
for c in X_train_enc.columns:
    X_train_enc[c] = pd.to_numeric(X_train_enc[c], errors="coerce")

for c in X_test_enc.columns:
    X_test_enc[c] = pd.to_numeric(X_test_enc[c], errors="coerce")

# Cast to float (keeps NaN as np.nan)
X_train_encoded = X_train_enc.astype(float).copy()
X_test_encoded  = X_test_enc.astype(float).copy()

# Target numeric
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test  = pd.to_numeric(y_test, errors="coerce").astype(float)

# Drop rows where target is missing (if any)
train_mask = y_train.notna()
test_mask  = y_test.notna()

X_train_encoded = X_train_encoded.loc[train_mask].copy()
y_train = y_train.loc[train_mask].copy()

X_test_encoded = X_test_encoded.loc[test_mask].copy()
y_test = y_test.loc[test_mask].copy()

print("Final encoded train shape:", X_train_encoded.shape)
print("Final encoded test shape :", X_test_encoded.shape)
print("Target train shape:", y_train.shape)
print("Target test shape :", y_test.shape)


⸻

12) Clean feature names for XGBoost (IMPORTANT — fixes your previous error)

This is the fix for:

feature_names must be string, and may not contain [, ] or <

# ==============================
# 12) Clean feature names for XGBoost
# ==============================
X_train_encoded.columns = X_train_encoded.columns.astype(str)
X_test_encoded.columns  = X_test_encoded.columns.astype(str)

X_train_encoded.columns = (
    X_train_encoded.columns
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "lt_", regex=False)
    .str.replace(">", "gt_", regex=False)
    .str.replace(" ", "_", regex=False)
)

X_test_encoded.columns = (
    X_test_encoded.columns
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "lt_", regex=False)
    .str.replace(">", "gt_", regex=False)
    .str.replace(" ", "_", regex=False)
)

# Ensure same columns / order
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Handle duplicate names after cleaning (rare but possible)
if X_train_encoded.columns.duplicated().any():
    new_cols = []
    counts = {}
    for c in X_train_encoded.columns:
        if c not in counts:
            counts[c] = 0
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f"{c}__dup{counts[c]}")
    X_train_encoded.columns = new_cols
    X_test_encoded.columns = new_cols

print("Bad feature names remaining:",
      [c for c in X_train_encoded.columns if ("[" in c or "]" in c or "<" in c)][:10])
print("All feature names ready for XGBoost ✅")


⸻

13) Preview results (so you verify before training)

# ==============================
# 13) Preview final transformed data
# ==============================
print("\nEncoded TRAIN sample:")
display(X_train_encoded.head())

print("\nEncoded TEST sample:")
display(X_test_encoded.head())

print("\nEncoding artifacts keys:")
print(encoding_artifacts.keys())

print("\nTarget Encoding maps:", list(encoding_artifacts["target_encoding_maps"].keys()))
print("Frequency Encoding maps:", list(encoding_artifacts["frequency_encoding_maps"].keys()))

print("\nMissing values in X_train_encoded (top 20):")
print(X_train_encoded.isna().sum().sort_values(ascending=False).head(20))


⸻

Other numerical transformation methods (what you asked)

For numerical columns (besides log transform), here are good options:
	•	No transform (often fine for XGBoost) ✅
Tree models are less sensitive to scale than linear models.
	•	log1p transform ✅
Best for positive skewed data (amounts, delays, counts).
	•	Robust scaling (median/IQR)
Useful if you use linear models / neural nets. Less necessary for XGBoost.
	•	Winsorization / clipping
Cap extreme outliers (e.g., 99th percentile) if huge outliers distort behavior.
	•	Yeo-Johnson / Box-Cox
More advanced distribution transforms. Usually not first choice with XGBoost.

Practical recommendation for your case (XGBoost + tabular)
	•	Keep raw numeric columns
	•	Add log1p versions for very skewed columns
	•	Keep NaN as NaN (XGBoost can use missing values)
	•	No need for StandardScaler for XGBoost

⸻

Important note about missing values (np.nan vs <NA>)

You asked this before:
	•	np.nan ✅ standard missing value (NumPy)
	•	<NA> ✅ pandas missing representation (nullable dtype display)

After final conversion to float, missing values become np.nan, which XGBoost handles.


```
