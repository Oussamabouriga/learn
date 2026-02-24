```
# ============================================================
# 2) Split columns into categorical / numerical
# 3) Apply chosen encodings by column name:
#    - One-Hot Encoding
#    - Target Encoding (fit on train only)
#    - Frequency/Count Encoding (fit on train only)
# ============================================================

# -------------------------
# Imports
# -------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Optional: if you want a cleaner display in notebook
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# -------------------------
# Example settings (EDIT THESE)
# -------------------------
target_col = "evaluate_note"   # <-- change to your target column name

# Columns you choose for each technique (by exact column name)
onehot_cols = [
    # "city", "channel"
]

target_encode_cols = [
    # "garage_name", "agent_name"
]

freq_encode_cols = [
    # "client_segment", "region_code"
]

# Optional: columns to force as categorical even if pandas guessed numeric
force_categorical_cols = [
    # "code_postal"
]

# Train/test split settings
test_size = 0.2
random_state = 42

# If your target is continuous (regression), stratify is usually None
# If you want pseudo-stratification on regression target, you can bin y separately (optional)
use_stratify = False

# Target encoding smoothing strength (higher = stronger smoothing toward global mean)
target_encoding_smoothing = 20

# Frequency encoding mode: "count" or "freq"
freq_mode = "count"   # "count" = counts, "freq" = proportions


# ============================================================
# STEP A — Basic checks
# ============================================================
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in df")

all_selected = set(onehot_cols + target_encode_cols + freq_encode_cols)

# Check duplicates across techniques
dup_check = (set(onehot_cols) & set(target_encode_cols)) | (set(onehot_cols) & set(freq_encode_cols)) | (set(target_encode_cols) & set(freq_encode_cols))
if len(dup_check) > 0:
    raise ValueError(f"These columns are assigned to more than one encoding technique: {dup_check}")

# Check columns exist
missing_selected = [c for c in all_selected if c not in df.columns]
if missing_selected:
    raise ValueError(f"These selected columns are not in df: {missing_selected}")

if target_col in all_selected:
    raise ValueError("Target column must not be included in encoding column lists.")


# ============================================================
# STEP B — Split X / y
# ============================================================
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Optional: remove rows where target is missing
target_notna_mask = y.notna()
X = X.loc[target_notna_mask].copy()
y = y.loc[target_notna_mask].copy()

# ============================================================
# STEP C — Detect categorical / numerical columns (before encoding)
# ============================================================
# Add forced categorical columns if needed
for c in force_categorical_cols:
    if c in X.columns:
        X[c] = X[c].astype("string")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Optional sanity check: selected encoding columns should generally be categorical
selected_not_in_X = [c for c in all_selected if c not in X.columns]
if selected_not_in_X:
    raise ValueError(f"Selected columns not in X: {selected_not_in_X}")


# ============================================================
# STEP D — Train / Test split
# IMPORTANT:
# - Target Encoding and Frequency Encoding MUST be fit on train only
# - Then applied to test using train mappings
# ============================================================
stratify_arg = y if use_stratify else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=stratify_arg
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# ============================================================
# STEP E — Prepare copies to transform
# ============================================================
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

# Keep track of mapping objects so you can reuse later on new data
encoding_artifacts = {
    "onehot_columns_created": [],
    "target_encoding_maps": {},
    "target_encoding_global_mean": None,
    "frequency_encoding_maps": {},
    "config": {
        "onehot_cols": onehot_cols,
        "target_encode_cols": target_encode_cols,
        "freq_encode_cols": freq_encode_cols,
        "target_col": target_col,
        "target_encoding_smoothing": target_encoding_smoothing,
        "freq_mode": freq_mode
    }
}


# ============================================================
# STEP F — Target Encoding (fit on TRAIN only)
# ------------------------------------------------------------
# Replaces category with a smoothed target mean
# Why "fit on train only"?
# - To avoid data leakage (test target must not influence encodings)
# ============================================================
global_target_mean = y_train.mean()
encoding_artifacts["target_encoding_global_mean"] = float(global_target_mean)

for col in target_encode_cols:
    # Convert to string category safely (so missing becomes a category label if needed)
    train_col = X_train_enc[col].astype("string")
    test_col = X_test_enc[col].astype("string")

    # Build stats on train only
    tmp = pd.DataFrame({
        col: train_col,
        "_target_": y_train.values
    })

    stats = tmp.groupby(col, dropna=False)["_target_"].agg(["mean", "count"]).reset_index()

    # Smoothed target mean:
    # enc = (count * cat_mean + smoothing * global_mean) / (count + smoothing)
    stats["target_enc"] = (
        (stats["count"] * stats["mean"]) + (target_encoding_smoothing * global_target_mean)
    ) / (stats["count"] + target_encoding_smoothing)

    te_map = pd.Series(stats["target_enc"].values, index=stats[col]).to_dict()

    # Save map
    encoding_artifacts["target_encoding_maps"][col] = te_map

    # Apply to train/test using train map only
    X_train_enc[f"{col}_te"] = train_col.map(te_map).astype(float)
    X_test_enc[f"{col}_te"] = test_col.map(te_map).astype(float)

    # Unseen categories in test -> fallback to global mean
    X_train_enc[f"{col}_te"] = X_train_enc[f"{col}_te"].fillna(global_target_mean)
    X_test_enc[f"{col}_te"] = X_test_enc[f"{col}_te"].fillna(global_target_mean)

    # Drop original column (optional, usually yes)
    X_train_enc.drop(columns=[col], inplace=True)
    X_test_enc.drop(columns=[col], inplace=True)

print("✅ Target Encoding applied on:", target_encode_cols)


# ============================================================
# STEP G — Frequency / Count Encoding (fit on TRAIN only)
# ------------------------------------------------------------
# Replaces category with:
# - count of occurrences in train (count mode)
# - proportion in train (freq mode)
# Why useful?
# - Good for high-cardinality categorical columns
# - Keeps only 1 numeric column per original feature
# ============================================================
for col in freq_encode_cols:
    train_col = X_train_enc[col].astype("string")
    test_col = X_test_enc[col].astype("string")

    if freq_mode == "count":
        freq_series = train_col.value_counts(dropna=False)
    elif freq_mode == "freq":
        freq_series = train_col.value_counts(dropna=False, normalize=True)
    else:
        raise ValueError("freq_mode must be 'count' or 'freq'")

    freq_map = freq_series.to_dict()
    encoding_artifacts["frequency_encoding_maps"][col] = freq_map

    X_train_enc[f"{col}_fe"] = train_col.map(freq_map).astype(float)
    X_test_enc[f"{col}_fe"] = test_col.map(freq_map).astype(float)

    # Unseen categories in test -> 0 (or you can choose np.nan)
    X_train_enc[f"{col}_fe"] = X_train_enc[f"{col}_fe"].fillna(0.0)
    X_test_enc[f"{col}_fe"] = X_test_enc[f"{col}_fe"].fillna(0.0)

    # Drop original column
    X_train_enc.drop(columns=[col], inplace=True)
    X_test_enc.drop(columns=[col], inplace=True)

print("✅ Frequency/Count Encoding applied on:", freq_encode_cols)


# ============================================================
# STEP H — One-Hot Encoding (fit on TRAIN columns, align TEST)
# ------------------------------------------------------------
# We use pandas.get_dummies here (simple notebook style, no function/pipeline)
# Important:
# - Create dummies on train and test
# - Then align columns so test has same columns as train
# ============================================================
# Only keep one-hot columns that still exist (not already dropped by TE/FE)
onehot_cols_existing = [c for c in onehot_cols if c in X_train_enc.columns]

if len(onehot_cols_existing) > 0:
    # Convert chosen OHE cols to string (safe / consistent)
    for c in onehot_cols_existing:
        X_train_enc[c] = X_train_enc[c].astype("string")
        X_test_enc[c] = X_test_enc[c].astype("string")

    X_train_enc = pd.get_dummies(
        X_train_enc,
        columns=onehot_cols_existing,
        dummy_na=True,      # add missing category column
        drop_first=False
    )

    X_test_enc = pd.get_dummies(
        X_test_enc,
        columns=onehot_cols_existing,
        dummy_na=True,
        drop_first=False
    )

    # Align test to train columns (critical)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    # Save created one-hot column names
    encoding_artifacts["onehot_columns_created"] = [c for c in X_train_enc.columns if any(c.startswith(f"{orig}_") for orig in onehot_cols_existing)]

print("✅ One-Hot Encoding applied on:", onehot_cols_existing)


# ============================================================
# STEP I — Final check: convert booleans to int (optional, cleaner for models)
# ============================================================
bool_cols_train = X_train_enc.select_dtypes(include=["bool"]).columns.tolist()
bool_cols_test = X_test_enc.select_dtypes(include=["bool"]).columns.tolist()

if bool_cols_train:
    X_train_enc[bool_cols_train] = X_train_enc[bool_cols_train].astype(int)
if bool_cols_test:
    X_test_enc[bool_cols_test] = X_test_enc[bool_cols_test].astype(int)

print("\nFinal encoded train shape:", X_train_enc.shape)
print("Final encoded test shape :", X_test_enc.shape)


# ============================================================
# STEP J — Preview results
# ============================================================
print("\nEncoded TRAIN sample:")
display(X_train_enc.head())

print("\nEncoded TEST sample:")
display(X_test_enc.head())

print("\nEncoding artifacts keys (save this for later prediction on new data):")
print(encoding_artifacts.keys())
print("\nTarget Encoding maps:", list(encoding_artifacts["target_encoding_maps"].keys()))
print("Frequency Encoding maps:", list(encoding_artifacts["frequency_encoding_maps"].keys()))


# ============================================================
# OPTIONAL — Quick notes (short)
# ============================================================
# - X_train_enc / X_test_enc are now ready for model training (XGBoost, etc.)
# - y_train / y_test are your targets
# - Target Encoding + Frequency Encoding were fit ONLY on train (correct)
# - One-Hot was aligned so test has the same columns as train


onehot_cols = ["city", "channel"]
target_encode_cols = ["garage_name"]
freq_encode_cols = ["client_segment"]
target_col = "evaluate_note"
```
