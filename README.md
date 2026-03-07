```
# ============================================================
# CatBoost Regressor — WEIGHTED DATA PREPARATION ONLY (MANUAL BINS)
# Unique names:
#   - X_train_cat_w, X_test_cat_w
#   - y_train_cat_w, y_test_cat_w
#   - y_bin_train_cat_w
#   - sample_weight_train_cat_w
#   - weights_debug_table_train_cat_w
#
# Manual bins:
#   - 0..6 together
#   - 7..8 together
#   - 9 alone
#   - 10 alone
#
# Inputs expected (CatBoost-ready data prepared earlier):
#   - X_train_cb, X_test_cb        (no one-hot / no freq encoding)
#   - y_train_no_te, y_test_no_te
#   - cat_cols_cb                  (categorical columns list for CatBoost Pool)
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Copy your CatBoost-ready dataset
# -----------------------------
X_train_cat_w = X_train_cb.copy()
X_test_cat_w  = X_test_cb.copy()

y_train_cat_w = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_cat_w  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

print("Train X:", X_train_cat_w.shape, "| Train y:", y_train_cat_w.shape)
print("Test  X:", X_test_cat_w.shape,  "| Test  y:", y_test_cat_w.shape)

# -----------------------------
# 1) Build sample weights on TRAIN ONLY (manual bins)
# -----------------------------
clip_min_train_cat_w = 0.5
clip_max_train_cat_w = 5.0

def _bin_0_6__7_8__9__10(y):
    n = int(np.rint(y))  # round to nearest integer note
    if n <= 6:
        return "0_6"
    elif n <= 8:
        return "7_8"
    elif n == 9:
        return "9"
    else:
        return "10"

# Bin label per train row
y_bin_train_cat_w = y_train_cat_w.apply(_bin_0_6__7_8__9__10)

# Count per bin
bin_counts_train_cat_w = y_bin_train_cat_w.value_counts(dropna=False)

# Raw weights = 1 / count(bin)
raw_weight_train_cat_w = y_bin_train_cat_w.map(
    lambda b: 1.0 / bin_counts_train_cat_w[b]
).astype(float).values

# Normalize around 1.0
sample_weight_train_cat_w = raw_weight_train_cat_w / np.mean(raw_weight_train_cat_w)

# Clip for stability
sample_weight_train_cat_w = np.clip(sample_weight_train_cat_w, clip_min_train_cat_w, clip_max_train_cat_w)

print("\n✅ sample_weight_train_cat_w created")
print(pd.Series(sample_weight_train_cat_w).describe())

# -----------------------------
# 2) Debug table: bin -> count -> weight
# -----------------------------
weights_debug_table_train_cat_w = pd.DataFrame({
    "bin": bin_counts_train_cat_w.index.astype(str),
    "count_in_bin": bin_counts_train_cat_w.values
})

weights_debug_table_train_cat_w["raw_weight_1_over_count"] = 1.0 / weights_debug_table_train_cat_w["count_in_bin"]
weights_debug_table_train_cat_w["normalized_weight"] = (
    weights_debug_table_train_cat_w["raw_weight_1_over_count"]
    / weights_debug_table_train_cat_w["raw_weight_1_over_count"].mean()
)
weights_debug_table_train_cat_w["clipped_weight_used"] = weights_debug_table_train_cat_w["normalized_weight"].clip(
    clip_min_train_cat_w, clip_max_train_cat_w
)

weights_debug_table_train_cat_w = weights_debug_table_train_cat_w.sort_values(
    "count_in_bin", ascending=True
).reset_index(drop=True)

print("\n=== Bin weights table (train_cat_w) ===")
display(weights_debug_table_train_cat_w)

# -----------------------------
# 3) Example mapping (one row)
# -----------------------------
example_note_train_cat_w = float(y_train_cat_w.iloc[0])
example_bin_train_cat_w = y_bin_train_cat_w.iloc[0]
example_weight_train_cat_w = float(sample_weight_train_cat_w[0])

print("\nExample mapping (train_cat_w):")
print("note:", example_note_train_cat_w)
print("bin :", example_bin_train_cat_w)
print("weight used:", example_weight_train_cat_w)

# Later (training) you will do:
# from catboost import Pool
# train_pool_cat_w = Pool(X_train_cat_w, y_train_cat_w, cat_features=cat_cols_cb, weight=sample_weight_train_cat_w)
# test_pool_cat_w  = Pool(X_test_cat_w,  y_test_cat_w,  cat_features=cat_cols_cb)
```
