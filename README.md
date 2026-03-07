```
# ============================================================
# XGBoost Regressor — WEIGHTED DATA PREPARATION ONLY
# (Imbalanced Regression) — Names include "xgboost_reg"
#
# Inputs already prepared (no target encoding):
#   - X_train_encoded_no_te, X_test_encoded_no_te
#   - y_train_no_te, y_test_no_te
#
# Outputs:
#   - X_train_xgboost_reg_w, X_test_xgboost_reg_w
#   - y_train_xgboost_reg_w, y_test_xgboost_reg_w
#   - sample_weight_train_xgboost_reg_w
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Copy your prepared dataset (NO target encoding)
# -----------------------------
X_train_xgboost_reg_w = X_train_encoded_no_te.copy().astype(float)
X_test_xgboost_reg_w  = X_test_encoded_no_te.copy().astype(float)

y_train_xgboost_reg_w = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_xgboost_reg_w  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

print("Train X:", X_train_xgboost_reg_w.shape, "| Train y:", y_train_xgboost_reg_w.shape)
print("Test  X:", X_test_xgboost_reg_w.shape,  "| Test  y:", y_test_xgboost_reg_w.shape)

# -----------------------------
# 1) Build sample weights on TRAIN ONLY
# Method: Target-Bin Inverse Frequency Weighting
# -----------------------------
n_bins_xgboost_reg_w = 10
clip_min_xgboost_reg_w = 0.5
clip_max_xgboost_reg_w = 5.0

# Create bins on y_train (quantile bins if possible)
try:
    y_bins_xgboost_reg_w = pd.qcut(
        y_train_xgboost_reg_w,
        q=min(n_bins_xgboost_reg_w, y_train_xgboost_reg_w.nunique()),
        duplicates="drop"
    )
except Exception:
    y_bins_xgboost_reg_w = pd.cut(
        y_train_xgboost_reg_w,
        bins=min(n_bins_xgboost_reg_w, max(2, y_train_xgboost_reg_w.nunique()))
    )

bin_counts_xgboost_reg_w = y_bins_xgboost_reg_w.value_counts(dropna=False)

# weight = 1 / freq(bin)
sample_weight_train_xgboost_reg_w = y_bins_xgboost_reg_w.map(
    lambda b: 1.0 / bin_counts_xgboost_reg_w[b]
).astype(float).values

# Normalize around 1.0 + clip for stability
sample_weight_train_xgboost_reg_w = sample_weight_train_xgboost_reg_w / np.mean(sample_weight_train_xgboost_reg_w)
sample_weight_train_xgboost_reg_w = np.clip(
    sample_weight_train_xgboost_reg_w,
    clip_min_xgboost_reg_w,
    clip_max_xgboost_reg_w
)

print("\n✅ sample_weight_train_xgboost_reg_w created")
print(pd.Series(sample_weight_train_xgboost_reg_w).describe())

```
