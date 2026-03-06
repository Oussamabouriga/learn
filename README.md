```

import numpy as np
import pandas as pd

# ============================================================
# Build ONE example row and apply the SAME transformations
# (NO Target Encoding version)
# Output: X_new_encoded_no_te aligned with X_train_encoded_no_te
# ============================================================

# 0) Your input
test_row_no_te = [{
    # "col1": value,
    # "code_postal": 59700,
}]

X_new = pd.DataFrame(test_row_no_te).copy()

# 1) Make sure it contains all raw columns (missing ones -> NaN)
#    (use X_train_no_te columns as reference because it’s "raw before encoding")
for c in X_train_no_te.columns:
    if c not in X_new.columns:
        X_new[c] = np.nan

# Keep same order as training raw
X_new = X_new[X_train_no_te.columns].copy()

# 2) Force code_postal categorical (same as training)
col_cp = "code_postal"
if col_cp in X_new.columns:
    X_new[col_cp] = X_new[col_cp].astype("string")

# 3) 0 -> NaN conversion (if you have that list)
#    (ONLY for columns where 0 means "missing" in your business logic)
#    Make sure you defined `zero_to_nan_cols` earlier.
if "zero_to_nan_cols" in globals():
    for c in zero_to_nan_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
            X_new.loc[X_new[c] == 0, c] = np.nan

# 4) Apply Frequency Encoding (using TRAIN-fitted freq_maps)
#    (freq_cols & freq_maps must exist from your earlier step)
X_new_fe = X_new.copy()

for c in freq_cols:
    if c in X_new_fe.columns:
        X_new_fe[c] = X_new_fe[c].map(freq_maps[c]).fillna(0).astype(float)

# 5) Apply One-Hot Encoding (same columns list as training)
X_new_oh = pd.get_dummies(X_new_fe, columns=onehot_cols, dummy_na=True)

# 6) Align to training encoded columns (important!)
X_new_encoded_no_te = X_new_oh.reindex(columns=X_train_encoded_no_te.columns, fill_value=0)

# 7) Apply log1p to the same columns (after encoding)
#    log1p_cols must exist from earlier step
for c in log1p_cols:
    if c in X_new_encoded_no_te.columns:
        X_new_encoded_no_te[c] = pd.to_numeric(X_new_encoded_no_te[c], errors="coerce").fillna(0)
        X_new_encoded_no_te[c] = X_new_encoded_no_te[c].clip(lower=0)
        X_new_encoded_no_te[c] = np.log1p(X_new_encoded_no_te[c])

# 8) Apply the SAME scaler (fitted on train) on the SAME columns
#    scaler and scale_cols must exist from earlier step
if "scaler" in globals() and "scale_cols" in globals() and len(scale_cols) > 0:
    X_new_encoded_no_te[scale_cols] = scaler.transform(X_new_encoded_no_te[scale_cols])

# 9) Final safety for XGBoost/CatBoost
X_new_encoded_no_te = X_new_encoded_no_te.astype(float)

print("✅ X_new_encoded_no_te shape:", X_new_encoded_no_te.shape)
print("✅ Same columns as train:", X_new_encoded_no_te.columns.equals(X_train_encoded_no_te.columns))
```
