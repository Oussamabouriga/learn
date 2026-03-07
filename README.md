```

import numpy as np
import pandas as pd
from catboost import Pool

# ============================================================
# FINAL CLEANUP (baseline CatBoost, no weights) - BEFORE Pool
# ============================================================

# Replace pd.NA -> np.nan (CatBoost cannot handle pd.NA)
X_train_cb_base = X_train_cb_base.astype(object).where(pd.notna(X_train_cb_base), np.nan)
X_test_cb_base  = X_test_cb_base.astype(object).where(pd.notna(X_test_cb_base),  np.nan)

# Force categorical columns to string
for c in cat_cols_cb:
    if c in X_train_cb_base.columns:
        X_train_cb_base[c] = X_train_cb_base[c].astype(str)
        X_test_cb_base[c]  = X_test_cb_base[c].astype(str)

        # Normalize missing labels inside categoricals
        X_train_cb_base.loc[X_train_cb_base[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
        X_test_cb_base.loc[X_test_cb_base[c].isin(["nan", "None", "<NA>"]), c]  = "__MISSING__"

# Make sure non-categorical columns are numeric
num_cols_cb_base = [c for c in X_train_cb_base.columns if c not in cat_cols_cb]
for c in num_cols_cb_base:
    X_train_cb_base[c] = pd.to_numeric(X_train_cb_base[c], errors="coerce")
    X_test_cb_base[c]  = pd.to_numeric(X_test_cb_base[c],  errors="coerce")

# ============================================================
# Create Pools (NO weights)
# ============================================================
train_pool_cb_base = Pool(X_train_cb_base, y_train_cb_base, cat_features=cat_cols_cb)
test_pool_cb_base  = Pool(X_test_cb_base,  y_test_cb_base,  cat_features=cat_cols_cb)
```
