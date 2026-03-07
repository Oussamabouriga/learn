```
import numpy as np
import pandas as pd

# -----------------------------
# 7bis) FINAL CLEANUP for CatBoost (SAFE)
# - pd.NA -> np.nan
# - categorical: keep real values, replace ONLY missing with "__MISSING__"
# - numeric: convert to numeric (np.nan ok)
# -----------------------------

# 1) Replace pd.NA -> np.nan everywhere (train + test)
X_train_cb = X_train_cb.astype(object).where(pd.notna(X_train_cb), np.nan)
X_test_cb  = X_test_cb.astype(object).where(pd.notna(X_test_cb),  np.nan)

# 2) Categorical cleanup (ONLY missing -> "__MISSING__")
missing_tokens = {"", "nan", "None", "<NA>", "NaN", "NULL", "null"}

for c in cat_cols_cb:
    if c in X_train_cb.columns:
        # convert to string safely (keep NaN as NaN first)
        s_tr = X_train_cb[c]
        s_te = X_test_cb[c]

        # mark missing before converting to string
        miss_tr = s_tr.isna()
        miss_te = s_te.isna()

        # convert non-missing to string
        X_train_cb.loc[~miss_tr, c] = s_tr.loc[~miss_tr].astype(str).str.strip()
        X_test_cb.loc[~miss_te,  c] = s_te.loc[~miss_te].astype(str).str.strip()

        # set missing to "__MISSING__"
        X_train_cb.loc[miss_tr, c] = "__MISSING__"
        X_test_cb.loc[miss_te,  c] = "__MISSING__"

        # extra safety: if some non-missing became a missing token string
        X_train_cb.loc[X_train_cb[c].isin(missing_tokens), c] = "__MISSING__"
        X_test_cb.loc[X_test_cb[c].isin(missing_tokens),  c] = "__MISSING__"

# 3) Numeric conversion
for c in numeric_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")

# -----------------------------
# Quick TESTS (to ensure not all became __MISSING__)
# -----------------------------
# show % missing label per categorical column
missing_rate = (X_train_cb[cat_cols_cb] == "__MISSING__").mean().sort_values(ascending=False)
print("Top categorical columns with __MISSING__ rate (train):")
print(missing_rate.head(15))

# sanity: check that some columns are NOT all missing
all_missing_cols = missing_rate[missing_rate == 1.0].index.tolist()
print("\nColumns that became 100% '__MISSING__' (should usually be empty):")
print(all_missing_cols)

```



```




from catboost import Pool

train_pool_catboost_baseline = Pool(
    X_train_cb,
    y_train_no_te,
    cat_features=cat_cols_cb
)

test_pool_catboost_baseline = Pool(
    X_test_cb,
    y_test_no_te,
    cat_features=cat_cols_cb
)


from catboost import Pool

train_pool_catboost_weighted = Pool(
    X_train_cb,
    y_train_no_te,
    cat_features=cat_cols_cb,
    weight=sample_weight_train_cat_w
)

test_pool_catboost_weighted = Pool(
    X_test_cb,
    y_test_no_te,
    cat_features=cat_cols_cb
)
```
