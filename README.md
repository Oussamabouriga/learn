```
# ==============================
# 1) Split Train/Test (NO Target Encoding version)
# ==============================
from sklearn.model_selection import train_test_split

target_col = "evaluate_note"  # change if needed

X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

X_train_no_te, X_test_no_te, y_train_no_te, y_test_no_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train (no TE):", X_train_no_te.shape, y_train_no_te.shape)
print("Test  (no TE):", X_test_no_te.shape, y_test_no_te.shape)




# ==============================
# 2) 0 -> NaN conversion (NO Target Encoding version)
# ==============================
import numpy as np
import pandas as pd

zero_to_nan_cols = [
    "col1",   # put your columns here
    # "col2",
]

for c in zero_to_nan_cols:
    if c in X_train_no_te.columns:
        X_train_no_te[c] = pd.to_numeric(X_train_no_te[c], errors="coerce")
        X_test_no_te[c]  = pd.to_numeric(X_test_no_te[c],  errors="coerce")

        X_train_no_te.loc[X_train_no_te[c] == 0, c] = np.nan
        X_test_no_te.loc[X_test_no_te[c] == 0, c]   = np.nan

print("Done 0->NaN (no TE) for:", [c for c in zero_to_nan_cols if c in X_train_no_te.columns])
```
