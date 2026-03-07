```
# ============================================================
# XGBoost CLASSIFICATION DATASET (NO Target Encoding)
# Same transformations as your regression dataset:
#   - uses the already prepared X_train_encoded_no_te / X_test_encoded_no_te
# Then creates classification labels from y:
#   0..2  -> "extrêmement mauvais"
#   3..6  -> "mauvais"
#   7..8  -> "neutre"
#   9     -> "bien"
#   10    -> "très bien"
#
# Outputs (unique names):
#   - X_train_xgb_cls_no_te, X_test_xgb_cls_no_te
#   - y_train_xgb_cls_no_te, y_test_xgb_cls_no_te   (class ids 0..4)
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Use your transformed features (same as regression)
# -----------------------------
X_train_xgb_cls_no_te = X_train_encoded_no_te.copy().astype(float)
X_test_xgb_cls_no_te  = X_test_encoded_no_te.copy().astype(float)

# Use your original target (no_te split)
y_train_raw_cls = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_raw_cls  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

# -----------------------------
# 1) Build class labels (0..4)
# -----------------------------
def to_satisfaction_class_id(y):
    # we round to nearest integer note to be robust with 7.8 etc.
    n = int(np.rint(y))
    if n <= 2:
        return 0  # extrêmement mauvais
    elif n <= 6:
        return 1  # mauvais
    elif n <= 8:
        return 2  # neutre
    elif n == 9:
        return 3  # bien
    else:
        return 4  # très bien

y_train_xgb_cls_no_te = y_train_raw_cls.apply(to_satisfaction_class_id).astype(int)
y_test_xgb_cls_no_te  = y_test_raw_cls.apply(to_satisfaction_class_id).astype(int)

# Optional: mapping (for readability)
class_names_fr = {
    0: "extrêmement mauvais (0–2)",
    1: "mauvais (3–6)",
    2: "neutre (7–8)",
    3: "bien (9)",
    4: "très bien (10)"
}

print("✅ X_train_xgb_cls_no_te:", X_train_xgb_cls_no_te.shape)
print("✅ X_test_xgb_cls_no_te :", X_test_xgb_cls_no_te.shape)

print("\nTrain class distribution:")
print(y_train_xgb_cls_no_te.value_counts().sort_index().rename(index=class_names_fr))

print("\nTest class distribution:")
print(y_test_xgb_cls_no_te.value_counts().sort_index().rename(index=class_names_fr))

```
