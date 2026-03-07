```

# ============================================================
# CATBOOST CLASSIFICATION DATASET (NO one-hot / NO frequency encoding)
# Same transformations as your CatBoost regression dataset:
#   - uses X_train_cb / X_test_cb (CatBoost-ready features)
#   - builds classification labels from y:
#       0..2  -> 0 (extrêmement mauvais)
#       3..6  -> 1 (mauvais)
#       7..8  -> 2 (neutre)
#       9     -> 3 (bien)
#       10    -> 4 (très bien)
#
# Outputs (unique names):
#   - X_train_cat_cls_no_te, X_test_cat_cls_no_te
#   - y_train_cat_cls_no_te, y_test_cat_cls_no_te
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Use your CatBoost-ready features (already transformed)
# -----------------------------
X_train_cat_cls_no_te = X_train_cb.copy()
X_test_cat_cls_no_te  = X_test_cb.copy()

# Use your original target (no_te split)
y_train_raw_cat_cls = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_raw_cat_cls  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

# -----------------------------
# 1) Build class labels (0..4)
# -----------------------------
def to_satisfaction_class_id(y):
    n = int(np.rint(y))  # robust if 7.8 etc.
    if n <= 2:
        return 0
    elif n <= 6:
        return 1
    elif n <= 8:
        return 2
    elif n == 9:
        return 3
    else:
        return 4

y_train_cat_cls_no_te = y_train_raw_cat_cls.apply(to_satisfaction_class_id).astype(int)
y_test_cat_cls_no_te  = y_test_raw_cat_cls.apply(to_satisfaction_class_id).astype(int)

class_names_fr = {
    0: "extrêmement mauvais (0–2)",
    1: "mauvais (3–6)",
    2: "neutre (7–8)",
    3: "bien (9)",
    4: "très bien (10)"
}

print("✅ X_train_cat_cls_no_te:", X_train_cat_cls_no_te.shape)
print("✅ X_test_cat_cls_no_te :", X_test_cat_cls_no_te.shape)

print("\nTrain class distribution:")
print(y_train_cat_cls_no_te.value_counts().sort_index().rename(index=class_names_fr))

print("\nTest class distribution:")
print(y_test_cat_cls_no_te.value_counts().sort_index().rename(index=class_names_fr))

# CatBoost classification will use:
# - X_train_cat_cls_no_te / y_train_cat_cls_no_te
# - X_test_cat_cls_no_te / y_test_cat_cls_no_te
# - cat_cols_cb as cat_features
```
