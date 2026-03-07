```
# ============================================================
# XGBoost CLASSIFIER — BASELINE (NO target weighting, NO optimization)
# - Train
# - Metrics (train + test): Accuracy, Precision, Recall, F1 (macro + weighted), ROC-AUC (OvR)
# - ROC Curve (OvR) for each class
# - Confusion Matrix
# - SHAP global + SHAP local (example row)
#
# INPUTS expected (already prepared):
#   X_train_xgb_cls_no_te, X_test_xgb_cls_no_te
#   y_train_xgb_cls_no_te, y_test_xgb_cls_no_te     (class ids 0..4)
#
# Example row:
#   X_new_encoded_no_te (encoded features) OR X_new_xgboost_cls_no_te
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

import shap


# ==============================
# 1) Data
# ==============================
X_train_xgb_cls_base = X_train_xgb_cls_no_te.copy().astype(float)
X_test_xgb_cls_base  = X_test_xgb_cls_no_te.copy().astype(float)

y_train_xgb_cls_base = pd.to_numeric(y_train_xgb_cls_no_te, errors="coerce").astype(int)
y_test_xgb_cls_base  = pd.to_numeric(y_test_xgb_cls_no_te,  errors="coerce").astype(int)

num_classes = int(len(np.unique(y_train_xgb_cls_base)))

print("Train:", X_train_xgb_cls_base.shape, y_train_xgb_cls_base.shape)
print("Test :", X_test_xgb_cls_base.shape,  y_test_xgb_cls_base.shape)
print("Classes:", sorted(np.unique(y_train_xgb_cls_base)))


# ==============================
# 2) Hyperparameters (dict)
# ==============================
xgb_cls_params_base = {
    "objective": "multi:softprob",   # probability output for multi-class
    "num_class": num_classes,
    "eval_metric": "mlogloss",

    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "gamma": 0.0,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

print("\nHyperparameters (baseline classifier):")
display(pd.DataFrame([xgb_cls_params_base]).T.rename(columns={0: "value"}))


# ==============================
# 3) Train baseline classifier
# ==============================
xgboost_cls_base = XGBClassifier(**xgb_cls_params_base)
xgboost_cls_base.fit(X_train_xgb_cls_base, y_train_xgb_cls_base)

print("\nModel trained (XGBoost classifier baseline)")


# ==============================
# 4) Predict + Probabilities
# ==============================
pred_train_cls = xgboost_cls_base.predict(X_train_xgb_cls_base)
pred_test_cls  = xgboost_cls_base.predict(X_test_xgb_cls_base)

proba_train = xgboost_cls_base.predict_proba(X_train_xgb_cls_base)
proba_test  = xgboost_cls_base.predict_proba(X_test_xgb_cls_base)


# ==============================
# 5) Metrics (train + test)
# ==============================
def cls_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

train_metrics = cls_metrics(y_train_xgb_cls_base, pred_train_cls)
test_metrics  = cls_metrics(y_test_xgb_cls_base,  pred_test_cls)

metrics_table = pd.DataFrame({
    "Metric": list(train_metrics.keys()),
    "Train":  list(train_metrics.values()),
    "Test":   list(test_metrics.values()),
    "Better if": ["Higher"] * len(train_metrics)
})

print("\nClassification metrics (baseline)")
display(metrics_table)


# ==============================
# 6) ROC-AUC (multi-class OvR)
# ==============================
# ROC-AUC needs probabilities + one-vs-rest strategy
roc_auc_train = float(roc_auc_score(y_train_xgb_cls_base, proba_train, multi_class="ovr"))
roc_auc_test  = float(roc_auc_score(y_test_xgb_cls_base,  proba_test,  multi_class="ovr"))

print("\nROC-AUC (OvR)")
print("Train ROC-AUC:", roc_auc_train)
print("Test  ROC-AUC:", roc_auc_test)


# ==============================
# 7) ROC curves (OvR) by class
# ==============================
# One-vs-rest: curve per class vs all others
plt.figure(figsize=(7, 6))
for c in range(num_classes):
    y_true_bin = (y_test_xgb_cls_base.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    plt.plot(fpr, tpr, label=f"Classe {c}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 8) Confusion matrix (test)
# ==============================
cm = confusion_matrix(y_test_xgb_cls_base, pred_test_cls)
cm_df = pd.DataFrame(cm, index=[f"Réel_{i}" for i in range(num_classes)], columns=[f"Prédit_{i}" for i in range(num_classes)])

print("\nMatrice de confusion (test)")
display(cm_df)


# ==============================
# 9) Example prediction (same encoded features as regression)
# ==============================
# If you already have a special encoded example, use it. Else use X_new_encoded_no_te.
if "X_new_xgboost_cls_no_te" in globals():
    X_one_cls = X_new_xgboost_cls_no_te.copy()
else:
    X_one_cls = X_new_encoded_no_te.copy()

X_one_cls = X_one_cls.reindex(columns=X_train_xgb_cls_base.columns, fill_value=0).astype(float)

pred_example_class = int(xgboost_cls_base.predict(X_one_cls)[0])
pred_example_proba = xgboost_cls_base.predict_proba(X_one_cls)[0]

print("\nExample classification prediction")
print("Predicted class id:", pred_example_class)
print("Probabilities:", pred_example_proba)


# ==============================
# 10) SHAP global + SHAP local (example)
# ==============================
# Note: for multi-class, SHAP returns a list/array per class.
explainer_cls = shap.TreeExplainer(xgboost_cls_base)

# Global SHAP on test sample
X_shap = X_test_xgb_cls_base.sample(min(300, len(X_test_xgb_cls_base)), random_state=42)

shap_values_global = explainer_cls.shap_values(X_shap)

# Global summary for the predicted class (or choose a class index)
class_for_global = pred_example_class  # you can change this
print("\nSHAP global (summary) — shown for class:", class_for_global)
shap.summary_plot(shap_values_global[class_for_global], X_shap, show=True)

print("\nSHAP global (bar) — shown for class:", class_for_global)
shap.summary_plot(shap_values_global[class_for_global], X_shap, plot_type="bar", show=True)

# Local SHAP for the example row (for its predicted class)
shap_values_one = explainer_cls.shap_values(X_one_cls)

print("\nSHAP local (waterfall) — class:", pred_example_class)
base_value = explainer_cls.expected_value[pred_example_class] if isinstance(explainer_cls.expected_value, (list, np.ndarray)) else explainer_cls.expected_value

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[pred_example_class][0],
        base_values=base_value,
        data=X_one_cls.iloc[0],
        feature_names=X_one_cls.columns
    ),
    max_display=20
)
plt.show()

```
