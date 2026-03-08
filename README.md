```
# ============================================================
# CATBOOST CLASSIFICATION — BASELINE (NO target weights)
# Same style/process as before:
# - Uses your prepared CatBoost classification data:
#     X_train_cat_cls_no_te, X_test_cat_cls_no_te
#     y_train_cat_cls_no_te, y_test_cat_cls_no_te
#     cat_cols_cb
#     class_names_fr (dict: id -> label)
# - Train baseline CatBoostClassifier
# - Metrics (train/test): accuracy, macro-F1, weighted-F1, precision, recall
# - Confusion matrix (with French class names)
# - ROC curves OvR (multi-class)
# - SHAP global + SHAP on ONE example row
# - Save model to: models/catboost/classification/<model_name>/
#
# Requirements:
#   pip install catboost shap scikit-learn
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import shap


# ==============================
# 0) Inputs + checks
# ==============================
assert "X_train_cat_cls_no_te" in globals() and "X_test_cat_cls_no_te" in globals()
assert "y_train_cat_cls_no_te" in globals() and "y_test_cat_cls_no_te" in globals()
assert "cat_cols_cb" in globals()
assert "class_names_fr" in globals()

X_train_cat_cls_base = X_train_cat_cls_no_te.copy()
X_test_cat_cls_base  = X_test_cat_cls_no_te.copy()

y_train_cat_cls_base = pd.to_numeric(pd.Series(y_train_cat_cls_no_te), errors="coerce").astype(int).values
y_test_cat_cls_base  = pd.to_numeric(pd.Series(y_test_cat_cls_no_te),  errors="coerce").astype(int).values

classes_sorted = sorted(np.unique(y_train_cat_cls_base))
num_classes = len(classes_sorted)

print("Train:", X_train_cat_cls_base.shape, y_train_cat_cls_base.shape)
print("Test :", X_test_cat_cls_base.shape,  y_test_cat_cls_base.shape)
print("Classes:", classes_sorted)
print("Cat cols:", len(cat_cols_cb))

# Label list in the right order for plots/tables
class_labels_fr = [class_names_fr[c] for c in classes_sorted]


# ==============================
# 1) Build Pools
# ==============================
train_pool_cat_cls_base = Pool(
    X_train_cat_cls_base,
    y_train_cat_cls_base,
    cat_features=cat_cols_cb
)

test_pool_cat_cls_base = Pool(
    X_test_cat_cls_base,
    y_test_cat_cls_base,
    cat_features=cat_cols_cb
)


# ==============================
# 2) Hyperparameters (baseline)
# ==============================
cat_params_cls_base = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 1200,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "subsample": 0.9,
    "colsample_bylevel": 0.9,
    "random_seed": 42,
    "verbose": 200,
    "allow_writing_files": False
}

model_name = "catboost_cls_baseline_no_te_v1"


# ==============================
# 3) Train baseline model
# ==============================
cat_cls_baseline = CatBoostClassifier(**cat_params_cls_base)

cat_cls_baseline.fit(
    train_pool_cat_cls_base,
    eval_set=test_pool_cat_cls_base,
    use_best_model=True,
    early_stopping_rounds=150
)

print("Model trained:", model_name)


# ==============================
# 4) Predict (train/test)
# ==============================
pred_train_cls = cat_cls_baseline.predict(X_train_cat_cls_base).astype(int).reshape(-1)
pred_test_cls  = cat_cls_baseline.predict(X_test_cat_cls_base).astype(int).reshape(-1)

proba_train = cat_cls_baseline.predict_proba(X_train_cat_cls_base)
proba_test  = cat_cls_baseline.predict_proba(X_test_cat_cls_base)


# ==============================
# 5) Metrics (train/test)
# ==============================
def metrics_classification(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

m_train = metrics_classification(y_train_cat_cls_base, pred_train_cls)
m_test  = metrics_classification(y_test_cat_cls_base,  pred_test_cls)

metrics_df = pd.DataFrame([
    {"Jeu": "Train", **m_train},
    {"Jeu": "Test",  **m_test},
])

print("\n=== Métriques (baseline CatBoost classification) ===")
display(metrics_df)

print("\n=== Rapport détaillé (test) ===")
print(classification_report(
    y_test_cat_cls_base,
    pred_test_cls,
    labels=classes_sorted,
    target_names=class_labels_fr,
    zero_division=0
))


# ==============================
# 6) Confusion matrix (test) with French labels
# ==============================
cm = confusion_matrix(y_test_cat_cls_base, pred_test_cls, labels=classes_sorted)
cm_df = pd.DataFrame(cm, index=[f"Réel: {n}" for n in class_labels_fr], columns=[f"Prédit: {n}" for n in class_labels_fr])

print("\n=== Matrice de confusion (test) ===")
display(cm_df)


# ==============================
# 7) ROC curves OvR (multi-class)
# ==============================
# Binarize y for OvR ROC
y_test_bin = label_binarize(y_test_cat_cls_base, classes=classes_sorted)

plt.figure(figsize=(7, 6))
for i, c in enumerate(classes_sorted):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_test[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_fr[c]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 8) SHAP (global + example)
# ==============================
# Pick ONE class to visualize globally (important for multiclass)
# If you want a different one: set class_id_for_shap = 0..4
class_id_for_shap = int(pred_test_cls[0])  # example: use predicted class of first test row

sample_size = min(300, len(X_test_cat_cls_base))
X_shap = X_test_cat_cls_base.sample(sample_size, random_state=42).copy()

explainer_cls = shap.TreeExplainer(cat_cls_baseline)

# For multiclass, shap_values is usually: list[num_classes] of arrays (n_samples, n_features)
shap_values_all = explainer_cls.shap_values(X_shap)

print("\nSHAP global (classe):", class_id_for_shap, "->", class_names_fr[class_id_for_shap])

# Robust handling if shap returns list vs array
if isinstance(shap_values_all, list):
    shap_vals_for_class = shap_values_all[class_id_for_shap]
else:
    # sometimes shape can be (n_samples, n_features, n_classes)
    shap_vals_for_class = shap_values_all[:, :, class_id_for_shap]

shap.summary_plot(shap_vals_for_class, X_shap, show=True)
shap.summary_plot(shap_vals_for_class, X_shap, plot_type="bar", show=True)


# ---- SHAP on ONE example row
# You can use your own prepared example dataframe for classification if you have it.
# If you already created something like X_example_cat_cls_no_te, set it here.
X_one = X_test_cat_cls_base.iloc[[0]].copy()

pred_one_class = int(cat_cls_baseline.predict(X_one)[0])
proba_one = cat_cls_baseline.predict_proba(X_one)[0]

print("\n=== Exemple (1 ligne) ===")
print("Classe prédite:", pred_one_class, "->", class_names_fr[pred_one_class])
print("Probabilités par classe:")
for c in classes_sorted:
    print(f" - {class_names_fr[c]}: {proba_one[c]:.4f}")

shap_one_all = explainer_cls.shap_values(X_one)

if isinstance(shap_one_all, list):
    shap_one = shap_one_all[pred_one_class]
    base_val = explainer_cls.expected_value[pred_one_class] if isinstance(explainer_cls.expected_value, (list, np.ndarray)) else explainer_cls.expected_value
else:
    shap_one = shap_one_all[:, :, pred_one_class]
    base_val = explainer_cls.expected_value[pred_one_class] if isinstance(explainer_cls.expected_value, (list, np.ndarray)) else explainer_cls.expected_value

# Waterfall for the predicted class
shap.plots.waterfall(
    shap.Explanation(
        values=shap_one[0],
        base_values=base_val,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()


# ==============================
# 9) Save model + params + metrics
# ==============================
save_dir = os.path.join("models", "catboost", "classification", model_name)
os.makedirs(save_dir, exist_ok=True)

cat_cls_baseline.save_model(os.path.join(save_dir, "model.cbm"))
metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(cat_params_cls_base, f, indent=2)

with open(os.path.join(save_dir, "classes_fr.json"), "w") as f:
    json.dump({int(k): v for k, v in class_names_fr.items()}, f, indent=2, ensure_ascii=False)

print("Saved to:", save_dir)

```
