```
# ============================================================
# CATBOOST CLASSIFICATION — RANDOM SEARCH (baseline process)
# - RandomSearchCV on CatBoostClassifier (MultiClass)
# - Train best model
# - Metrics train/test + confusion matrix + ROC OvR
# - SHAP global + SHAP example (1 row)
# - Save to models/catboost/classification/<model_name>/
#
# Uses your existing prepared data:
#   X_train_cat_cls_no_te, X_test_cat_cls_no_te
#   y_train_cat_cls_no_te, y_test_cat_cls_no_te
#   cat_cols_cb
#   class_names_fr (dict: id -> label)
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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

X_train_cat_cls_rs = X_train_cat_cls_no_te.copy()
X_test_cat_cls_rs  = X_test_cat_cls_no_te.copy()

y_train_cat_cls_rs = pd.to_numeric(pd.Series(y_train_cat_cls_no_te), errors="coerce").astype(int).values
y_test_cat_cls_rs  = pd.to_numeric(pd.Series(y_test_cat_cls_no_te),  errors="coerce").astype(int).values

classes_sorted = sorted(np.unique(y_train_cat_cls_rs))
num_classes = len(classes_sorted)
class_labels_fr = [class_names_fr[c] for c in classes_sorted]

print("Train:", X_train_cat_cls_rs.shape, y_train_cat_cls_rs.shape)
print("Test :", X_test_cat_cls_rs.shape,  y_test_cat_cls_rs.shape)
print("Classes:", classes_sorted)


# ==============================
# 1) Pools (for final training only)
# ==============================
train_pool_cat_cls_rs = Pool(X_train_cat_cls_rs, y_train_cat_cls_rs, cat_features=cat_cols_cb)
test_pool_cat_cls_rs  = Pool(X_test_cat_cls_rs,  y_test_cat_cls_rs,  cat_features=cat_cols_cb)


# ==============================
# 2) Base estimator (fixed parts)
# ==============================
# Note: We keep `loss_function=MultiClass` and tune the rest.
cat_cls_base_est = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    allow_writing_files=False,
    verbose=False
)


# ==============================
# 3) Random Search space (compute-friendly)
# ==============================
# Keep it focused: main knobs that matter most.
param_distributions_cat_cls = {
    "iterations": [600, 900, 1200, 1600, 2200],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "depth": [4, 5, 6, 7, 8, 9, 10],
    "l2_leaf_reg": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "random_strength": [0.0, 0.5, 1.0, 1.5, 2.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.7, 0.8, 0.9, 1.0],
    "grow_policy": ["SymmetricTree", "Lossguide"],
}

# CV strategy (classification => stratified)
cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# How many random configs to try (adjust to your compute)
n_iter_random = 30


# ==============================
# 4) RandomizedSearchCV
# ==============================
# scoring: choose ONE primary metric for selection
# (we will still compute many metrics after training best model)
random_search_cat_cls = RandomizedSearchCV(
    estimator=cat_cls_base_est,
    param_distributions=param_distributions_cat_cls,
    n_iter=n_iter_random,
    scoring="f1_macro",
    cv=cv_cls,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

random_search_cat_cls.fit(
    X_train_cat_cls_rs,
    y_train_cat_cls_rs,
    cat_features=cat_cols_cb
)

best_params_cat_cls = random_search_cat_cls.best_params_
best_cv_score_cat_cls = float(random_search_cat_cls.best_score_)

print("\n=== Random Search done ===")
print("Best CV f1_macro:", best_cv_score_cat_cls)
print("Best params:", best_params_cat_cls)


# ==============================
# 5) Train final model with best params (with early stopping)
# ==============================
model_name = "catboost_cls_random_search_no_te_v1"

cat_cls_random_best = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    allow_writing_files=False,
    verbose=200,
    **best_params_cat_cls
)

cat_cls_random_best.fit(
    train_pool_cat_cls_rs,
    eval_set=test_pool_cat_cls_rs,
    use_best_model=True,
    early_stopping_rounds=150
)

print("Final model trained:", model_name)


# ==============================
# 6) Predict + metrics (train/test)
# ==============================
pred_train = cat_cls_random_best.predict(X_train_cat_cls_rs).astype(int).reshape(-1)
pred_test  = cat_cls_random_best.predict(X_test_cat_cls_rs).astype(int).reshape(-1)

proba_train = cat_cls_random_best.predict_proba(X_train_cat_cls_rs)
proba_test  = cat_cls_random_best.predict_proba(X_test_cat_cls_rs)

def metrics_classification(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

m_train = metrics_classification(y_train_cat_cls_rs, pred_train)
m_test  = metrics_classification(y_test_cat_cls_rs,  pred_test)

metrics_df = pd.DataFrame([
    {"Jeu": "Train", **m_train},
    {"Jeu": "Test",  **m_test},
])

print("\n=== Métriques (CatBoost classification — Random Search) ===")
display(metrics_df)

print("\n=== Rapport détaillé (test) ===")
print(classification_report(
    y_test_cat_cls_rs,
    pred_test,
    labels=classes_sorted,
    target_names=class_labels_fr,
    zero_division=0
))


# ==============================
# 7) Confusion matrix (test) with French labels
# ==============================
cm = confusion_matrix(y_test_cat_cls_rs, pred_test, labels=classes_sorted)
cm_df = pd.DataFrame(cm, index=[f"Réel: {n}" for n in class_labels_fr], columns=[f"Prédit: {n}" for n in class_labels_fr])

print("\n=== Matrice de confusion (test) ===")
display(cm_df)


# ==============================
# 8) ROC curves OvR (multi-class)
# ==============================
y_test_bin = label_binarize(y_test_cat_cls_rs, classes=classes_sorted)

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
# 9) SHAP (global + example)
# ==============================
class_id_for_shap = int(pred_test[0])  # show SHAP for the predicted class of first test row
sample_size = min(300, len(X_test_cat_cls_rs))
X_shap = X_test_cat_cls_rs.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_cls_random_best)
shap_values_all = explainer.shap_values(X_shap)

print("\nSHAP global (classe):", class_id_for_shap, "->", class_names_fr[class_id_for_shap])

if isinstance(shap_values_all, list):
    shap_vals_for_class = shap_values_all[class_id_for_shap]
else:
    shap_vals_for_class = shap_values_all[:, :, class_id_for_shap]

shap.summary_plot(shap_vals_for_class, X_shap, show=True)
shap.summary_plot(shap_vals_for_class, X_shap, plot_type="bar", show=True)

# Local SHAP on one row (test[0])
X_one = X_test_cat_cls_rs.iloc[[0]].copy()
pred_one = int(cat_cls_random_best.predict(X_one)[0])
proba_one = cat_cls_random_best.predict_proba(X_one)[0]

print("\n=== Exemple (1 ligne) ===")
print("Classe prédite:", pred_one, "->", class_names_fr[pred_one])
print("Probabilités par classe:")
for c in classes_sorted:
    print(f" - {class_names_fr[c]}: {proba_one[c]:.4f}")

shap_one_all = explainer.shap_values(X_one)

if isinstance(shap_one_all, list):
    shap_one = shap_one_all[pred_one]
    base_val = explainer.expected_value[pred_one] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    shap_one = shap_one_all[:, :, pred_one]
    base_val = explainer.expected_value[pred_one] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

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
# 10) Save model + artifacts
# ==============================
save_dir = os.path.join("models", "catboost", "classification", model_name)
os.makedirs(save_dir, exist_ok=True)

cat_cls_random_best.save_model(os.path.join(save_dir, "model.cbm"))
metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params_cat_cls, f, indent=2)

with open(os.path.join(save_dir, "cv_best_score.json"), "w") as f:
    json.dump({"best_cv_f1_macro": best_cv_score_cat_cls}, f, indent=2)

with open(os.path.join(save_dir, "classes_fr.json"), "w") as f:
    json.dump({int(k): v for k, v in class_names_fr.items()}, f, indent=2, ensure_ascii=False)

print("Saved to:", save_dir)


```
