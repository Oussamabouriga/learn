```
# ============================================================
# XGBoost CLASSIFICATION — RANDOM SEARCH (NO target encoding)
# + Train best model
# + Metrics (accuracy / F1 / precision / recall)
# + ROC curves (OvR) with class names
# + Confusion matrix with class names
# + SHAP global + SHAP local (example row)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)

import shap

# ==============================
# 0) DATA (use your existing variables)
#    Change these 4 names if yours are different.
# ==============================
X_train = X_train_xgb_cls_base.copy()
X_test  = X_test_xgb_cls_base.copy()
y_train = y_train_xgb_cls_base.copy()
y_test  = y_test_xgb_cls_base.copy()

# Example row already prepared/encoded for classification (1 row DataFrame)
# If you don't have it yet, set it to a real encoded row with same columns as X_train.
X_example = X_example_cls.copy()   # must be DataFrame with same columns as X_train

# Make sure feature names are strings and safe for XGBoost
X_train.columns = X_train.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)
X_test.columns  = X_test.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)
X_example = X_example.copy()
X_example.columns = X_example.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)

# Align example to train columns
X_example = X_example.reindex(columns=X_train.columns, fill_value=0)

# ==============================
# 1) Class names (index -> label)
# ==============================
class_names = [
    "Extrêmement mauvais (0–2)",
    "Mauvais (3–6)",
    "Neutre (7–8)",
    "Bien (9)",
    "Très bien (10)",
]
num_classes = len(class_names)

# ==============================
# 2) Baseline model (used by Random Search)
# ==============================
xgb_cls_base = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
)

# ==============================
# 3) Random Search setup (compute-friendly)
# ==============================
param_distributions = {
    "n_estimators": [200, 400, 600, 800, 1000],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_cls_base,
    param_distributions=param_distributions,
    n_iter=40,                 # increase to 60-100 if you have more compute
    scoring="f1_macro",        # good metric for imbalanced multi-class
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True
)

# ==============================
# 4) Fit Random Search
# ==============================
random_search.fit(X_train, y_train)

xgb_cls_random_best = random_search.best_estimator_
print("\nBest params (Random Search):")
print(random_search.best_params_)
print("Best CV score (f1_macro):", random_search.best_score_)

# ==============================
# 5) Predictions + Probabilities
# ==============================
pred_train = xgb_cls_random_best.predict(X_train)
pred_test  = xgb_cls_random_best.predict(X_test)

proba_train = xgb_cls_random_best.predict_proba(X_train)
proba_test  = xgb_cls_random_best.predict_proba(X_test)

# ==============================
# 6) Metrics (Train/Test)
# ==============================
def _cls_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

m_train = _cls_metrics(y_train, pred_train)
m_test  = _cls_metrics(y_test, pred_test)

metrics_df = pd.DataFrame([m_train, m_test], index=["Train", "Test"]).T
print("\n=== Résumé métriques (classification) ===")
display(metrics_df)

print("\n=== Rapport classification (TEST) ===")
print(classification_report(y_test, pred_test, target_names=class_names, digits=3))

# ==============================
# 7) ROC curves (OvR) by class (with names)
# ==============================
plt.figure(figsize=(7, 6))
for c in range(num_classes):
    y_true_bin = (y_test.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================
# 8) Confusion matrix (with names)
# ==============================
cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))
cm_df = pd.DataFrame(
    cm,
    index=[f"Réel — {name}" for name in class_names],
    columns=[f"Prédit — {name}" for name in class_names],
)
print("\nMatrice de confusion (TEST)")
display(cm_df)

# ==============================
# 9) Example prediction
# ==============================
pred_example_class = int(xgb_cls_random_best.predict(X_example)[0])
proba_example = xgb_cls_random_best.predict_proba(X_example)[0]

print("\n=== Exemple (1 ligne) ===")
print("Classe prédite:", pred_example_class, "-", class_names[pred_example_class])
print("Probabilités par classe:")
for i, p in enumerate(proba_example):
    print(f"  {i} - {class_names[i]} : {p:.4f}")

# ==============================
# 10) SHAP (GLOBAL + LOCAL)
#     Fixes the common shape mismatch:
#     - if SHAP returns a bias/offset column => drop last column
# ==============================

# Use a sample for speed
sample_size = min(400, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42)

explainer = shap.TreeExplainer(xgb_cls_random_best)
shap_values = explainer.shap_values(X_shap)

def _get_shap_for_class(shap_values, class_idx):
    # shap_values can be:
    # - list of arrays [class] -> (n_samples, n_features)  OR (n_samples, n_features+1)
    # - array (n_samples, n_features, n_classes) OR (n_samples, n_features+1, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values[:, :, class_idx]

    # Drop possible bias column if present
    if sv.shape[1] == X_shap.shape[1] + 1:
        sv = sv[:, :-1]
    return sv

# Choose which class to show in global plots (example: predicted class for example row)
class_for_global = pred_example_class
sv_global = _get_shap_for_class(shap_values, class_for_global)

print(f"\nSHAP global (classe affichée): {class_for_global} - {class_names[class_for_global]}")
shap.summary_plot(sv_global, X_shap, show=True)
shap.summary_plot(sv_global, X_shap, plot_type="bar", show=True)

# Local SHAP for the example row
X_one = X_example.copy()
sv_one_all = explainer.shap_values(X_one)

# Extract SHAP for the predicted class
if isinstance(sv_one_all, list):
    sv_one = sv_one_all[pred_example_class]
else:
    sv_one = sv_one_all[:, :, pred_example_class]

# Drop bias column if present
if sv_one.shape[1] == X_one.shape[1] + 1:
    sv_one = sv_one[:, :-1]

# Waterfall plot (local explanation)
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[pred_example_class]

shap.plots.waterfall(
    shap.Explanation(
        values=sv_one[0],
        base_values=base_value,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()

```
