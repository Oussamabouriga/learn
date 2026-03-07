```
# ============================================================
# XGBoost Classification — RANDOM SEARCH (no target encoding)
# Uses SAME DATA you provided:
#   X_train_xgb_cls_no_te, X_test_xgb_cls_no_te
#   y_train_xgb_cls_no_te, y_test_xgb_cls_no_te
#
# Includes:
# - RandomizedSearchCV (with CV)
# - Train best model
# - Metrics (train/test): accuracy, f1 (macro/weighted), precision/recall, logloss
# - ROC curves OvR with class names
# - Confusion matrix with class names
# - SHAP global + SHAP local (example row)
# - Save model to: models/xgboost/classification/<model_name>/
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, log_loss
)

import joblib

# SHAP
import shap


# ==============================
# 1) Data (as you requested)
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
# 2) Class names (edit if you want)
#    Must match your class indices 0..4
# ==============================
class_names = [
    "Extrêmement mauvais (0–2)",
    "Mauvais (3–6)",
    "Neutre (7–8)",
    "Bien (9)",
    "Très bien (10)",
]
# safety: if different number of classes
if len(class_names) != num_classes:
    class_names = [f"Classe {i}" for i in range(num_classes)]


# ==============================
# 3) Base model (fixed parts)
# ==============================
xgb_cls_base = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)


# ==============================
# 4) Random Search space (compute-friendly)
# ==============================
param_distributions = {
    "n_estimators": [200, 400, 600, 900, 1200],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


# ==============================
# 5) CV setup
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scoring: choose one primary metric for the search
# (You can change to "f1_macro" if you prefer)
scoring_main = "f1_macro"


# ==============================
# 6) RandomizedSearchCV
# ==============================
random_search = RandomizedSearchCV(
    estimator=xgb_cls_base,
    param_distributions=param_distributions,
    n_iter=40,                 # adjust 20..80 depending on compute
    scoring=scoring_main,
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

random_search.fit(X_train_xgb_cls_base, y_train_xgb_cls_base)

best_model_xgb_cls_random = random_search.best_estimator_
best_params_xgb_cls_random = random_search.best_params_
best_cv_score_xgb_cls_random = random_search.best_score_

print("\nBest CV score (", scoring_main, "):", best_cv_score_xgb_cls_random)
print("Best params:", best_params_xgb_cls_random)


# ==============================
# 7) Predictions + probabilities
# ==============================
pred_train_cls = best_model_xgb_cls_random.predict(X_train_xgb_cls_base)
pred_test_cls  = best_model_xgb_cls_random.predict(X_test_xgb_cls_base)

proba_train = best_model_xgb_cls_random.predict_proba(X_train_xgb_cls_base)
proba_test  = best_model_xgb_cls_random.predict_proba(X_test_xgb_cls_base)


# ==============================
# 8) Metrics (train/test)
# ==============================
def classification_metrics_table(y_true, y_pred, y_proba, split_name="Train"):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    ll = log_loss(y_true, y_proba, labels=list(range(num_classes)))
    return {
        "Split": split_name,
        "Accuracy": acc,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "Precision_macro": prec_macro,
        "Recall_macro": rec_macro,
        "LogLoss": ll
    }

m_train = classification_metrics_table(y_train_xgb_cls_base, pred_train_cls, proba_train, "Train")
m_test  = classification_metrics_table(y_test_xgb_cls_base,  pred_test_cls,  proba_test,  "Test")

metrics_df = pd.DataFrame([m_train, m_test])
print("\nMetrics (classification) — Random Search best")
display(metrics_df)

print("\nRapport de classification (test):")
print(classification_report(
    y_test_xgb_cls_base, pred_test_cls,
    target_names=class_names, zero_division=0
))


# ==============================
# 9) ROC curves OvR (test) with class names
# ==============================
plt.figure(figsize=(7, 6))
for c in range(num_classes):
    y_true_bin = (y_test_xgb_cls_base.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test (Random Search)")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 10) Confusion matrix (test) with class names
# ==============================
cm = confusion_matrix(y_test_xgb_cls_base, pred_test_cls, labels=list(range(num_classes)))
cm_df = pd.DataFrame(
    cm,
    index=[f"Réel — {name}" for name in class_names],
    columns=[f"Prédit — {name}" for name in class_names],
)

print("\nMatrice de confusion (test)")
display(cm_df)


# ==============================
# 11) SHAP — global + local (example row)
#     NOTE: For multi-class, shap_values can be:
#       - list of arrays [n_classes] each (n_samples, n_features)
#       - or array (n_samples, n_features, n_classes)
#     We'll normalize it safely.
# ==============================
# Sample from test for global SHAP (faster)
sample_size = min(300, len(X_test_xgb_cls_base))
X_shap = X_test_xgb_cls_base.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(best_model_xgb_cls_random)
shap_values = explainer.shap_values(X_shap)

# Normalize to list-of-classes format: shap_values_list[class] -> (n_samples, n_features)
if isinstance(shap_values, list):
    shap_values_list = shap_values
else:
    # could be (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        shap_values_list = [shap_values[:, :, k] for k in range(shap_values.shape[2])]
    else:
        # fallback (binary sometimes returns (n_samples, n_features))
        shap_values_list = [shap_values]

# Pick a class to show global importance (example: predicted class of example or class 0)
class_for_global = 0
print("\nSHAP global (summary) — classe:", class_for_global, "-", class_names[class_for_global])

# If SHAP adds an extra bias column, trim it safely
sv = shap_values_list[class_for_global]
if sv.shape[1] == X_shap.shape[1] + 1:
    sv = sv[:, :-1]

shap.summary_plot(sv, X_shap, show=True)
shap.summary_plot(sv, X_shap, plot_type="bar", show=True)


# ==============================
# 12) SHAP local (example row)
#     Use your prepared example row:
#       X_example_xgb_cls_no_te (must be encoded like X_train_xgb_cls_no_te)
# If you have a different variable name, replace it here.
# ==============================
# ---- choose your example dataframe variable here
# Example must be a DataFrame with same columns as X_train_xgb_cls_base
X_example = None
try:
    X_example = X_example_xgb_cls_no_te.copy().astype(float)  # <-- change if needed
except Exception:
    pass

if X_example is None:
    print("\nNo example row dataframe found. Please set: X_example = <your_example_df>.")
else:
    # Align columns just in case
    X_example = X_example.reindex(columns=X_train_xgb_cls_base.columns, fill_value=0.0)

    proba_example = best_model_xgb_cls_random.predict_proba(X_example)[0]
    pred_example_class = int(np.argmax(proba_example))

    print("\nExample prediction:")
    print("Classe prédite:", pred_example_class, "-", class_names[pred_example_class])
    print("Probabilités par classe:")
    for i, p in enumerate(proba_example):
        print(f"  {i} - {class_names[i]} : {p:.4f}")

    shap_one = explainer.shap_values(X_example)

    # normalize one-row shap to list-of-classes
    if isinstance(shap_one, list):
        shap_one_list = shap_one
    else:
        if shap_one.ndim == 3:
            shap_one_list = [shap_one[:, :, k] for k in range(shap_one.shape[2])]
        else:
            shap_one_list = [shap_one]

    sv_one = shap_one_list[pred_example_class]
    # remove bias column if present
    if sv_one.shape[1] == X_example.shape[1] + 1:
        sv_one = sv_one[:, :-1]

    # Waterfall for predicted class
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[pred_example_class]

    shap.plots.waterfall(
        shap.Explanation(
            values=sv_one[0],
            base_values=base_val,
            data=X_example.iloc[0],
            feature_names=X_example.columns
        ),
        max_display=20
    )
    plt.show()


# ==============================
# 13) Save model + metadata
# ==============================
model_name = "xgb_cls_randomsearch_no_te_v1"
save_dir = os.path.join("models", "xgboost", "classification", model_name)
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_model_xgb_cls_random, os.path.join(save_dir, "model.joblib"))

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params_xgb_cls_random, f, indent=2)

metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)
cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

print("\nModel saved to:", save_dir)

```
