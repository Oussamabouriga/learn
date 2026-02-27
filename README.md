```
# ============================================================
# GATED ENSEMBLE (Mixture of Experts)
# 1) XGBoost CLASSIFIER -> predicts 5 satisfaction classes
# 2) 5 XGBoost REGRESSORS -> each predicts the note INSIDE its class interval
# 3) Final prediction is CLIPPED to the class interval (guaranteed)
#
# + Metrics (classification + regression + Accuracy@±tol)
# + SHAP global + SHAP local for:
#     - classifier (global + example)
#     - chosen regressor (global + example)
# + Predict your example row
#
# IMPORTANT:
# - This code assumes you ALREADY HAVE:
#   X_train_encoded, X_test_encoded, y_train, y_test
#   (same encoded feature columns for train/test)
#
# Requirements:
#   pip install xgboost shap scikit-learn pandas numpy matplotlib
# ============================================================

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# 0) Safety checks
# -----------------------------
print("X_train_encoded:", X_train_encoded.shape)
print("X_test_encoded :", X_test_encoded.shape)
print("y_train:", y_train.shape, " y_test:", y_test.shape)

# Ensure numeric target
y_train = pd.to_numeric(pd.Series(y_train), errors="coerce")
y_test  = pd.to_numeric(pd.Series(y_test), errors="coerce")

# Drop missing target if any
train_mask = y_train.notna()
test_mask  = y_test.notna()
X_train_encoded = X_train_encoded.loc[train_mask].copy()
y_train = y_train.loc[train_mask].astype(float).copy()
X_test_encoded = X_test_encoded.loc[test_mask].copy()
y_test = y_test.loc[test_mask].astype(float).copy()

# Ensure columns are strings and safe for XGBoost
X_train_encoded.columns = X_train_encoded.columns.astype(str)
X_test_encoded.columns  = X_test_encoded.columns.astype(str)

# If you previously had feature-name issues, uncomment this:
# X_train_encoded.columns = X_train_encoded.columns.str.replace(r"[\[\]<>]", "_", regex=True)
# X_test_encoded.columns  = X_test_encoded.columns.str.replace(r"[\[\]<>]", "_", regex=True)

# -----------------------------
# 1) Define your 5 classes + ranges
# -----------------------------
CLASS_RANGES = {
    0: (0.0, 2.0),   # extrêmement mauvais
    1: (3.0, 6.0),   # mauvais
    2: (7.0, 8.0),   # neutre
    3: (9.0, 9.0),   # bien
    4: (10.0, 10.0)  # très bien
}

CLASS_NAMES_FR = {
    0: "extrêmement mauvais (0-2)",
    1: "mauvais (3-6)",
    2: "neutre (7-8)",
    3: "bien (9)",
    4: "très bien (10)"
}

def y_to_class(y):
    # y is numeric note 0..10
    if y <= 2: return 0
    if y <= 6: return 1
    if y <= 8: return 2
    if y == 9: return 3
    return 4

# -----------------------------
# 2) Create class labels for train/test
# -----------------------------
y_class_train = y_train.apply(y_to_class).astype(int)
y_class_test  = y_test.apply(y_to_class).astype(int)

print("\nClass distribution (train):")
print(y_class_train.value_counts().sort_index())
print("\nClass distribution (test):")
print(y_class_test.value_counts().sort_index())

# -----------------------------
# 3) Train the GATE classifier (5 classes)
# -----------------------------
clf_gate = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

clf_gate.fit(X_train_encoded, y_class_train)
print("\n✅ Classifier trained")

# -----------------------------
# 4) Train 5 regressors (experts), one per class
#    Each regressor learns ONLY within its class interval
# -----------------------------
reg_experts = {}

for k, (lo, hi) in CLASS_RANGES.items():
    idx = (y_class_train.values == k)

    Xk = X_train_encoded.loc[idx].copy()
    yk = y_train.loc[idx].astype(float).copy()

    # clip targets inside bucket (safety)
    yk = np.clip(yk, lo, hi)

    # Handle degenerate buckets (e.g., class 4 always 10) with a simple constant regressor fallback
    # but we can still train XGBRegressor; it will just learn constant.
    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    reg.fit(Xk, yk)
    reg_experts[k] = reg

print("✅ Regressors trained:", list(reg_experts.keys()))

# ============================================================
# 5) PREDICT with the GATED system (guaranteed inside interval)
# ============================================================
def gated_predict(X_encoded_df, tol=1.0):
    """
    Returns:
      - pred_class (int)
      - pred_note_raw (float)  (regressor output before clip)
      - pred_note (float)      (clipped to class interval)
      - pred_range (tuple)
      - proba (np.array shape (n,5))
    """
    proba = clf_gate.predict_proba(X_encoded_df)
    pred_class = np.argmax(proba, axis=1).astype(int)

    pred_raw = np.zeros(len(X_encoded_df), dtype=float)
    pred_clipped = np.zeros(len(X_encoded_df), dtype=float)
    pred_ranges = []

    # predict per class for efficiency
    for k in range(5):
        mask = (pred_class == k)
        if not np.any(mask):
            continue
        lo, hi = CLASS_RANGES[k]
        pk = reg_experts[k].predict(X_encoded_df.loc[mask])
        pred_raw[mask] = pk
        pred_clipped[mask] = np.clip(pk, lo, hi)
        # store ranges
    pred_ranges = [CLASS_RANGES[int(k)] for k in pred_class]

    return pred_class, pred_raw, pred_clipped, pred_ranges, proba

# ============================================================
# 6) Evaluate: classifier metrics + regression metrics + Accuracy@±tol
# ============================================================
tol = 1.0  # change to 0.5 if you want

# --- Train
pred_class_train, pred_raw_train, pred_note_train, _, _ = gated_predict(X_train_encoded, tol=tol)
# --- Test
pred_class_test, pred_raw_test, pred_note_test, _, _ = gated_predict(X_test_encoded, tol=tol)

# ---- Classification metrics
acc_cls_train = accuracy_score(y_class_train, pred_class_train)
acc_cls_test  = accuracy_score(y_class_test, pred_class_test)

print("\n=== CLASSIFICATION (GATE) ===")
print(f"Accuracy train: {acc_cls_train:.4f}")
print(f"Accuracy test : {acc_cls_test:.4f}")
print("\nConfusion matrix (test):")
print(confusion_matrix(y_class_test, pred_class_test))

print("\nClassification report (test):")
print(classification_report(
    y_class_test, pred_class_test,
    target_names=[CLASS_NAMES_FR[i] for i in range(5)],
    zero_division=0
))

# ---- Regression metrics (final note)
def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    acc_tol = (np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100.0
    return mae, rmse, r2, acc_tol

mae_tr, rmse_tr, r2_tr, acc_tol_tr = reg_metrics(y_train, pred_note_train)
mae_te, rmse_te, r2_te, acc_tol_te = reg_metrics(y_test, pred_note_test)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}", "ClsAccuracy(5 classes)"],
    "Train":  [mae_tr, rmse_tr, r2_tr, acc_tol_tr, acc_cls_train],
    "Test":   [mae_te, rmse_te, r2_te, acc_tol_te, acc_cls_test],
    "Better if": ["Lower", "Lower", "Higher", "Higher", "Higher"]
})

print("\n=== GATED ENSEMBLE METRICS (final note) ===")
display(metrics_df)

# ============================================================
# 7) Your EXAMPLE row -> encode must already match X_train_encoded columns
#    If you already produced X_new_encoded earlier, use it directly.
# ============================================================

# ---- Put your example here EXACTLY (same as your previous dict)
new_rows = [{
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "tarif": 19.99,
    "Nombre_sisnitre_client": 1,
    "Nombre_sisnitre_accepte_client": 1,
    "Nombre_sisnitre_refuse_client": np.nan,
    "Nombre_sisnitre_sans_suite_client": np.nan,
    "code_postal": 59700,
    "operating_system": "Android",
    "marque": "Google",
    "model": "Pixel 7 Pro ",
    "ancienneté_de_contrat": 509555,
    "garantie": "Dommage",
    "Age": 43,
    "dossier_complet": 1,
    "decision_ai": 0,
    "nombre_prestation_ko": 0,
    "Nbr_ticket_pieces": 0,
    "Nbr_ticket_information": 4,
    "list_prest": "ADVANCED_SWAP",
    "delai_declaration": 279000,
    "delai_de_completude": np.nan,
    "delai_decision": 13090,
    "delai_reparation": 4,
    "delai_indemnisation": 4,
    "montant_indem": np.nan,
    "delai_Sinistre": 602000,
}]

X_new_raw = pd.DataFrame(new_rows)

# NOTE:
# This code assumes you already have the SAME encoding logic used to create X_train_encoded.
# If you used a pipeline: use that pipeline.transform(X_new_raw)
# If you used manual encoding artifacts: apply them here and then align columns.
#
# If you ALREADY HAVE X_new_encoded from your notebook, SKIP this block and set X_new_encoded = your variable.

# -----------------------------
# (A) Minimal alignment-only fallback:
# If your X_train_encoded is already numeric encoded, we cannot "guess" the encoding here.
# So we require YOU to provide X_new_encoded OR you must reuse your encoding artifacts.
# -----------------------------
# If you have an encoded version already, comment the next line and set X_new_encoded manually:
# X_new_encoded = ...

# >>> If you used: X_new_encoded = encode_new_rows_with_artifacts(...)
# >>> put it here.

# For safety: create a placeholder to remind you:
if "X_new_encoded" not in globals():
    raise ValueError(
        "You must provide X_new_encoded (the encoded version of your example row) "
        "using the SAME encoding as X_train_encoded."
    )

# Ensure same columns order
X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# ============================================================
# 8) Predict on the example row (gated)
# ============================================================
pred_class_ex, pred_raw_ex, pred_note_ex, pred_ranges_ex, proba_ex = gated_predict(X_new_encoded, tol=tol)

k_ex = int(pred_class_ex[0])
lo_ex, hi_ex = pred_ranges_ex[0]
print("\n=== EXAMPLE PREDICTION (GATED) ===")
print("Predicted class:", k_ex, "-", CLASS_NAMES_FR[k_ex])
print("Class range:", (lo_ex, hi_ex))
print("Classifier proba:", np.round(proba_ex[0], 4))
print("Pred raw (regressor):", float(pred_raw_ex[0]))
print("Pred clipped (final):", float(pred_note_ex[0]))

# ============================================================
# 9) SHAP — Classifier: global + local (example)
# ============================================================
# SHAP for XGBoost classifier is multi-class:
# shap_values_clf can be list of arrays (one per class) OR array with extra dimension depending on SHAP version.

sample_size = min(400, len(X_test_encoded))
X_shap = X_test_encoded.sample(sample_size, random_state=42).copy()

print("\n=== SHAP (Classifier) — Global importance ===")
explainer_clf = shap.TreeExplainer(clf_gate)
shap_values_clf = explainer_clf.shap_values(X_shap)

# --- Global importance (bar)
plt.figure()
try:
    # Newer SHAP sometimes returns (n, features, classes)
    if isinstance(shap_values_clf, np.ndarray) and shap_values_clf.ndim == 3:
        # Use mean(|shap|) across samples and classes
        mean_abs = np.abs(shap_values_clf).mean(axis=(0, 2))
        global_clf = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        display(global_clf.head(20))
        plt.barh(global_clf.head(20).iloc[::-1]["feature"], global_clf.head(20).iloc[::-1]["mean_abs_shap"])
        plt.title("SHAP (Classifier) - Importance globale (Top 20)")
        plt.tight_layout()
        plt.show()

    else:
        # Older SHAP often returns list[class] of arrays (n, features)
        # Pick predicted class for global plot (more meaningful)
        shap_values_for_class = shap_values_clf[k_ex] if isinstance(shap_values_clf, list) else shap_values_clf
        shap.summary_plot(shap_values_for_class, X_shap, plot_type="bar", show=True)
except Exception as e:
    print("[WARN] SHAP classifier global plot failed:", e)

# --- Local explanation (example)
print("\n=== SHAP (Classifier) — Local explanation (example) ===")
try:
    shap_values_ex_clf = explainer_clf.shap_values(X_new_encoded)

    # Handle formats
    if isinstance(shap_values_ex_clf, np.ndarray) and shap_values_ex_clf.ndim == 3:
        # (n, features, classes) -> take the predicted class
        sv = shap_values_ex_clf[0, :, k_ex]
        base = explainer_clf.expected_value[k_ex] if isinstance(explainer_clf.expected_value, (list, np.ndarray)) else explainer_clf.expected_value
    elif isinstance(shap_values_ex_clf, list):
        sv = shap_values_ex_clf[k_ex][0]
        base = explainer_clf.expected_value[k_ex]
    else:
        sv = shap_values_ex_clf[0]
        base = explainer_clf.expected_value

    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_new_encoded.iloc[0],
        feature_names=X_new_encoded.columns
    )
    shap.plots.waterfall(exp, show=True)
except Exception as e:
    print("[WARN] SHAP classifier local plot failed:", e)

# ============================================================
# 10) SHAP — Regressor expert (chosen class): global + local (example)
# ============================================================
reg_chosen = reg_experts[k_ex]

print("\n=== SHAP (Regressor expert) — Global importance (chosen class) ===")
explainer_reg = shap.TreeExplainer(reg_chosen)

# Use a subset of test rows that were predicted as this class (so SHAP is meaningful)
mask_test_k = (pred_class_test == k_ex)
X_test_k = X_test_encoded.loc[mask_test_k].copy()

if len(X_test_k) < 50:
    # fallback: use generic test sample
    X_test_k = X_test_encoded.sample(min(200, len(X_test_encoded)), random_state=42).copy()
else:
    X_test_k = X_test_k.sample(min(400, len(X_test_k)), random_state=42).copy()

try:
    shap_values_reg = explainer_reg.shap_values(X_test_k)
    shap.summary_plot(shap_values_reg, X_test_k, plot_type="bar", show=True)
except Exception as e:
    print("[WARN] SHAP regressor global plot failed:", e)

print("\n=== SHAP (Regressor expert) — Local explanation (example) ===")
try:
    shap_values_ex_reg = explainer_reg.shap_values(X_new_encoded)
    # shap_values_ex_reg shape: (n, features)
    sv = shap_values_ex_reg[0]
    base = explainer_reg.expected_value
    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_new_encoded.iloc[0],
        feature_names=X_new_encoded.columns
    )
    shap.plots.waterfall(exp, show=True)
except Exception as e:
    print("[WARN] SHAP regressor local plot failed:", e)

print("\n✅ Done.")

```
