```
# ============================================================
# CATBOOST CLASSIFICATION — RANDOM SEARCH (SANS ERREUR)
# Fix: bootstrap_type=Bernoulli (subsample compatible)
# + multi-metrics CV (Accuracy, F1 macro, F1 weighted)
# + refit final avec early stopping
# + ROC + Confusion matrix
# + SHAP global + SHAP local (exemple)
# ============================================================

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)

import matplotlib.pyplot as plt
import shap


# ==============================
# 1) Data (doit déjà exister)
# ==============================
X_train_cat_cls = X_train_cat_cls_no_te.copy()
X_test_cat_cls  = X_test_cat_cls_no_te.copy()

y_train_cat_cls = pd.to_numeric(y_train_cat_cls_no_te, errors="coerce").astype(int)
y_test_cat_cls  = pd.to_numeric(y_test_cat_cls_no_te,  errors="coerce").astype(int)

num_classes = int(len(np.unique(y_train_cat_cls)))
print("Train:", X_train_cat_cls.shape, y_train_cat_cls.shape)
print("Test :", X_test_cat_cls.shape,  y_test_cat_cls.shape)
print("Classes:", sorted(np.unique(y_train_cat_cls)))


# ==============================
# 2) CLEANUP CatBoost (CRITICAL)
# - remplace pd.NA -> np.nan
# - force toutes les colonnes catégorielles en str
# - remplace les "nan"/"<NA>"/None par "__MISSING__"
# ==============================
X_train_cat_cls = X_train_cat_cls.astype(object).where(pd.notna(X_train_cat_cls), np.nan)
X_test_cat_cls  = X_test_cat_cls.astype(object).where(pd.notna(X_test_cat_cls),  np.nan)

for c in cat_cols_cb:
    if c in X_train_cat_cls.columns:
        X_train_cat_cls[c] = X_train_cat_cls[c].astype(str)
        X_test_cat_cls[c]  = X_test_cat_cls[c].astype(str)

        X_train_cat_cls.loc[X_train_cat_cls[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
        X_test_cat_cls.loc[X_test_cat_cls[c].isin(["nan", "None", "<NA>"]), c]  = "__MISSING__"

# force non-cat numeric
num_cols = [c for c in X_train_cat_cls.columns if c not in cat_cols_cb]
for c in num_cols:
    X_train_cat_cls[c] = pd.to_numeric(X_train_cat_cls[c], errors="coerce")
    X_test_cat_cls[c]  = pd.to_numeric(X_test_cat_cls[c],  errors="coerce")


# ==============================
# 3) Base estimator (pas fitted)
# IMPORTANT: bootstrap_type='Bernoulli' pour autoriser subsample
# ==============================
cat_base = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    allow_writing_files=False,
    verbose=0,
    bootstrap_type="Bernoulli"
)

# ==============================
# 4) Random Search space (COMPUTE FRIENDLY)
# - PAS de Bayesian ici
# ==============================
param_distributions = {
    "iterations": [400, 700, 1000, 1400],
    "learning_rate": [0.02, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 8.0, 12.0],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
    "colsample_bylevel": [0.7, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],  # OK car bootstrap=Bernoulli
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "acc": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted"
}

random_search = RandomizedSearchCV(
    estimator=cat_base,
    param_distributions=param_distributions,
    n_iter=25,                 # ajuste 10..80 selon budget
    scoring=scoring,
    refit="f1_weighted",       # on sélectionne le "meilleur" selon F1 weighted
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)

print("\n--- Random Search: lancement ---")
random_search.fit(X_train_cat_cls, y_train_cat_cls, cat_features=cat_cols_cb)

best_params = random_search.best_params_
best_cv = random_search.best_score_

print("\n--- Random Search terminé ---")
print("Best CV (F1_weighted):", best_cv)
print("Best params:", best_params)


# ==============================
# 5) Refit FINAL propre (avec early stopping)
# (on NE fait PAS set_params sur un modèle fitted)
# ==============================
train_pool = Pool(X_train_cat_cls, y_train_cat_cls, cat_features=cat_cols_cb)
test_pool  = Pool(X_test_cat_cls,  y_test_cat_cls,  cat_features=cat_cols_cb)

cat_cls_random_best = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    allow_writing_files=False,
    verbose=200,
    bootstrap_type="Bernoulli",   # force stable
    **best_params
)

cat_cls_random_best.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=150
)

print("\nModel trained: cat_cls_random_best")


# ==============================
# 6) Metrics (train/test)
# ==============================
pred_train = cat_cls_random_best.predict(X_train_cat_cls).astype(int).reshape(-1)
pred_test  = cat_cls_random_best.predict(X_test_cat_cls).astype(int).reshape(-1)

proba_train = cat_cls_random_best.predict_proba(X_train_cat_cls)
proba_test  = cat_cls_random_best.predict_proba(X_test_cat_cls)

acc_train = accuracy_score(y_train_cat_cls, pred_train)
acc_test  = accuracy_score(y_test_cat_cls,  pred_test)

f1m_train = f1_score(y_train_cat_cls, pred_train, average="macro")
f1m_test  = f1_score(y_test_cat_cls,  pred_test,  average="macro")

f1w_train = f1_score(y_train_cat_cls, pred_train, average="weighted")
f1w_test  = f1_score(y_test_cat_cls,  pred_test,  average="weighted")

print("\n=== Résultats (Classification) ===")
print("Accuracy train:", acc_train)
print("Accuracy test :", acc_test)
print("F1 macro train:", f1m_train)
print("F1 macro test :", f1m_test)
print("F1 weighted train:", f1w_train)
print("F1 weighted test :", f1w_test)

print("\nRapport de classification (test):")
print(classification_report(y_test_cat_cls, pred_test, digits=4))


# ==============================
# 7) ROC curves (OvR) + noms classes
# ==============================
plt.figure(figsize=(7, 6))
for c in range(num_classes):
    y_true_bin = (y_test_cat_cls.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    auc_c = auc(fpr, tpr)

    label_name = class_names_fr.get(c, f"Classe {c}") if isinstance(class_names_fr, dict) else f"Classe {c}"
    plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc_c:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 8) Confusion matrix (avec noms)
# ==============================
cm = confusion_matrix(y_test_cat_cls, pred_test)

idx_names = [class_names_fr.get(i, f"Réel_{i}") for i in range(num_classes)] if isinstance(class_names_fr, dict) else [f"Réel_{i}" for i in range(num_classes)]
col_names = [class_names_fr.get(i, f"Prédit_{i}") for i in range(num_classes)] if isinstance(class_names_fr, dict) else [f"Prédit_{i}" for i in range(num_classes)]

cm_df = pd.DataFrame(cm, index=idx_names, columns=col_names)

print("\nMatrice de confusion (test)")
display(cm_df)


# ==============================
# 9) SHAP global + SHAP local (exemple)
# ==============================
# --- dataset SHAP (petit échantillon)
sample_size = min(300, len(X_test_cat_cls))
X_shap = X_test_cat_cls.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_cls_random_best)
shap_values = explainer.shap_values(X_shap)  # multiclass -> list[num_classes] of arrays

# --- choisir une classe pour affichage global (ex: classe 4 ou celle dominante)
class_for_global = 4 if num_classes > 4 else 0
print("\nSHAP global — classe:", class_for_global, "|", class_names_fr.get(class_for_global, class_for_global) if isinstance(class_names_fr, dict) else class_for_global)

sv = shap_values[class_for_global]

# Fix possible mismatch: some versions add an extra "bias" column
if sv.shape[1] == X_shap.shape[1] + 1:
    sv = sv[:, :-1]

shap.summary_plot(sv, X_shap, show=True)
shap.summary_plot(sv, X_shap, plot_type="bar", show=True)


# --- SHAP sur l'exemple (si tu as déjà préparé test_row_cat_cls_no_te)
#     (sinon commente ce bloc)
X_one = pd.DataFrame(test_row_cat_cls_no_te).copy()

# align cols
for c in X_train_cat_cls.columns:
    if c not in X_one.columns:
        X_one[c] = np.nan
X_one = X_one[X_train_cat_cls.columns].copy()

# same cleanup
X_one = X_one.astype(object).where(pd.notna(X_one), np.nan)
for c in cat_cols_cb:
    if c in X_one.columns:
        X_one[c] = X_one[c].astype(str)
        X_one.loc[X_one[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
for c in num_cols:
    X_one[c] = pd.to_numeric(X_one[c], errors="coerce")

pred_example_class = int(cat_cls_random_best.predict(X_one).reshape(-1)[0])
print("\n=== EXEMPLE ===")
print("Classe prédite:", pred_example_class, "|", class_names_fr.get(pred_example_class, pred_example_class) if isinstance(class_names_fr, dict) else pred_example_class)

shap_one = explainer.shap_values(X_one)  # list
sv_one = shap_one[pred_example_class]

if sv_one.shape[1] == X_one.shape[1] + 1:
    sv_one = sv_one[:, :-1]

# waterfall (top 20 features)
base_val = explainer.expected_value[pred_example_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

shap.plots.waterfall(
    shap.Explanation(
        values=sv_one[0],
        base_values=base_val,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()

```
