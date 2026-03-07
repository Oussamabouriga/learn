```
# ============================================================
# XGBoost CLASSIFICATION — RANDOM SEARCH + CLASS WEIGHTING
# (NO Target Encoding) + Metrics + ROC (OvR) + Confusion Matrix
# + SHAP Global + SHAP for your example row
# + Save model to: models/xgboost/classification/<MODEL_NAME>/
#
# Assumes you ALREADY prepared (encoded) data like before:
#   X_train_encoded_no_te, X_test_encoded_no_te
#   y_train_no_te, y_test_no_te   (continuous satisfaction target)
# And you already have your example row prepared (encoded) OR raw:
#   - If raw: X_new_encoded_no_te (aligned to X_train_encoded_no_te columns)
#   - If already encoded: X_new_xgb_cls_no_te
# ============================================================



class_names = {
    0: "Extrêmement mauvais (0–2)",
    1: "Mauvais (3–6)",
    2: "Neutre (7–8)",
    3: "Bien (9)",
    4: "Très bien (10)"
}
class_labels_in_order = [class_names[i] for i in range(5)]


# ==============================
# 2) X data (encoded, numeric)
#    Fix feature names (XGBoost error with [, ] , <)
# ==============================
X_train_xgb_cls = X_train_encoded_no_te.copy()
X_test_xgb_cls  = X_test_encoded_no_te.copy()

# Force numeric float
X_train_xgb_cls = X_train_xgb_cls.apply(pd.to_numeric, errors="coerce").astype(float)
X_test_xgb_cls  = X_test_xgb_cls.apply(pd.to_numeric, errors="coerce").astype(float)

# Clean feature names to avoid: "feature_names must be string, and may not contain [, ] or <"
def _clean_feature_names(cols):
    cols = cols.astype(str)
    cols = cols.str.replace("[", "(", regex=False)
    cols = cols.str.replace("]", ")", regex=False)
    cols = cols.str.replace("<", "lt_", regex=False)
    cols = cols.str.replace(">", "gt_", regex=False)
    cols = cols.str.replace(",", "_", regex=False)
    return cols

X_train_xgb_cls.columns = _clean_feature_names(X_train_xgb_cls.columns.to_series()).values
X_test_xgb_cls.columns  = _clean_feature_names(X_test_xgb_cls.columns.to_series()).values


# ==============================
# 3) Class weighting -> sample_weight (TRAIN ONLY)
# ==============================
classes = np.unique(y_train_cls.values)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls.values)
class_weight_dict = dict(zip(classes, cw))

sample_weight_train_cls = y_train_cls.map(class_weight_dict).values.astype(float)

print("Class weights:", class_weight_dict)
print("Train shape:", X_train_xgb_cls.shape, "| Test shape:", X_test_xgb_cls.shape)


# ==============================
# 4) Random Search config
# ==============================
random_state = 42

base_clf = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=random_state,
    verbosity=0
)

param_distributions = {
    "n_estimators": [200, 400, 600, 800, 1000],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 8, 12],
    "gamma": [0.0, 0.1, 0.3, 0.7, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

random_search_xgb_cls = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_distributions,
    n_iter=40,                 # adjust 20..80 depending on compute
    scoring="f1_macro",        # good for imbalanced multi-class
    cv=cv,
    verbose=2,
    random_state=random_state,
    n_jobs=-1,
    refit=True
)


# ==============================
# 5) Fit Random Search (pass sample_weight)
# ==============================
random_search_xgb_cls.fit(X_train_xgb_cls, y_train_cls, sample_weight=sample_weight_train_cls)

best_xgb_cls = random_search_xgb_cls.best_estimator_
best_params = random_search_xgb_cls.best_params_
best_cv_score = random_search_xgb_cls.best_score_

print("\nBest CV f1_macro:", best_cv_score)
print("Best params:", best_params)


# ==============================
# 6) Evaluate Train/Test
# ==============================
pred_train = best_xgb_cls.predict(X_train_xgb_cls)
pred_test  = best_xgb_cls.predict(X_test_xgb_cls)

proba_train = best_xgb_cls.predict_proba(X_train_xgb_cls)
proba_test  = best_xgb_cls.predict_proba(X_test_xgb_cls)

acc_train = accuracy_score(y_train_cls, pred_train)
acc_test  = accuracy_score(y_test_cls, pred_test)

bacc_train = balanced_accuracy_score(y_train_cls, pred_train)
bacc_test  = balanced_accuracy_score(y_test_cls, pred_test)

f1_train_macro = f1_score(y_train_cls, pred_train, average="macro")
f1_test_macro  = f1_score(y_test_cls,  pred_test,  average="macro")

prec_test_macro = precision_score(y_test_cls, pred_test, average="macro", zero_division=0)
rec_test_macro  = recall_score(y_test_cls, pred_test, average="macro", zero_division=0)

print("\n=== Metrics (TRAIN) ===")
print("Accuracy:", acc_train)
print("Balanced Accuracy:", bacc_train)
print("F1 macro:", f1_train_macro)

print("\n=== Metrics (TEST) ===")
print("Accuracy:", acc_test)
print("Balanced Accuracy:", bacc_test)
print("F1 macro:", f1_test_macro)
print("Precision macro:", prec_test_macro)
print("Recall macro:", rec_test_macro)

print("\n=== Classification report (TEST) ===")
print(classification_report(
    y_test_cls, pred_test,
    target_names=class_labels_in_order,
    zero_division=0
))


# ==============================
# 7) ROC curves (OvR) by class (TEST)
# ==============================
plt.figure(figsize=(8, 6))
for c in range(5):
    y_true_bin = (y_test_cls.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    auc_c = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc_c:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==============================
# 8) Confusion matrix (TEST)
# ==============================
cm = confusion_matrix(y_test_cls, pred_test)
cm_df = pd.DataFrame(cm, index=[f"Réel: {class_names[i]}" for i in range(5)],
                     columns=[f"Prédit: {class_names[i]}" for i in range(5)])
print("\nMatrice de confusion (test)")
display(cm_df)


# ==============================
# 9) SHAP — Global + Example row
#    Fix: SHAP multiclass output shape differences
# ==============================
# Sample for speed
sample_size = min(300, len(X_test_xgb_cls))
X_shap = X_test_xgb_cls.sample(sample_size, random_state=42)

explainer = shap.TreeExplainer(best_xgb_cls)
shap_values = explainer.shap_values(X_shap)

# In multiclass, shap_values is usually a list: [array(n, p), ...] per class
# We choose which class to visualize globally:
class_for_global = 0  # change 0..4 if you want
print("\nSHAP global (summary) — classe:", class_for_global, "-", class_names[class_for_global])

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[class_for_global], X_shap, show=True)
    shap.summary_plot(shap_values[class_for_global], X_shap, plot_type="bar", show=True)
else:
    # fallback (rare): single array
    shap.summary_plot(shap_values, X_shap, show=True)
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=True)


# ==============================
# 10) Predict your example row + SHAP local
#     You MUST provide it already encoded & aligned to training columns
# ==============================
# If you already built an encoded example row aligned to X_train_encoded_no_te:
# Example variable name: X_new_encoded_no_te (1 row)
# If you named it differently, set it here:
X_one_cls = None

# Try to detect common names; otherwise set manually
for cand in ["X_new_encoded_no_te", "X_new_xgb_cls_no_te", "X_new_no_te_encoded"]:
    if cand in globals():
        X_one_cls = globals()[cand]
        break

if X_one_cls is None:
    print("\n[INFO] No encoded example row found. Create it and name it X_new_encoded_no_te (1-row DataFrame).")
else:
    X_one_cls = X_one_cls.copy()
    # Ensure same columns + order + clean names
    X_one_cls = X_one_cls.reindex(columns=X_train_xgb_cls.columns, fill_value=0)
    X_one_cls.columns = X_train_xgb_cls.columns
    X_one_cls = X_one_cls.apply(pd.to_numeric, errors="coerce").astype(float)

    proba_ex = best_xgb_cls.predict_proba(X_one_cls)[0]
    pred_ex_class = int(np.argmax(proba_ex))
    print("\n=== Example prediction ===")
    print("Classe prédite:", pred_ex_class, "-", class_names[pred_ex_class])
    print("Probabilités par classe:")
    for i in range(5):
        print(f"  {i} - {class_names[i]}: {proba_ex[i]:.4f}")

    # Local SHAP for predicted class
    shap_one = explainer.shap_values(X_one_cls)

    if isinstance(shap_one, list):
        sv = shap_one[pred_ex_class][0]  # (p,)
        base_val = explainer.expected_value[pred_ex_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        sv = shap_one[0]
        base_val = explainer.expected_value

    # Waterfall explanation
    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=base_val,
            data=X_one_cls.iloc[0],
            feature_names=X_one_cls.columns
        ),
        max_display=20
    )
    plt.show()


# ==============================
# 11) Save model + metadata
# ==============================
MODEL_NAME = "xgb_cls_randomsearch_weighted_no_te_v1"
save_dir = os.path.join("models", "xgboost", "classification", MODEL_NAME)
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_xgb_cls, os.path.join(save_dir, "model.joblib"))

metadata = {
    "model_name": MODEL_NAME,
    "best_params": best_params,
    "best_cv_f1_macro": float(best_cv_score),
    "metrics_train": {
        "accuracy": float(acc_train),
        "balanced_accuracy": float(bacc_train),
        "f1_macro": float(f1_train_macro),
    },
    "metrics_test": {
        "accuracy": float(acc_test),
        "balanced_accuracy": float(bacc_test),
        "f1_macro": float(f1_test_macro),
        "precision_macro": float(prec_test_macro),
        "recall_macro": float(rec_test_macro),
    },
    "class_names": class_names,
}

with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# Save feature names used (important for consistent inference later)
with open(os.path.join(save_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
    for col in X_train_xgb_cls.columns:
        f.write(str(col) + "\n")

print("\nSaved to:", save_dir)

```
