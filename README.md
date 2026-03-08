```
from catboost import CatBoostClassifier, Pool

# ==============================
# Refit FINAL (Option A)
# - garder bootstrap_type=Bayesian
# - supprimer subsample si présent
# ==============================

best_params_refit = dict(best_params)

# Option A: si bootstrap_type = Bayesian => on retire subsample (incompatible)
if str(best_params_refit.get("bootstrap_type", "")).lower() == "bayesian":
    best_params_refit.pop("subsample", None)

# Nouveau modèle (important: best_estimator_ est déjà fitted)
best_cat_cls_refit = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    **best_params_refit
)

# Pools (réutilise tes variables)
train_pool_cat_cls = Pool(X_train_cat_cls_no_te, y_train_cat_cls_no_te, cat_features=cat_cols_cb)
test_pool_cat_cls  = Pool(X_test_cat_cls_no_te,  y_test_cat_cls_no_te,  cat_features=cat_cols_cb)

best_cat_cls_refit.fit(
    train_pool_cat_cls,
    eval_set=test_pool_cat_cls,
    use_best_model=True,
    early_stopping_rounds=150
)

print("Refit OK (Option A) — sans relancer Random Search")
print("bootstrap_type:", best_cat_cls_refit.get_params().get("bootstrap_type"))
print("subsample:", best_cat_cls_refit.get_params().get("subsample"))

```
