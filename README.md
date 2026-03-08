```
# ==============================
# SAVE — CatBoost classification (BASELINE)
# Saves to: models/catboost/classification/baseline/<model_name>/
# Needs variables:
#   - cat_cls  (trained CatBoostClassifier)
# Optional:
#   - metrics_df (train/test metrics DataFrame)
#   - cm_df      (confusion matrix DataFrame)
#   - cat_params_cls_base (dict)
# ==============================
import os
import json
import time

technique = "baseline"
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = f"catboost_cls_{technique}_{timestamp}"

save_dir = os.path.join("models", "catboost", "classification", technique, model_name)
os.makedirs(save_dir, exist_ok=True)

cat_cls.save_model(os.path.join(save_dir, "model.cbm"))

if "metrics_df" in globals():
    metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

if "cm_df" in globals():
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

if "cat_params_cls_base" in globals():
    with open(os.path.join(save_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(cat_params_cls_base, f, ensure_ascii=False, indent=2)

meta = {
    "technique": technique,
    "model_name": model_name,
    "created_at": timestamp
}
with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved to:", save_dir)

```


```
# ==============================
# SAVE — CatBoost classification (RANDOM SEARCH)
# Saves to: models/catboost/classification/random_search/<model_name>/
# Needs variables:
#   - best_cat_cls (trained best CatBoostClassifier from Random Search)
# Optional:
#   - metrics_df, cm_df, best_params, best_cv
# ==============================
import os
import json
import time

technique = "random_search"
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = f"catboost_cls_{technique}_{timestamp}"

save_dir = os.path.join("models", "catboost", "classification", technique, model_name)
os.makedirs(save_dir, exist_ok=True)

best_cat_cls.save_model(os.path.join(save_dir, "model.cbm"))

if "metrics_df" in globals():
    metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

if "cm_df" in globals():
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

if "best_params" in globals():
    with open(os.path.join(save_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

meta = {
    "technique": technique,
    "model_name": model_name,
    "created_at": timestamp
}
if "best_cv" in globals():
    meta["best_cv_score"] = float(best_cv)

with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved to:", save_dir)

```

```
# ==============================
# SAVE — CatBoost classification (GRID SEARCH)
# Saves to: models/catboost/classification/grid_search/<model_name>/
# Needs variables:
#   - best_cat_cls (trained best CatBoostClassifier from Grid Search)
# Optional:
#   - metrics_df, cm_df, best_params, best_cv
# ==============================
import os
import json
import time

technique = "grid_search"
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = f"catboost_cls_{technique}_{timestamp}"

save_dir = os.path.join("models", "catboost", "classification", technique, model_name)
os.makedirs(save_dir, exist_ok=True)

best_cat_cls.save_model(os.path.join(save_dir, "model.cbm"))

if "metrics_df" in globals():
    metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

if "cm_df" in globals():
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

if "best_params" in globals():
    with open(os.path.join(save_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

meta = {
    "technique": technique,
    "model_name": model_name,
    "created_at": timestamp
}
if "best_cv" in globals():
    meta["best_cv_score"] = float(best_cv)

with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved to:", save_dir)


```


```
# ==============================
# SAVE — CatBoost classification (BAYESIAN / OPTUNA)
# Saves to: models/catboost/classification/bayesian_optuna/<model_name>/
# Needs variables:
#   - cat_cls_bayes (trained final CatBoostClassifier from Optuna)
# Optional:
#   - metrics_df, cm_df, best_params, best_cv_score
# ==============================
import os
import json
import time

technique = "bayesian_optuna"
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = f"catboost_cls_{technique}_{timestamp}"

save_dir = os.path.join("models", "catboost", "classification", technique, model_name)
os.makedirs(save_dir, exist_ok=True)

cat_cls_bayes.save_model(os.path.join(save_dir, "model.cbm"))

if "metrics_df" in globals():
    metrics_df.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

if "cm_df" in globals():
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix_test.csv"), index=True)

if "best_params" in globals():
    with open(os.path.join(save_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

meta = {
    "technique": technique,
    "model_name": model_name,
    "created_at": timestamp
}
if "best_cv_score" in globals():
    meta["best_cv_score"] = float(best_cv_score)

with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved to:", save_dir)

```
