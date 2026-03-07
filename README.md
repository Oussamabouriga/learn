```
import numpy as np
import pandas as pd

# 1) Toujours aligner sur les colonnes du train
X_new_encoded_no_te = X_new_encoded_no_te.copy()
X_new_encoded_no_te = X_new_encoded_no_te.reindex(columns=X_train_encoded_no_te.columns, fill_value=0)

print("X_new_encoded_no_te.shape =", X_new_encoded_no_te.shape)

# 2) Predict
pred_new = float(np.clip(xgboost_reg_no_te.predict(X_new_encoded_no_te)[0], 0, 10))
print("✅ Prediction (note 0-10) =", pred_new)

```

```
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- SHAP Explainer (XGBoost Tree model)
explainer = shap.TreeExplainer(xgboost_reg_no_te)

# ============================
# A) SHAP GLOBAL (sur un échantillon du test)
# ============================
# prends un petit sample pour aller vite
X_shap = X_test_encoded_no_te.copy()
X_shap = X_shap.reindex(columns=X_train_encoded_no_te.columns, fill_value=0)

sample_size = min(300, len(X_shap))
X_shap_sample = X_shap.sample(sample_size, random_state=42)

shap_values_global = explainer.shap_values(X_shap_sample)

plt.figure()
shap.summary_plot(shap_values_global, X_shap_sample, show=True)          # beeswarm
plt.figure()
shap.summary_plot(shap_values_global, X_shap_sample, plot_type="bar", show=True)  # bar importance

# ============================
# B) SHAP LOCAL (sur ton exemple)
# ============================
X_one = X_new_encoded_no_te.copy()
X_one = X_one.reindex(columns=X_train_encoded_no_te.columns, fill_value=0)

shap_values_one = explainer.shap_values(X_one)

# waterfall (explication claire)
# expected_value = moyenne de sortie du modèle (baseline)
base_value = explainer.expected_value
pred_value = float(xgboost_reg_no_te.predict(X_one)[0])

print("\n--- Lecture rapide ---")
print("E[f(X)] (baseline) =", base_value)
print("f(X) (prédiction modèle) =", pred_value)
print("Somme(SHAP) + baseline ≈ prédiction")

# Plot waterfall
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[0],
        base_values=base_value,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()
```
