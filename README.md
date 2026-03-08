```

# ============================================================
# 7) EXAMPLE PREDICTION (robuste: alignement colonnes)
# ============================================================

# Priority:
# - if X_new_encoded_no_te exists: use it (already encoded)
# - else if test_row_xgb_no_te exists: build df and align (but MUST be encoded the same way!)
X_example = None

if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()

elif "test_row_xgb_no_te" in globals():
    # ATTENTION: ceci marche uniquement si test_row_xgb_no_te est déjà ENCODÉ
    # (mêmes colonnes que X_train_all, sinon il faut passer par TON pipeline d'encodage)
    X_example = pd.DataFrame(test_row_xgb_no_te).copy()

else:
    raise ValueError("Provide X_new_encoded_no_te (encoded example) or test_row_xgb_no_te (already-encoded dict).")


# -----------------------------
# FIX: aligner les colonnes avec le train
# -----------------------------
# 1) Ajouter les colonnes manquantes
missing_cols = [c for c in X_train_all.columns if c not in X_example.columns]
for c in missing_cols:
    X_example[c] = 0.0

# 2) Supprimer les colonnes en trop
extra_cols = [c for c in X_example.columns if c not in X_train_all.columns]
if len(extra_cols) > 0:
    X_example = X_example.drop(columns=extra_cols)

# 3) Réordonner exactement comme le train
X_example = X_example[X_train_all.columns].copy()

# 4) Forcer float
X_example = X_example.astype(float)

print("✅ Example aligned shape:", X_example.shape)
print("✅ Example columns match train:", list(X_example.columns) == list(X_train_all.columns))

# -----------------------------
# Gated prediction
# -----------------------------
ex_class_arr, ex_value_arr = gated_predict(X_example, xgb_cls_gated, xgb_reg_by_class)

ex_class = int(ex_class_arr[0])
ex_value = float(ex_value_arr[0])

low, high, _ = class_ranges[ex_class]

print("\n=== EXAMPLE PREDICTION (GATED) ===")
print("Classe prédite:", ex_class, "|", class_names_fr.get(ex_class, ex_class))
print("Intervalle:", (low, high))
print("Note prédite:", ex_value)
```
