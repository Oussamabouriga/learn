```

# ============================================================
# 7) EXAMPLE PREDICTION (robuste: alignement colonnes)
# + Exemple dict à tester (RAW)
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# A) Exemple RAW (tu peux modifier les valeurs)
# -----------------------------
example_row_raw = {
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
}

# Version liste (si tu veux plusieurs lignes)
test_row_xgb_no_te = [example_row_raw]


# -----------------------------
# B) Construire X_example
# Priority:
#  - if X_new_encoded_no_te exists: use it (already encoded)
#  - else: use the dict above BUT it MUST already be ENCODED like X_train_all
# -----------------------------
X_example = None

if "X_new_encoded_no_te" in globals():
    # ✅ Recommandé: exemple déjà encodé avec ton pipeline
    X_example = X_new_encoded_no_te.copy()

else:
    # ⚠️ Ici: ton dict "example_row_raw" est RAW.
    # Pour que ça marche, il doit être transformé par le MÊME pipeline d'encodage que le train.
    # Donc: soit tu as déjà une variable encodée (X_new_encoded_no_te),
    # soit tu dois appeler ton code d'encodage pour produire X_example_encoded.
    X_example = pd.DataFrame(test_row_xgb_no_te).copy()


# -----------------------------
# C) FIX: aligner les colonnes avec le train (évite KeyError)
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

print("Example aligned shape:", X_example.shape)
print("Example columns match train:", list(X_example.columns) == list(X_train_all.columns))


# -----------------------------
# D) Gated prediction
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
