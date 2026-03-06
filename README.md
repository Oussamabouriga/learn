```
# ==============================
# 3) Forcer "code_postal" en catégorielle + séparer numériques/catégorielles (SANS Target Encoding)
#    + Encodage catégoriel AU CHOIX (One-Hot / Frequency)
#    + Transformation numérique AU CHOIX :
#        1) log1p sur colonnes "délais" + "prix/montants" (TU CHOISIS)
#        2) RobustScaler sur ces colonnes transformées (TU CHOISIS)
#        3) Colonnes "counts" (0–5) laissées telles quelles (TU CHOISIS)
# ==============================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ------------------------------------------------------------
# A) Forcer code_postal en variable catégorielle
# ------------------------------------------------------------
col_cp = "code_postal"  # modifier si besoin
if col_cp in X_train_no_te.columns:
    X_train_no_te[col_cp] = X_train_no_te[col_cp].astype("string")
    X_test_no_te[col_cp]  = X_test_no_te[col_cp].astype("string")

print("Type code_postal (train):", X_train_no_te[col_cp].dtype if col_cp in X_train_no_te.columns else "non trouvé")
print("Type code_postal (test) :", X_test_no_te[col_cp].dtype  if col_cp in X_test_no_te.columns  else "non trouvé")

# ------------------------------------------------------------
# B) Copier les datasets (travail propre)
# ------------------------------------------------------------
Xtr = X_train_no_te.copy()
Xte = X_test_no_te.copy()

# ------------------------------------------------------------
# C) Identifier colonnes numériques vs catégorielles (avant encodage)
# ------------------------------------------------------------
numeric_cols_no_te = Xtr.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_no_te = [c for c in Xtr.columns if c not in numeric_cols_no_te]

print("\nNombre de colonnes numériques   :", len(numeric_cols_no_te))
print("Nombre de colonnes catégorielles:", len(categorical_cols_no_te))

# ------------------------------------------------------------
# D) Encodage des catégorielles (TU CHOISIS)
#    - One-Hot Encoding : petites cardinalités
#    - Frequency/Count Encoding : grandes cardinalités
# ------------------------------------------------------------
onehot_cols = [
    # "operating_system",
    # "garantie",
]
freq_cols = [
    # "model",
    # "code_postal",
]

onehot_cols = [c for c in onehot_cols if c in Xtr.columns]
freq_cols   = [c for c in freq_cols if c in Xtr.columns]

# -- Frequency Encoding (fit sur TRAIN seulement)
freq_maps = {}
for c in freq_cols:
    freq_maps[c] = Xtr[c].value_counts(dropna=False).to_dict()
    Xtr[c] = Xtr[c].map(freq_maps[c]).fillna(0).astype(float)
    Xte[c] = Xte[c].map(freq_maps[c]).fillna(0).astype(float)

# -- One-Hot Encoding
Xtr_oh = pd.get_dummies(Xtr, columns=onehot_cols, dummy_na=True)
Xte_oh = pd.get_dummies(Xte, columns=onehot_cols, dummy_na=True)

# Align colonnes train/test
X_train_encoded_no_te, X_test_encoded_no_te = Xtr_oh.align(Xte_oh, join="left", axis=1, fill_value=0)

print("\nShapes après encodage:")
print("Train encoded:", X_train_encoded_no_te.shape)
print("Test encoded :", X_test_encoded_no_te.shape)

# ------------------------------------------------------------
# E) Transformation NUMÉRIQUE (TU CHOISIS LES COLONNES)
#    1) log1p sur colonnes "délais" + "prix/montants"
#    2) RobustScaler sur ces colonnes log-transformées
#    3) Colonnes de type "count" (0–5) laissées telles quelles
# ------------------------------------------------------------

# 1) Colonnes à passer en log1p (TU CHOISIS)
log1p_cols = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant",
    # "price",
]

# 2) Colonnes "count" à laisser telles quelles (TU CHOISIS)
count_cols_no_scale = [
    # "nb_ko",
    # "nb_tickets",
]

# Garder uniquement les colonnes qui existent après encodage
log1p_cols = [c for c in log1p_cols if c in X_train_encoded_no_te.columns]
count_cols_no_scale = [c for c in count_cols_no_scale if c in X_train_encoded_no_te.columns]

# 3) Appliquer log1p (après encodage) — sur TRAIN+TEST avec la même règle
#    (log1p nécessite des valeurs >= -1 ; idéalement >= 0)
for c in log1p_cols:
    X_train_encoded_no_te[c] = pd.to_numeric(X_train_encoded_no_te[c], errors="coerce")
    X_test_encoded_no_te[c]  = pd.to_numeric(X_test_encoded_no_te[c],  errors="coerce")

    # sécurité: éviter valeurs négatives inattendues
    X_train_encoded_no_te[c] = X_train_encoded_no_te[c].clip(lower=0)
    X_test_encoded_no_te[c]  = X_test_encoded_no_te[c].clip(lower=0)

    X_train_encoded_no_te[c] = np.log1p(X_train_encoded_no_te[c])
    X_test_encoded_no_te[c]  = np.log1p(X_test_encoded_no_te[c])

print("\nlog1p appliqué sur:", log1p_cols)

# 4) RobustScaler sur les colonnes log-transformées (et éventuellement autres grandes colonnes)
#    Ici: on scale uniquement log1p_cols (recommandé)
scale_cols = [c for c in log1p_cols if c not in count_cols_no_scale]

if len(scale_cols) > 0:
    scaler = RobustScaler()
    X_train_encoded_no_te[scale_cols] = scaler.fit_transform(X_train_encoded_no_te[scale_cols])
    X_test_encoded_no_te[scale_cols]  = scaler.transform(X_test_encoded_no_te[scale_cols])

print("RobustScaler appliqué sur:", scale_cols)
print("Count cols laissées telles quelles:", count_cols_no_scale)


import numpy as np
import pandas as pd

# 1) Same columns order / count
print("Train shape:", X_train_encoded_no_te.shape)
print("Test  shape:", X_test_encoded_no_te.shape)
print("Same columns:", X_train_encoded_no_te.columns.equals(X_test_encoded_no_te.columns))

# 2) All columns must be numeric
non_num_train = X_train_encoded_no_te.select_dtypes(exclude=[np.number]).columns.tolist()
non_num_test  = X_test_encoded_no_te.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric cols (train):", non_num_train[:20])
print("Non-numeric cols (test) :", non_num_test[:20])

# 3) Any weird missing types (ensure only np.nan)
print("Any NaN in train:", X_train_encoded_no_te.isna().any().any())
print("Any NaN in test :", X_test_encoded_no_te.isna().any().any())

# 4) Convert to float for XGBoost safety (recommended)
X_train_encoded_no_te = X_train_encoded_no_te.astype(float)
X_test_encoded_no_te  = X_test_encoded_no_te.astype(float)

print(" Ready for training if:")
print("- Same columns == True")
print("- Non-numeric cols lists are empty")

```
