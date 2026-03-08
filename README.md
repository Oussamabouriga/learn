```


# ============================================================
# EXAMPLE RAW -> ENCODED (NO TE) -> GATED PREDICTION
# (utilise exactement les mêmes règles que ton train)
#
# Prérequis (doivent exister dans ton notebook) :
# - X_train_no_te, X_test_no_te         (raw split)
# - X_train_encoded_no_te, X_test_encoded_no_te  (encoded split)
# - X_train_all                         (matrice finale utilisée par le gated model)
# - onehot_cols, freq_cols              (listes choisies)
# - freq_maps                           (dict {col: {category: frequency}} appris sur TRAIN)
# - zero_to_nan_cols_no_te              (colonnes où 0 = "missing métier")
# - log1p_cols_no_te                    (colonnes à log1p)
# - scaler + scale_cols                 (si tu as appliqué un scaler)  [optionnel]
# - xgb_cls_gated, xgb_reg_by_class, gated_predict, class_ranges, class_names_fr
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Ton exemple RAW (tu modifies ici)
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

X_raw = pd.DataFrame([example_row_raw]).copy()


# -----------------------------
# 1) Aligner aux colonnes RAW du train (avant encodage)
# -----------------------------
raw_cols_train = list(X_train_no_te.columns)

for c in raw_cols_train:
    if c not in X_raw.columns:
        X_raw[c] = np.nan

# Supprimer colonnes "en trop"
extra = [c for c in X_raw.columns if c not in raw_cols_train]
if extra:
    X_raw = X_raw.drop(columns=extra)

# Réordonner
X_raw = X_raw[raw_cols_train].copy()


# -----------------------------
# 2) 0 -> NaN (mêmes colonnes que le train)
# -----------------------------
if "zero_to_nan_cols_no_te" in globals():
    for c in zero_to_nan_cols_no_te:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
            X_raw.loc[X_raw[c] == 0, c] = np.nan


# -----------------------------
# 3) log1p (mêmes colonnes que le train)
# -----------------------------
if "log1p_cols_no_te" in globals():
    for c in log1p_cols_no_te:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
            X_raw[c] = X_raw[c].clip(lower=0)
            X_raw[c] = np.log1p(X_raw[c])


# -----------------------------
# 4) Frequency/Count encoding (mêmes maps apprises sur TRAIN)
# -----------------------------
if "freq_cols" in globals() and "freq_maps" in globals():
    for c in freq_cols:
        if c in X_raw.columns:
            # IMPORTANT: on mappe avec le dict appris sur TRAIN
            mp = freq_maps.get(c, {})
            X_raw[c] = X_raw[c].map(mp).fillna(0).astype(float)


# -----------------------------
# 5) One-Hot encoding (mêmes colonnes que le train)
#    - ici on utilise pd.get_dummies comme au training
#    - puis on aligne sur X_train_encoded_no_te
# -----------------------------
if "onehot_cols" in globals():
    X_oh = pd.get_dummies(X_raw, columns=[c for c in onehot_cols if c in X_raw.columns], dummy_na=True)

    # Align avec le train encodé (mêmes colonnes)
    X_oh_aligned, _ = X_oh.align(X_train_encoded_no_te, join="right", axis=1, fill_value=0)
    X_encoded = X_oh_aligned.copy()
else:
    # Si tu n'as pas de onehot, alors l'encoded = raw (après freq/log/0->nan)
    X_encoded = X_raw.copy()


# -----------------------------
# 6) Scaling (si tu as appliqué un scaler au training)
#    => on ré-applique le même scaler sur les mêmes colonnes
# -----------------------------
if "scaler" in globals() and "scale_cols" in globals() and len(scale_cols) > 0:
    cols_to_scale = [c for c in scale_cols if c in X_encoded.columns]
    if len(cols_to_scale) > 0:
        X_encoded[cols_to_scale] = scaler.transform(X_encoded[cols_to_scale])


# -----------------------------
# 7) Align final EXACT avec la matrice du gated model (X_train_all)
#    (c'est ça qui évite ton KeyError)
# -----------------------------
for c in X_train_all.columns:
    if c not in X_encoded.columns:
        X_encoded[c] = 0.0

extra2 = [c for c in X_encoded.columns if c not in X_train_all.columns]
if extra2:
    X_encoded = X_encoded.drop(columns=extra2)

X_example = X_encoded[X_train_all.columns].astype(float).copy()

print("X_example final shape:", X_example.shape)
print("Columns match X_train_all:", list(X_example.columns) == list(X_train_all.columns))


# -----------------------------
# 8) Gated prediction (classification -> regression par classe)
# -----------------------------
ex_class_arr, ex_value_arr = gated_predict(X_example, xgb_cls_gated, xgb_reg_by_class)

ex_class = int(ex_class_arr[0])
ex_value = float(ex_value_arr[0])

low, high, _ = class_ranges[ex_class]

print("\n=== EXAMPLE PREDICTION (GATED) ===")
print("Classe prédite:", ex_class, "|", class_names_fr.get(ex_class, ex_class))
print("Intervalle:", (low, high))
print("Note prédite:", ex_value)

````
