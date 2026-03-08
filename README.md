```
# ============================================================
# NOTEBOOK: GATED MODEL (XGBoost Classifier -> XGBoost Regressors)
# FIXED: meta.json may NOT contain feature_cols -> we recover feature names from XGBoost model
# (NO SHAP PLOTS) -> SHAP AS TABLE + send SHAP + data + context to Ollama LLM
#
# PREREQUISITES (outside python):
#   1) ollama serve
#   2) ollama run llama3.1:latest
#
# Python installs (run once):
#   pip install -U openai shap xgboost pandas numpy tabulate
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

import shap
import xgboost as xgb
from tabulate import tabulate
from openai import OpenAI


# ============================================================
# 0) USER CONFIG (EDIT ME)
# ============================================================

# ---- A) Project root (auto)
PROJECT_ROOT = Path.cwd()
print("PROJECT_ROOT =", PROJECT_ROOT)

# ---- B) Where gated model is saved
GATED_MODEL_DIR = PROJECT_ROOT / "models" / "assembled_models" / "xgb_gated_optuna_v1"

XGB_CLS_FILE = GATED_MODEL_DIR / "xgb_classifier.json"
META_FILE    = GATED_MODEL_DIR / "meta.json"
REG_DIR      = GATED_MODEL_DIR / "regressors_by_class"

print("GATED_MODEL_DIR =", GATED_MODEL_DIR)
print("Classifier exists:", XGB_CLS_FILE.exists())
print("Meta exists      :", META_FILE.exists())
print("Reg dir exists   :", REG_DIR.exists())

# ---- C) Ollama (OpenAI compatible)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:latest"

# ---- D) Business bounds
PRED_MIN, PRED_MAX = 0.0, 10.0

# ---- E) Your project context (you will edit this text)
PROJECT_CONTEXT_TEXT = """
Contexte:
Nous utilisons un système "gated" pour prédire une note de satisfaction client.
1) Un modèle de classification prédit une classe (zone de note).
2) Ensuite, un modèle de régression spécialisé (un par classe) prédit la note dans l'intervalle de cette classe.
Objectif: produire une prédiction interprétable et actionnable.
""".strip()

# ---- F) CLASS TABLE (EDITABLE)
class_ranges = {
    0: [0.0, 2.0,  "extrêmement mauvais (0–2)"],
    1: [3.0, 6.0,  "mauvais (3–6)"],
    2: [7.0, 8.0,  "neutre (7–8)"],
    3: [9.0, 9.0,  "bien (9)"],
    4: [10.0, 10.0, "très bien (10)"],
}
class_names_fr = {k: v[2] for k, v in class_ranges.items()}

# ---- G) RAW example input (EDIT ME)
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

# ---- H) Encoding config (must match training)
# If you didn't actually use one of these methods in training, remove it here too.
onehot_cols = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "operating_system"]
freq_cols   = ["marque", "model", "garantie", "list_prest"]
force_categorical_cols = ["code_postal"]

# If you used these during training, enable them (must match training)
log1p_cols = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant_indem",
]
zero_to_nan_cols = [
    # columns where 0 means "missing"
]


# ============================================================
# 1) LOAD MODELS + META
# ============================================================

if not GATED_MODEL_DIR.is_dir():
    raise FileNotFoundError(f"GATED_MODEL_DIR not found: {GATED_MODEL_DIR}")
if not XGB_CLS_FILE.is_file():
    raise FileNotFoundError(f"Missing classifier file: {XGB_CLS_FILE}")
if not REG_DIR.is_dir():
    raise FileNotFoundError(f"Missing regressors folder: {REG_DIR}")

meta = {}
if META_FILE.is_file():
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

# classifier
xgb_cls_gated = xgb.XGBClassifier()
xgb_cls_gated.load_model(str(XGB_CLS_FILE))

# regressors per class
xgb_reg_by_class = {}
for class_id in sorted(class_ranges.keys()):
    reg_file = REG_DIR / f"xgb_regressor_class_{class_id}.json"
    if not reg_file.is_file():
        raise FileNotFoundError(f"Missing regressor for class {class_id}: {reg_file}")
    reg = xgb.XGBRegressor()
    reg.load_model(str(reg_file))
    xgb_reg_by_class[class_id] = reg

print("\nLoaded:")
print("- classifier:", XGB_CLS_FILE.name)
print("- regressors:", len(xgb_reg_by_class))

# ============================================================
# 1bis) FEATURE NAMES FIX (NO meta.json needed)
# We recover encoded feature columns from the classifier model itself.
# ============================================================

def _get_feature_names_from_xgb(model) -> list:
    booster = model.get_booster()
    names = booster.feature_names
    if names is None:
        # fallback: infer from f0..fN using num_features()
        n = booster.num_features()
        names = [f"f{i}" for i in range(n)]
    return list(names)

feature_cols = _get_feature_names_from_xgb(xgb_cls_gated)
print("- encoded feature cols (from model):", len(feature_cols))

# ============================================================
# 1ter) FREQ MAPS
# If meta doesn't have them, we fallback to empty maps (safe but less accurate)
# IMPORTANT: If you used freq encoding during training, you SHOULD store freq_maps in meta.json.
# ============================================================

freq_maps = meta.get("freq_maps", {})
if not isinstance(freq_maps, dict):
    freq_maps = {}

missing_freq = [c for c in freq_cols if c not in freq_maps]
if len(missing_freq) > 0:
    print("\n[WARNING] meta.json has no freq_maps for:", missing_freq)
    print("          => fallback to 0 for unseen categories in these columns.")
    for c in missing_freq:
        freq_maps[c] = {}  # empty -> will map everything to 0


# ============================================================
# 2) PREPROCESSING: raw dict -> encoded df aligned with feature_cols
# ============================================================

def prepare_example_encoded(
    raw_row: dict,
    feature_cols: list,
    onehot_cols: list,
    freq_cols: list,
    freq_maps: dict,
    force_categorical_cols: list,
    log1p_cols: list,
    zero_to_nan_cols: list,
) -> pd.DataFrame:
    X_new = pd.DataFrame([raw_row]).copy()

    # forced categoricals as string
    for c in force_categorical_cols:
        if c in X_new.columns:
            X_new[c] = X_new[c].astype("string")

    # 0 -> NaN
    for c in zero_to_nan_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
            X_new.loc[X_new[c] == 0, c] = np.nan

    # log1p
    for c in log1p_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
            X_new[c] = X_new[c].clip(lower=0)
            X_new[c] = np.log1p(X_new[c])

    # frequency encoding
    for c in freq_cols:
        mapping = freq_maps.get(c, {})
        if c in X_new.columns:
            X_new[c] = X_new[c].map(mapping).fillna(0).astype(float)
        else:
            X_new[c] = 0.0

    # one-hot
    for c in onehot_cols:
        if c not in X_new.columns:
            X_new[c] = pd.NA
    X_new_oh = pd.get_dummies(X_new, columns=onehot_cols, dummy_na=True)

    # align to training encoded columns
    for c in feature_cols:
        if c not in X_new_oh.columns:
            X_new_oh[c] = 0.0
    extra = [c for c in X_new_oh.columns if c not in feature_cols]
    if len(extra) > 0:
        X_new_oh = X_new_oh.drop(columns=extra)

    X_new_oh = X_new_oh[feature_cols].copy()

    # IMPORTANT: XGBoost models saved in JSON may expect float
    X_new_oh = X_new_oh.astype(float)
    return X_new_oh


X_example = prepare_example_encoded(
    raw_row=example_row_raw,
    feature_cols=feature_cols,
    onehot_cols=onehot_cols,
    freq_cols=freq_cols,
    freq_maps=freq_maps,
    force_categorical_cols=force_categorical_cols,
    log1p_cols=log1p_cols,
    zero_to_nan_cols=zero_to_nan_cols,
)

print("\nExample encoded shape:", X_example.shape)
print("Columns match model  :", list(X_example.columns) == list(feature_cols))


# ============================================================
# 3) GATED PREDICT (class -> reg -> clip to interval)
# ============================================================

def gated_predict(X_encoded: pd.DataFrame, cls_model, reg_models_by_class: dict):
    proba = cls_model.predict_proba(X_encoded)
    pred_class = np.argmax(proba, axis=1).astype(int)

    pred_values = []
    for i, cid in enumerate(pred_class):
        cid = int(cid)
        reg = reg_models_by_class[cid]
        v = float(reg.predict(X_encoded.iloc[[i]])[0])

        v = float(np.clip(v, PRED_MIN, PRED_MAX))
        low, high, _ = class_ranges[cid]
        v = float(np.clip(v, low, high))

        pred_values.append(v)

    return pred_class, np.array(pred_values), proba


pred_class_arr, pred_note_arr, pred_proba = gated_predict(
    X_example, xgb_cls_gated, xgb_reg_by_class
)
pred_class_id = int(pred_class_arr[0])
pred_note = float(pred_note_arr[0])
low, high, label = class_ranges[pred_class_id]

print("\n=== GATED PREDICTION ===")
print("Classe prédite:", pred_class_id, "|", class_names_fr.get(pred_class_id))
print("Intervalle:", (low, high))
print("Note prédite:", pred_note)
print("Probas classes:", np.round(pred_proba[0], 4))


# ============================================================
# 4) SHAP (NO plots) -> TABLES + JSON payload for LLM
# ============================================================

def shap_top_table_from_vector(feature_names, feature_values, shap_vector, top_k=12):
    abs_vals = np.abs(shap_vector)
    idx = np.argsort(abs_vals)[::-1][:top_k]

    rows = []
    for j in idx:
        rows.append({
            "feature": str(feature_names[j]),
            "value": float(feature_values[j]) if pd.notna(feature_values[j]) else None,
            "contribution": float(shap_vector[j]),
            "importance_abs": float(abs_vals[j]),
        })

    df = pd.DataFrame(rows).sort_values("importance_abs", ascending=False).reset_index(drop=True)
    return df


# Background for SHAP
# We use a neutral background of zeros with correct columns.
# (If you saved a real background during training, you can load it from meta.json and replace this.)
X_bg = pd.DataFrame(np.zeros((200, len(feature_cols))), columns=feature_cols).astype(float)

# ---- A) Classifier SHAP
explainer_cls = shap.TreeExplainer(xgb_cls_gated, data=X_bg, feature_perturbation="interventional")

sv_one_cls = explainer_cls.shap_values(X_example)

# SHAP output can be:
# - list of arrays (one per class) shape (n_samples, n_features)
# - or ndarray shape (n_samples, n_features, n_classes)
if isinstance(sv_one_cls, list):
    sv_cls_local = sv_one_cls[pred_class_id][0]
else:
    sv_cls_local = sv_one_cls[0, :, pred_class_id]

cls_local_table = shap_top_table_from_vector(
    feature_names=X_example.columns.to_list(),
    feature_values=X_example.iloc[0].values,
    shap_vector=sv_cls_local,
    top_k=12
)

print("\n--- SHAP (Classification) : TOP facteurs pour la classe prédite ---")
print(tabulate(cls_local_table, headers="keys", tablefmt="github", showindex=False))

# global importance
sv_bg_cls = explainer_cls.shap_values(X_bg)
if isinstance(sv_bg_cls, list):
    sv_cls_global = sv_bg_cls[pred_class_id]
else:
    sv_cls_global = sv_bg_cls[:, :, pred_class_id]

mean_abs = np.mean(np.abs(sv_cls_global), axis=0)
cls_global_table = pd.DataFrame({
    "feature": X_bg.columns.astype(str),
    "mean_abs_contribution": mean_abs.astype(float),
}).sort_values("mean_abs_contribution", ascending=False).head(20).reset_index(drop=True)

print("\n--- SHAP (Classification) : importance globale (Top 20) ---")
print(tabulate(cls_global_table, headers="keys", tablefmt="github", showindex=False))

# ---- B) Regressor SHAP (for predicted class)
reg_model = xgb_reg_by_class[pred_class_id]
explainer_reg = shap.TreeExplainer(reg_model, data=X_bg, feature_perturbation="interventional")

sv_reg_local = explainer_reg.shap_values(X_example)[0]
reg_local_table = shap_top_table_from_vector(
    feature_names=X_example.columns.to_list(),
    feature_values=X_example.iloc[0].values,
    shap_vector=sv_reg_local,
    top_k=12
)

print("\n--- SHAP (Régression) : TOP facteurs pour la note (classe prédite) ---")
print(tabulate(reg_local_table, headers="keys", tablefmt="github", showindex=False))

sv_bg_reg = explainer_reg.shap_values(X_bg)
mean_abs_reg = np.mean(np.abs(sv_bg_reg), axis=0)
reg_global_table = pd.DataFrame({
    "feature": X_bg.columns.astype(str),
    "mean_abs_contribution": mean_abs_reg.astype(float),
}).sort_values("mean_abs_contribution", ascending=False).head(20).reset_index(drop=True)

print("\n--- SHAP (Régression) : importance globale (Top 20) ---")
print(tabulate(reg_global_table, headers="keys", tablefmt="github", showindex=False))


# ============================================================
# 5) Build LLM payload (input + prediction + SHAP tables)
# ============================================================

encoded_nonzero = X_example.iloc[0]
encoded_nonzero = encoded_nonzero[encoded_nonzero != 0.0].sort_values(ascending=False)
encoded_used_compact = encoded_nonzero.head(80).to_dict()

payload = {
    "contexte_projet": PROJECT_CONTEXT_TEXT,
    "donnees_entree_raw": example_row_raw,
    "donnees_utilisees_par_modele_encoded_compact": encoded_used_compact,
    "prediction": {
        "classe_id": pred_class_id,
        "classe_nom": class_names_fr.get(pred_class_id),
        "intervalle": [float(low), float(high)],
        "note_predite": float(pred_note),
        "probabilites_par_classe": {class_names_fr[i]: float(pred_proba[0][i]) for i in range(len(class_ranges))}
    },
    "explications_modele": {
        "classification": {
            "top_local": cls_local_table.to_dict(orient="records"),
            "top_global": cls_global_table.to_dict(orient="records")
        },
        "regression": {
            "top_local": reg_local_table.to_dict(orient="records"),
            "top_global": reg_global_table.to_dict(orient="records")
        }
    },
    "instructions_sortie": {
        "langue": "français",
        "style": "humain, clair, non-technique",
        "structure": [
            "1) Résumé (3–5 lignes)",
            "2) Pourquoi la classe (5 bullets max)",
            "3) Pourquoi la note dans cette classe (5 bullets max)",
            "4) Problème probable côté client (1–2 phrases)",
            "5) Actions recommandées (3 bullets)"
        ],
        "contraintes": [
            "Ne pas citer SHAP, ni termes trop techniques",
            "Ne pas citer les noms exacts des variables/colonnes: parler en termes métier (délais, manque d'info, décisions, pièces manquantes, etc.)"
        ]
    }
}


# ============================================================
# 6) Call Ollama via OpenAI client
# ============================================================

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

system_msg = """
Tu es un assistant expert en interprétation de modèles ML et en rédaction d'explications pour un public non technique.

Tu reçois:
- le contexte projet
- les données d'entrée (cas client)
- la prédiction (classe + note)
- des tableaux de facteurs influents (locaux + globaux) pour la classe et la note

Ta tâche:
- Expliquer clairement pourquoi la note est comme ça, et quel problème probable le client a rencontré.
- Proposer des actions concrètes.

Contraintes:
- Ne jamais citer "SHAP" ou des détails techniques.
- Ne jamais citer les noms exacts des variables/colonnes.
- Utiliser un langage métier (délais, manque d'information, décisions, pièces manquantes, etc.).
- Respecter la structure demandée.
""".strip()

user_msg = json.dumps(payload, ensure_ascii=False, indent=2)

resp = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    temperature=0.2
)

llm_text = resp.choices[0].message.content
print("\n================= EXPLICATION HUMAINE (LLM) =================\n")
print(llm_text)

```
