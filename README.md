```
# ============================================================
# NOTEBOOK: GATED MODEL (XGBoost Classifier -> XGBoost Regressors)
# + Load saved gated models (JSON)
# + Prepare ONE raw example dict -> encode/align -> predict (class + note)
# + SHAP global + SHAP local (for classifier + chosen regressor)
# + Send: context + prediction + main reasons to local Ollama (OpenAI compatible)
#
# PREREQUISITES (outside python):
#   1) Start Ollama server:
#        ollama serve
#   2) Make sure the model exists locally:
#        ollama run llama3.1:latest
#
# Python installs (run once if needed):
#   pip install -U openai shap xgboost pandas numpy matplotlib
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import xgboost as xgb
from openai import OpenAI


# ============================================================
# 0) USER CONFIG (EDIT ME)
# ============================================================

# ---- A) Project root (auto)
PROJECT_ROOT = Path.cwd()  # should be .../nps_final when you run the notebook
print("PROJECT_ROOT =", PROJECT_ROOT)

# ---- B) Where gated model is saved (relative to PROJECT_ROOT)
# Your folder is: models/assembled_models/xgb_gated_optuna_v1
GATED_MODEL_DIR = PROJECT_ROOT / "models" / "assembled_models" / "xgb_gated_optuna_v1"

# Files inside
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

# ---- E) Accuracy tolerance (for optional display)
TOL_POINTS = 1.0

# ---- F) Your project context (you can write it yourself)
PROJECT_CONTEXT_TEXT = """
Contexte:
Nous utilisons un système "gated" (à deux étages) pour prédire une note de satisfaction client.
1) Un modèle de classification prédit une classe (zone de note).
2) Ensuite, un modèle de régression spécialisé (un par classe) prédit la note à l'intérieur de l'intervalle de cette classe.
On veut une prédiction interprétable (explications) et utilisable par des équipes non techniques.

Tu vas recevoir:
- les données d'entrée (un cas client)
- la classe + note prédite
- les principales raisons (facteurs les plus influents) calculées à partir d'un outil d'explicabilité.
Ta tâche: expliquer en langage humain pourquoi la note est comme ça et quel est le problème probable du client.
""".strip()

# ---- G) CLASS TABLE (EDITABLE): class_id -> [low, high, French name]
# IMPORTANT: intervals are inclusive of low and high in our clipping step.
class_ranges = {
    0: [0.0, 2.0,  "extrêmement mauvais (0–2)"],
    1: [3.0, 6.0,  "mauvais (3–6)"],
    2: [7.0, 8.0,  "neutre (7–8)"],
    3: [9.0, 9.0,  "bien (9)"],
    4: [10.0, 10.0, "très bien (10)"],
}
class_names_fr = {k: v[2] for k, v in class_ranges.items()}

# ---- H) RAW example input (EDIT ME)
# Put the raw values here (same as you used before)
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

# ---- I) Preprocessing config for encoding (must match your training)
# If you saved these lists somewhere else, copy them here.
onehot_cols = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "operating_system"]
freq_cols   = ["marque", "model", "garantie", "list_prest"]
force_categorical_cols = ["code_postal"]  # will be cast to string

# OPTIONAL: log1p columns (only if you used them during training)
log1p_cols = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant_indem",
]

# OPTIONAL: "0 means missing" columns (only if you used this rule)
zero_to_nan_cols = [
    # e.g. "delai_decision",
]


# ============================================================
# 1) LOAD MODELS + META
# ============================================================

if not GATED_MODEL_DIR.is_dir():
    raise FileNotFoundError(f"GATED_MODEL_DIR not found: {GATED_MODEL_DIR}")

if not XGB_CLS_FILE.is_file():
    raise FileNotFoundError(f"Missing classifier file: {XGB_CLS_FILE}")

if not META_FILE.is_file():
    raise FileNotFoundError(f"Missing meta.json file: {META_FILE}")

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

# feature list (encoded columns) must exist in meta
feature_cols = (
    meta.get("feature_cols")
    or meta.get("train_columns")
    or meta.get("X_train_columns")
    or meta.get("encoded_feature_names")
)
if feature_cols is None:
    raise ValueError("meta.json must contain feature columns list (feature_cols/train_columns/...)")

# (optional) load encoding artifacts from meta
freq_maps = meta.get("freq_maps", {})  # dict of dicts: col -> {category: count}
# If your meta doesn't store it, we'll fit freq maps on the fly is impossible here.
# So we require it for frequency encoding.
missing_freq = [c for c in freq_cols if c not in freq_maps]
if len(missing_freq) > 0:
    raise ValueError(
        f"meta.json is missing freq_maps for columns: {missing_freq}. "
        "You must save freq_maps used in training to meta.json."
    )

# Load XGBoost classifier
xgb_cls_gated = xgb.XGBClassifier()
xgb_cls_gated.load_model(str(XGB_CLS_FILE))

# Load regressors per class
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
print("- feature cols:", len(feature_cols))


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
    # 1) DataFrame
    X_new = pd.DataFrame([raw_row]).copy()

    # 2) Ensure forced categorical columns as string (to match training)
    for c in force_categorical_cols:
        if c in X_new.columns:
            X_new[c] = X_new[c].astype("string")

    # 3) 0 -> NaN conversion if configured
    for c in zero_to_nan_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
            X_new.loc[X_new[c] == 0, c] = np.nan

    # 4) log1p if configured
    for c in log1p_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
            X_new[c] = X_new[c].clip(lower=0)
            X_new[c] = np.log1p(X_new[c])

    # 5) Frequency encoding
    for c in freq_cols:
        if c in X_new.columns:
            mapping = freq_maps.get(c, {})
            X_new[c] = X_new[c].map(mapping).fillna(0).astype(float)
        else:
            # if missing, create it as 0
            X_new[c] = 0.0

    # 6) One-hot encoding (dummy_na=True to match training)
    # Make sure onehot cols exist
    for c in onehot_cols:
        if c not in X_new.columns:
            X_new[c] = pd.NA

    X_new_oh = pd.get_dummies(X_new, columns=onehot_cols, dummy_na=True)

    # 7) Align to training encoded columns
    # Add missing columns with 0
    for c in feature_cols:
        if c not in X_new_oh.columns:
            X_new_oh[c] = 0.0
    # Drop extra columns
    extra = [c for c in X_new_oh.columns if c not in feature_cols]
    if len(extra) > 0:
        X_new_oh = X_new_oh.drop(columns=extra)

    # Reorder + float
    X_new_oh = X_new_oh[feature_cols].copy()
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
print("Columns match train  :", list(X_example.columns) == list(feature_cols))


# ============================================================
# 3) GATED PREDICT: classifier -> regressor(class) -> clip into interval
# ============================================================

def gated_predict(X_encoded: pd.DataFrame, cls_model, reg_models_by_class: dict):
    proba = cls_model.predict_proba(X_encoded)
    pred_class = np.argmax(proba, axis=1).astype(int)

    pred_values = []
    for i, cid in enumerate(pred_class):
        reg = reg_models_by_class[int(cid)]
        v = float(reg.predict(X_encoded.iloc[[i]])[0])

        # clip to [0,10]
        v = float(np.clip(v, PRED_MIN, PRED_MAX))

        # clip to class interval
        low, high, _ = class_ranges[int(cid)]
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
# 4) SHAP (Classifier + selected regressor)
# ============================================================

# We need a background dataset for SHAP. For inference notebook:
# - either load from meta (recommended)
# - or build synthetic background (not recommended)
#
# We'll try to load "shap_background" from meta if saved; otherwise we use
# a small random noise baseline around 0 (works but less meaningful).

X_bg = None
if "shap_background" in meta and isinstance(meta["shap_background"], dict):
    # meta["shap_background"] contains {"data": [[...], ...]}
    bg_data = meta["shap_background"].get("data")
    if bg_data is not None:
        X_bg = pd.DataFrame(bg_data, columns=feature_cols).astype(float)

if X_bg is None:
    # fallback: 200 rows of zeros (valid shape)
    X_bg = pd.DataFrame(np.zeros((200, len(feature_cols))), columns=feature_cols).astype(float)

# ---- A) SHAP for classifier (global on one class + local on example)
# Use TreeExplainer; for multiclass it returns list/array per class depending on version.
explainer_cls = shap.TreeExplainer(xgb_cls_gated, data=X_bg, feature_perturbation="interventional")
shap_values_cls = explainer_cls.shap_values(X_bg)

# pick class id to visualize (predicted class)
cls_for_global = pred_class_id

# normalize shap_values_cls to a 2D matrix for that class
# Possible shapes:
# - list of [n_samples, n_features] per class
# - array [n_samples, n_features, n_classes]
if isinstance(shap_values_cls, list):
    sv_global = shap_values_cls[cls_for_global]
else:
    sv_global = shap_values_cls[:, :, cls_for_global]

# Global bar plot
print("\nSHAP global (classifier) — classe:", cls_for_global, "|", class_names_fr.get(cls_for_global))
shap.summary_plot(sv_global, X_bg, plot_type="bar", show=True)

# Local explanation for example
sv_one_cls = explainer_cls.shap_values(X_example)
if isinstance(sv_one_cls, list):
    sv_one_cls_k = sv_one_cls[cls_for_global][0]
else:
    sv_one_cls_k = sv_one_cls[0, :, cls_for_global]

base_val_cls = explainer_cls.expected_value
if isinstance(base_val_cls, (list, np.ndarray)):
    base_val_cls = base_val_cls[cls_for_global]

print("\nSHAP local (classifier) — example")
shap.plots.waterfall(
    shap.Explanation(
        values=sv_one_cls_k,
        base_values=base_val_cls,
        data=X_example.iloc[0],
        feature_names=X_example.columns,
    ),
    max_display=20
)
plt.show()

# ---- B) SHAP for regressor of predicted class (global + local)
reg_model = xgb_reg_by_class[pred_class_id]
explainer_reg = shap.TreeExplainer(reg_model, data=X_bg, feature_perturbation="interventional")
sv_reg_bg = explainer_reg.shap_values(X_bg)

print("\nSHAP global (regressor) — classe:", pred_class_id, "|", class_names_fr.get(pred_class_id))
shap.summary_plot(sv_reg_bg, X_bg, plot_type="bar", show=True)

sv_reg_one = explainer_reg.shap_values(X_example)[0]
base_val_reg = explainer_reg.expected_value

print("\nSHAP local (regressor) — example")
shap.plots.waterfall(
    shap.Explanation(
        values=sv_reg_one,
        base_values=base_val_reg,
        data=X_example.iloc[0],
        feature_names=X_example.columns,
    ),
    max_display=20
)
plt.show()

# ---- Extract top reasons from regressor SHAP (best for explaining the note)
abs_vals = np.abs(sv_reg_one)
top_idx = np.argsort(abs_vals)[::-1][:10]

top_features = []
for idx in top_idx:
    feat = X_example.columns[idx]
    val = X_example.iloc[0, idx]
    contrib = float(sv_reg_one[idx])
    top_features.append({
        "feature": str(feat),
        "value": float(val) if pd.notna(val) else None,
        "shap_contribution": contrib
    })

print("\nTop features driving the NOTE (regressor):")
for t in top_features[:8]:
    sign = "+" if t["shap_contribution"] > 0 else ""
    print(f"- {t['feature']}: value={t['value']} | contrib={sign}{t['shap_contribution']:.4f}")


# ============================================================
# 5) SEND TO OLLAMA via OpenAI Python client
# ============================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # any string works for local Ollama
)

# Build ranked reasons without mentioning SHAP explicitly
reasons = []
for t in top_features[:8]:
    direction = "augmente" if t["shap_contribution"] > 0 else "diminue"
    reasons.append({
        "facteur": t["feature"],
        "valeur": t["value"],
        "effet": direction,
        "importance": float(abs(t["shap_contribution"]))
    })
reasons = sorted(reasons, key=lambda x: x["importance"], reverse=True)

system_msg = f"""
Tu es un assistant expert en interprétation de modèles ML et en rédaction d'explications pour un public non technique.

Objectif:
- Expliquer pourquoi le modèle a prédit cette note (et cette classe), en français clair.
- Décrire le problème probable vécu par le client.
- Proposer 3 actions concrètes.

Contraintes:
- Ne jamais citer "SHAP" ni des concepts techniques.
- Utilise un langage métier (retards, manque d'information, décisions, etc.) sans nommer des colonnes.
- Structure attendue:
  1) Résumé (3–5 lignes)
  2) Raisons principales (5–8 bullets max)
  3) Problème probable (1–2 phrases)
  4) Actions recommandées (3 bullets)

Contexte projet:
{PROJECT_CONTEXT_TEXT}
""".strip()

user_payload = {
    "prediction": {
        "classe_predite": class_names_fr.get(pred_class_id),
        "intervalle": [float(low), float(high)],
        "note_predite": float(pred_note),
        "probabilites_par_classe": {class_names_fr[i]: float(pred_proba[0][i]) for i in range(len(class_ranges))}
    },
    "donnees_entree": example_row_raw,
    "raisons_principales": reasons
}

resp = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ],
    temperature=0.2
)

llm_text = resp.choices[0].message.content
print("\n================= EXPLICATION HUMAINE (LLM) =================\n")
print(llm_text)


```
