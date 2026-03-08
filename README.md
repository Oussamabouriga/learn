```

# ============================================================
# NOTEBOOK: GATED MODEL (XGBoost Classifier -> XGBoost Regressors)
# + SHAP (global + local on the example)
# + Send everything to local Ollama LLM via OpenAI Python client
#
# PREREQUISITES (outside python):
#   1) Start Ollama server:
#        ollama serve
#   2) Ensure model is pulled / runnable:
#        ollama run llama3.1:latest
#
# Python installs (run once if needed):
#   pip install -U openai shap joblib pandas numpy
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from openai import OpenAI


# ============================================================
# 0) USER CONFIG (EDIT ME)
# ============================================================

# --- A) Where your gated model is saved
# Put your real folder here (the one you used when saving)
GATED_MODEL_DIR = "models/assembled_models/xgboost_gated_optuna_v1"

# Expected files inside GATED_MODEL_DIR (change if your names differ)
# - xgb_classifier.pkl
# - xgb_regressors_by_class.pkl
# - preprocessing_artifacts.pkl
# - config.json (optional)
XGB_CLS_FILE = os.path.join(GATED_MODEL_DIR, "xgb_classifier.pkl")
XGB_REGS_FILE = os.path.join(GATED_MODEL_DIR, "xgb_regressors_by_class.pkl")
PREP_FILE = os.path.join(GATED_MODEL_DIR, "preprocessing_artifacts.pkl")
CFG_FILE = os.path.join(GATED_MODEL_DIR, "config.json")  # optional

# --- B) Local Ollama (OpenAI-compatible) settings
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:latest"   # use exactly what "ollama run ..." uses

# --- C) Your editable “big context” (LLM instructions)
PROJECT_CONTEXT_TEXT = """
Contexte:
Tu es un auditeur/validateur de décision pour un système de prédiction de satisfaction client.
Nous avons un modèle ML “gated” en 2 étages:
(1) Un classifieur prédit une classe de satisfaction (intervalle de notes).
(2) Un régresseur spécialisé par classe prédit la note finale à l’intérieur de l’intervalle.

Ta mission:
- Valider si la prédiction est cohérente avec les informations d’entrée.
- Expliquer si certaines variables semblent contradictoires avec la note prédite.
- Donner des recommandations (données manquantes à vérifier, variables incohérentes, etc.).
- Donner un verdict: “OK / À revoir” avec une justification courte.

Règles:
- Ne pas inventer des données.
- S’appuyer sur les features importantes fournies (SHAP).
- Si une info est manquante, le dire explicitement.
"""

# --- D) Tolerance for business accuracy (optional, used for interpretation text only)
TOL_POINTS = 1.0

# --- E) Example row (RAW) you want the gated model to predict on
# Put ONLY raw values here (before any encoding)
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


# ============================================================
# 1) LOAD MODELS + PREPROCESSING ARTIFACTS
# ============================================================

if not os.path.isdir(GATED_MODEL_DIR):
    raise FileNotFoundError(f"GATED_MODEL_DIR not found: {GATED_MODEL_DIR}")

if not os.path.isfile(XGB_CLS_FILE):
    raise FileNotFoundError(f"Missing file: {XGB_CLS_FILE}")

if not os.path.isfile(XGB_REGS_FILE):
    raise FileNotFoundError(f"Missing file: {XGB_REGS_FILE}")

if not os.path.isfile(PREP_FILE):
    raise FileNotFoundError(
        f"Missing file: {PREP_FILE}\n"
        f"You must save preprocessing artifacts used during training (onehot_cols, freq_cols, freq_maps, etc.)."
    )

xgb_cls_gated = joblib.load(XGB_CLS_FILE)
xgb_reg_by_class = joblib.load(XGB_REGS_FILE)
prep = joblib.load(PREP_FILE)

# Optional config (class definitions, names, ranges)
config = {}
if os.path.isfile(CFG_FILE):
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

print("Loaded classifier:", type(xgb_cls_gated))
print("Loaded regressors_by_class keys:", list(xgb_reg_by_class.keys())[:10])
print("Loaded prep keys:", list(prep.keys())[:20])
print("Loaded config keys:", list(config.keys())[:20])


# ============================================================
# 2) UNPACK PREPROCESSING ARTIFACTS (must match training)
# ============================================================

# REQUIRED (must be saved at training time)
raw_cols_train = prep["raw_cols_train"]          # list of raw columns expected before encoding
X_train_all_cols = prep["X_train_all_cols"]      # final encoded columns used by the gated models

onehot_cols = prep.get("onehot_cols", [])
freq_cols = prep.get("freq_cols", [])
freq_maps = prep.get("freq_maps", {})            # dict {col: {category: frequency}}
zero_to_nan_cols_no_te = prep.get("zero_to_nan_cols_no_te", [])
log1p_cols_no_te = prep.get("log1p_cols_no_te", [])

# scaler is optional (may be None)
scaler = prep.get("scaler", None)
scale_cols = prep.get("scale_cols", [])

# Class ranges / names (prefer config, else prep, else fallback)
class_ranges = config.get("class_ranges", prep.get("class_ranges", None))
class_names_fr = config.get("class_names_fr", prep.get("class_names_fr", None))

if class_ranges is None:
    raise ValueError("Missing class_ranges in config.json or preprocessing_artifacts.pkl")

if class_names_fr is None:
    # fallback (ids only)
    class_names_fr = {int(k): f"Classe_{k}" for k in range(len(class_ranges))}

# Convert class_ranges to dict[int] -> (low, high, name)
# We accept either:
#  - dict: { "0": [0,2,"xxx"], ... } or {0: (0,2,"xxx"), ...}
#  - list: [(0,2,"xxx"), (3,6,"yyy"), ...]
if isinstance(class_ranges, dict):
    _tmp = {}
    for k, v in class_ranges.items():
        kk = int(k)
        _tmp[kk] = (float(v[0]), float(v[1]), str(v[2]) if len(v) > 2 else f"Classe_{kk}")
    class_ranges = _tmp
elif isinstance(class_ranges, list):
    _tmp = {}
    for i, v in enumerate(class_ranges):
        _tmp[i] = (float(v[0]), float(v[1]), str(v[2]) if len(v) > 2 else f"Classe_{i}")
    class_ranges = _tmp

print("\nClass ranges used:")
for k in sorted(class_ranges.keys()):
    print(k, "=>", class_ranges[k], "|", class_names_fr.get(k, k))


# ============================================================
# 3) PREPARE EXAMPLE ROW (RAW -> ENCODED) EXACTLY LIKE TRAINING
# ============================================================

# 3.1 Build 1-row DF from example_row_raw
X_raw = pd.DataFrame([example_row_raw]).copy()

# 3.2 Align to raw training columns (add missing, drop extras, reorder)
for c in raw_cols_train:
    if c not in X_raw.columns:
        X_raw[c] = np.nan

extra = [c for c in X_raw.columns if c not in raw_cols_train]
if extra:
    X_raw = X_raw.drop(columns=extra)

X_raw = X_raw[raw_cols_train].copy()

# 3.3 Apply "0 -> NaN" business rule columns (same as training)
for c in zero_to_nan_cols_no_te:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        X_raw.loc[X_raw[c] == 0, c] = np.nan

# 3.4 Apply log1p on same columns as training
for c in log1p_cols_no_te:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        X_raw[c] = X_raw[c].clip(lower=0)
        X_raw[c] = np.log1p(X_raw[c])

# 3.5 Frequency/Count encoding using TRAIN maps only
for c in freq_cols:
    if c in X_raw.columns:
        mp = freq_maps.get(c, {})
        X_raw[c] = X_raw[c].map(mp).fillna(0).astype(float)

# 3.6 One-hot (pd.get_dummies), then align to training encoded columns
if len(onehot_cols) > 0:
    X_oh = pd.get_dummies(X_raw, columns=[c for c in onehot_cols if c in X_raw.columns], dummy_na=True)
else:
    X_oh = X_raw.copy()

# Align to training encoded feature space (X_train_all_cols)
# add missing => 0, drop extras, reorder
for c in X_train_all_cols:
    if c not in X_oh.columns:
        X_oh[c] = 0.0

extra2 = [c for c in X_oh.columns if c not in X_train_all_cols]
if extra2:
    X_oh = X_oh.drop(columns=extra2)

X_encoded = X_oh[X_train_all_cols].copy()

# 3.7 Optional scaling (if a scaler was used at training)
if scaler is not None and len(scale_cols) > 0:
    cols_to_scale = [c for c in scale_cols if c in X_encoded.columns]
    if len(cols_to_scale) > 0:
        X_encoded[cols_to_scale] = scaler.transform(X_encoded[cols_to_scale])

X_example = X_encoded.astype(float).copy()

print("\nX_example encoded shape:", X_example.shape)
print("Columns match train feature space:", list(X_example.columns) == list(X_train_all_cols))


# ============================================================
# 4) GATED PREDICT (CLASS -> REGRESSOR OF THAT CLASS)
# ============================================================

# 4.1 Class prediction (+ probabilities if available)
pred_class = xgb_cls_gated.predict(X_example)
pred_class_id = int(pred_class[0])

proba = None
if hasattr(xgb_cls_gated, "predict_proba"):
    proba = xgb_cls_gated.predict_proba(X_example)[0]
    proba = [float(x) for x in proba]

# 4.2 Regression prediction using the regressor of predicted class
if pred_class_id not in xgb_reg_by_class:
    raise KeyError(f"Regressor for predicted class {pred_class_id} not found in xgb_reg_by_class keys: {list(xgb_reg_by_class.keys())}")

reg_model = xgb_reg_by_class[pred_class_id]
pred_value = float(reg_model.predict(X_example)[0])

# 4.3 Force prediction inside the class interval (hard clamp)
low, high, _label = class_ranges[pred_class_id]
pred_value_clamped = float(np.clip(pred_value, low, high))

print("\n=== GATED PREDICTION ===")
print("Classe prédite:", pred_class_id, "|", class_names_fr.get(pred_class_id, pred_class_id))
print("Intervalle:", (low, high))
print("Note prédite (raw):", pred_value)
print("Note prédite (clamp intervalle):", pred_value_clamped)
if proba is not None:
    print("Probabilités par classe:", proba)


# ============================================================
# 5) SHAP EXPLANATIONS (TEXT-FRIENDLY: TOP FEATURES)
#   - Classifier: explain why it chose this class
#   - Regressor: explain why it chose this note (inside class)
# ============================================================

TOPK = 12  # number of features to send to LLM

def top_features_from_shap(values_1d, feature_names, topk=12):
    s = pd.Series(values_1d, index=feature_names)
    s_abs = s.abs().sort_values(ascending=False).head(topk)
    out = []
    for feat, absval in s_abs.items():
        out.append({
            "feature": str(feat),
            "shap_value": float(s.loc[feat]),
            "abs_shap": float(absval)
        })
    return out

# ---------- 5.1 Classifier SHAP (local on example) ----------
clf_explainer = shap.TreeExplainer(xgb_cls_gated)

# For multi-class, SHAP may return:
# - list of arrays (one array per class) OR
# - array with shape (n, features, classes)
shap_clf = clf_explainer.shap_values(X_example)

if isinstance(shap_clf, list):
    # list[class] -> (n, features)
    shap_clf_for_class = shap_clf[pred_class_id][0]
elif isinstance(shap_clf, np.ndarray) and shap_clf.ndim == 3:
    # (n, features, classes)
    shap_clf_for_class = shap_clf[0, :, pred_class_id]
else:
    # binary case: (n, features)
    shap_clf_for_class = shap_clf[0]

top_shap_classifier = top_features_from_shap(
    values_1d=shap_clf_for_class,
    feature_names=X_example.columns,
    topk=TOPK
)

# ---------- 5.2 Regressor SHAP (local on example) ----------
reg_explainer = shap.TreeExplainer(reg_model)
shap_reg = reg_explainer.shap_values(X_example)

# shap_reg: (n, features)
if isinstance(shap_reg, np.ndarray) and shap_reg.ndim == 2:
    shap_reg_1d = shap_reg[0]
else:
    shap_reg_1d = np.array(shap_reg).reshape(-1)

top_shap_regressor = top_features_from_shap(
    values_1d=shap_reg_1d,
    feature_names=X_example.columns,
    topk=TOPK
)

print("\nTop SHAP features (classifier, predicted class):")
for i, d in enumerate(top_shap_classifier, 1):
    print(f"{i:02d}. {d['feature']} | shap={d['shap_value']:.6f} | abs={d['abs_shap']:.6f}")

print("\nTop SHAP features (regressor of predicted class):")
for i, d in enumerate(top_shap_regressor, 1):
    print(f"{i:02d}. {d['feature']} | shap={d['shap_value']:.6f} | abs={d['abs_shap']:.6f}")


# ============================================================
# 6) BUILD THE LLM PAYLOAD (context + input + model outputs + SHAP)
# ============================================================

payload = {
    "project_context": PROJECT_CONTEXT_TEXT.strip(),
    "input_example_raw": example_row_raw,
    "gated_prediction": {
        "predicted_class_id": pred_class_id,
        "predicted_class_name_fr": class_names_fr.get(pred_class_id, str(pred_class_id)),
        "class_interval": [low, high],
        "predicted_note_raw": pred_value,
        "predicted_note_clamped": pred_value_clamped,
        "class_probabilities": proba
    },
    "shap_explanations": {
        "classifier_top_features_for_predicted_class": top_shap_classifier,
        "regressor_top_features_for_predicted_class": top_shap_regressor
    }
}

# Optional: print the JSON you will send (debug)
print("\nPayload keys:", list(payload.keys()))
print("Pred note:", payload["gated_prediction"]["predicted_note_clamped"])


# ============================================================
# 7) CALL OLLAMA LLM USING OpenAI CLIENT (LOCAL)
# ============================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # Ollama ignores key but OpenAI client requires one
)

system_msg = (
    "Tu es un expert ML + métier. "
    "Tu reçois une prédiction d'un modèle et ses explications (SHAP). "
    "Tu dois valider la décision et signaler incohérences / données manquantes. "
    "Réponds en français, clair, structuré."
)

user_msg = f"""
Voici le dossier complet (JSON) à analyser :

{json.dumps(payload, ensure_ascii=False, indent=2)}

Tâches:
1) Résume la décision du modèle (classe + note) et si c'est cohérent.
2) Explique les 5 facteurs principaux (à partir de SHAP) qui ont poussé la classe et la note.
3) Donne un verdict: OK / À revoir, avec 3 raisons max.
4) Propose 2 actions concrètes (ex: vérifier une donnée, compléter une info manquante).
"""

resp = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    temperature=0.2
)

llm_text = resp.choices[0].message.content
print("\n================ LLM RESPONSE ================\n")
print(llm_text)
```
