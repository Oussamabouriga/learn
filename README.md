```
# ============================================================
# NOTEBOOK: GATED MODEL (XGBoost Classifier -> XGBoost Regressors)
# + SHAP (global + local on the example)
# + Send everything to local Ollama LLM via OpenAI Python client (Ollama)
#
# PREREQUISITES (outside python):
#   1) Start Ollama server:
#        ollama serve
#   2) Ensure model is pulled / runnable:
#        ollama run llama3.1:latest
#
# Python installs (run once if needed):
#   pip install -U openai shap pandas numpy xgboost
# ============================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from openai import OpenAI

# ============================================================
# 0) USER CONFIG (EDIT ME)
# ============================================================

# --- A) Root of your project (folder that contains "models/")
# If your notebook is inside nps_final/, this will work:
PROJECT_ROOT = Path.cwd()

# If needed, hardcode it (recommended if you move notebooks):
# PROJECT_ROOT = Path("/Users/oussama bouriga/Documents/nps_final")

MODEL_NAME = "xgb_gated_optuna_v1"
GATED_MODEL_DIR = PROJECT_ROOT / "models" / "assembled_models" / MODEL_NAME

XGB_CLS_FILE = GATED_MODEL_DIR / "xgb_classifier.json"
META_FILE    = GATED_MODEL_DIR / "meta.json"
REG_DIR      = GATED_MODEL_DIR / "regressors_by_class"

# --- B) Ollama (OpenAI-compatible)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:latest"
TEMPERATURE = 0.2

# --- C) SHAP settings
SHAP_BACKGROUND_SIZE = 200   # use a small background for speed
SHAP_GLOBAL_SIZE = 400       # sample size for global plot (if you have X_test)
MAX_DISPLAY = 20

# --- D) Example row you want to test (RAW features)
# Put your real raw example here (same keys as your original raw dataset)
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
# 1) CHECK PATHS
# ============================================================
print("PROJECT_ROOT   =", PROJECT_ROOT)
print("GATED_MODEL_DIR=", GATED_MODEL_DIR)
print("Classifier exists:", XGB_CLS_FILE.exists())
print("Meta exists      :", META_FILE.exists())
print("Reg dir exists   :", REG_DIR.exists())

assert GATED_MODEL_DIR.exists(), f"GATED_MODEL_DIR not found: {GATED_MODEL_DIR}"
assert XGB_CLS_FILE.exists(), f"Missing file: {XGB_CLS_FILE}"
assert META_FILE.exists(), f"Missing file: {META_FILE}"
assert REG_DIR.exists(), f"Missing folder: {REG_DIR}"

# ============================================================
# 2) LOAD META + MODELS
# ============================================================
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

# meta should contain at least the encoded feature list
# We'll support multiple possible keys to avoid mismatch.
feature_cols = (
    meta.get("feature_cols")
    or meta.get("train_columns")
    or meta.get("X_train_columns")
    or meta.get("encoded_feature_names")
)
if feature_cols is None:
    raise ValueError("meta.json must contain feature columns list (feature_cols/train_columns/...)")

# class ranges if saved
class_ranges = meta.get("class_ranges", None)  # dict {class_id: [low, high, name]}
class_names_fr = meta.get("class_names_fr", None)  # dict {class_id: "label"}

# Load classifier + regressors from JSON
xgb_cls = xgb.XGBClassifier()
xgb_cls.load_model(str(XGB_CLS_FILE))

# Load regressors by class: xgb_regressor_class_0.json etc.
xgb_reg_by_class = {}
for p in sorted(REG_DIR.glob("xgb_regressor_class_*.json")):
    # class id from filename
    cls_id = int(p.stem.split("_")[-1])
    reg = xgb.XGBRegressor()
    reg.load_model(str(p))
    xgb_reg_by_class[cls_id] = reg

print("Loaded classifier + regressors:", sorted(xgb_reg_by_class.keys()))
print("Encoded feature count:", len(feature_cols))

# ============================================================
# 3) PREPARE EXAMPLE (encoded alignment)
# IMPORTANT: You must provide the encoded example.
# If you don't have a saved preprocessing pipeline, the simplest robust approach:
#   - You already had an encoded example before (X_new_encoded_no_te).
#   - Otherwise, you must rebuild the SAME encoding pipeline in this notebook.
#
# Here: we assume you are providing a RAW dict, so we convert it to a DataFrame,
# then we ALIGN to encoded columns using:
#   - missing -> 0
#   - extra -> dropped
#
# This ONLY works if your example_row_raw is ALREADY in encoded form.
# If your example is raw (categoricals not one-hot/freq), you need the pipeline.
# ============================================================

# If you already have an encoded example dataframe in memory, use it:
X_example = None
if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()
    print("Using existing X_new_encoded_no_te (already encoded).")
else:
    # fallback: treat the dict as encoded (only safe if keys match encoded columns)
    X_example = pd.DataFrame([example_row_raw]).copy()
    print("Using example_row_raw as-is (WARNING: this must already be encoded).")

# --- Align columns to training encoded columns
missing_cols = [c for c in feature_cols if c not in X_example.columns]
for c in missing_cols:
    X_example[c] = 0.0

extra_cols = [c for c in X_example.columns if c not in feature_cols]
if extra_cols:
    X_example = X_example.drop(columns=extra_cols)

X_example = X_example[feature_cols].copy()

# force numeric
for c in X_example.columns:
    X_example[c] = pd.to_numeric(X_example[c], errors="coerce").fillna(0.0)

print("Example aligned shape:", X_example.shape)

# ============================================================
# 4) GATED PREDICTION (class -> regressor -> clipped to interval)
# ============================================================

def gated_predict_one(X_one: pd.DataFrame):
    # class prediction
    pred_class = int(xgb_cls.predict(X_one)[0])

    # regression inside class
    if pred_class not in xgb_reg_by_class:
        raise ValueError(f"No regressor found for class {pred_class}")

    pred_value = float(xgb_reg_by_class[pred_class].predict(X_one)[0])

    # enforce interval if available
    if class_ranges is not None and str(pred_class) in class_ranges:
        low, high, _name = class_ranges[str(pred_class)]
        pred_value = float(np.clip(pred_value, low, high))
        interval = (low, high)
    elif class_ranges is not None and pred_class in class_ranges:
        low, high, _name = class_ranges[pred_class]
        pred_value = float(np.clip(pred_value, low, high))
        interval = (low, high)
    else:
        interval = None

    return pred_class, pred_value, interval

pred_class, pred_note, interval = gated_predict_one(X_example)

print("\n=== GATED PREDICTION ===")
print("Classe prédite:", pred_class, "|", (class_names_fr.get(str(pred_class)) if isinstance(class_names_fr, dict) else None) or (class_names_fr.get(pred_class) if isinstance(class_names_fr, dict) else None) or "")
print("Intervalle:", interval)
print("Note prédite:", pred_note)

# ============================================================
# 5) SHAP (Classifier + regressor of predicted class)
# ============================================================

# Background for SHAP: use a small random background from the example itself if no X_test exists.
# Best practice: pass a real background sample from your training/test encoded data.
if "X_test_encoded_no_te" in globals():
    X_bg = X_test_encoded_no_te.sample(min(SHAP_BACKGROUND_SIZE, len(X_test_encoded_no_te)), random_state=42).copy()
    # align to feature_cols
    X_bg = X_bg.reindex(columns=feature_cols, fill_value=0.0)
    X_bg = X_bg.apply(pd.to_numeric, errors="coerce").fillna(0.0)
else:
    # fallback background = repeated example (works but less meaningful)
    X_bg = pd.concat([X_example]*min(50, SHAP_BACKGROUND_SIZE), ignore_index=True)

# ---- Classifier SHAP
expl_cls = shap.TreeExplainer(xgb_cls, data=X_bg, feature_perturbation="tree_path_dependent")
sv_cls = expl_cls.shap_values(X_example)

# Multi-class: shap_values can be list (per class) or ndarray (n, p, k)
# We'll unify to "per-class array".
if isinstance(sv_cls, list):
    sv_cls_by_class = sv_cls
elif isinstance(sv_cls, np.ndarray) and sv_cls.ndim == 3:
    # (n, p, k) -> list k of (n,p)
    sv_cls_by_class = [sv_cls[:, :, k] for k in range(sv_cls.shape[2])]
else:
    # binary: (n,p)
    sv_cls_by_class = [sv_cls]

# Global plot for predicted class (bar)
cls_for_shap = pred_class if pred_class < len(sv_cls_by_class) else 0
sv_cls_mat = sv_cls_by_class[cls_for_shap]

# some SHAP versions return an extra bias column -> drop if needed
if sv_cls_mat.shape[1] == X_example.shape[1] + 1:
    sv_cls_mat = sv_cls_mat[:, :-1]

print(f"\nSHAP global (classifier) — classe: {cls_for_shap}")
shap.summary_plot(sv_cls_mat, X_example, plot_type="bar", show=True, max_display=MAX_DISPLAY)

print("\nSHAP local (classifier) — example")
shap.plots.waterfall(
    shap.Explanation(
        values=sv_cls_mat[0],
        base_values=(expl_cls.expected_value[cls_for_shap] if isinstance(expl_cls.expected_value, (list, np.ndarray)) else expl_cls.expected_value),
        data=X_example.iloc[0],
        feature_names=X_example.columns
    ),
    max_display=MAX_DISPLAY
)

# ---- Regressor SHAP (for predicted class regressor)
reg_model = xgb_reg_by_class[pred_class]
expl_reg = shap.TreeExplainer(reg_model, data=X_bg, feature_perturbation="tree_path_dependent")
sv_reg = expl_reg.shap_values(X_example)

if isinstance(sv_reg, np.ndarray) and sv_reg.ndim == 2 and sv_reg.shape[1] == X_example.shape[1] + 1:
    sv_reg = sv_reg[:, :-1]

print(f"\nSHAP global (regressor class {pred_class})")
shap.summary_plot(sv_reg, X_example, plot_type="bar", show=True, max_display=MAX_DISPLAY)

print(f"\nSHAP local (regressor class {pred_class}) — example")
shap.plots.waterfall(
    shap.Explanation(
        values=sv_reg[0],
        base_values=expl_reg.expected_value,
        data=X_example.iloc[0],
        feature_names=X_example.columns
    ),
    max_display=MAX_DISPLAY
)

# Also get top features for LLM context
def top_features_from_shap(values_row, cols, k=10):
    abs_vals = np.abs(values_row)
    idx = np.argsort(abs_vals)[::-1][:k]
    return [(cols[i], float(values_row[i])) for i in idx]

top_cls = top_features_from_shap(sv_cls_mat[0], list(X_example.columns), k=10)
top_reg = top_features_from_shap(sv_reg[0], list(X_example.columns), k=10)

# ============================================================
# 6) SEND TO OLLAMA LLM (OpenAI-compatible)
# ============================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # required by client, ignored by Ollama
)

system_msg = """Tu es un assistant expert en data science.
Tu dois valider / critiquer la prédiction d'un système ML "gated" :
- un classifieur XGBoost choisit une classe de satisfaction
- puis un régresseur XGBoost spécialisé prédit une note dans l'intervalle de cette classe
Tu dois expliquer si la prédiction te semble cohérente, et pourquoi, en te basant sur les features importantes (SHAP).
Réponds en français, clair, structuré, avec recommandations."""
# You can replace the user context below with your own long context text.
user_context = "Contexte métier: (à compléter par toi) ..."

payload = {
    "prediction": {
        "predicted_class": pred_class,
        "predicted_note": pred_note,
        "interval": interval,
        "class_label": (class_names_fr.get(str(pred_class)) if isinstance(class_names_fr, dict) else None) or (class_names_fr.get(pred_class) if isinstance(class_names_fr, dict) else None)
    },
    "example_row_used": example_row_raw,  # what you provided
    "top_shap_classifier": top_cls,
    "top_shap_regressor": top_reg
}

user_msg = f"""{user_context}

Voici le résultat du modèle gated + explications SHAP.
Analyse et dis si tu es d'accord ou non, et quelles infos supplémentaires seraient utiles.

DATA (json):
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

resp = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    temperature=TEMPERATURE
)

print("\n================ LLM RESPONSE ================\n")
print(resp.choices[0].message.content)
```
