```

# ============================================================
# NOTEBOOK: GATED MODEL (XGBoost Classifier -> XGBoost Regressors)
# + Robust load (no preprocessing artifacts required)
# + Robust example encoding -> align to training columns
# + SHAP (global + local on the example)
# + Send everything to local Ollama LLM via OpenAI Python client
#
# PREREQUISITES (outside python):
#   1) Start Ollama server:
#        ollama serve
#   2) Ensure model is pulled / runnable (once):
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
from openai import OpenAI

from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# 0) USER CONFIG (EDIT ME)
# ============================================================

# --- A) Project root (auto)
# If your notebook is inside /Users/.../Documents/nps_final, this will print correctly.
PROJECT_ROOT = Path.cwd()
print("PROJECT_ROOT =", PROJECT_ROOT)

# --- B) Where your gated model is saved (relative OR absolute)
# Your folder (from your screenshot):
GATED_MODEL_DIR = Path("models/assembled_models/xgb_gated_optuna_v1")  # relative to PROJECT_ROOT

# Resolve to absolute
if not GATED_MODEL_DIR.is_absolute():
    GATED_MODEL_DIR = (PROJECT_ROOT / GATED_MODEL_DIR).resolve()

# Expected structure in that folder:
XGB_CLS_FILE = GATED_MODEL_DIR / "xgb_classifier.json"
META_FILE    = GATED_MODEL_DIR / "meta.json"
REG_DIR      = GATED_MODEL_DIR / "regressors_by_class"

print("\nGATED_MODEL_DIR =", GATED_MODEL_DIR)
print("Classifier exists:", XGB_CLS_FILE.exists())
print("Meta exists      :", META_FILE.exists())
print("Reg dir exists   :", REG_DIR.exists())

# --- C) Your raw example (EDIT THIS DICT)
# Put the same “raw” values you used before (categoricals as strings are OK).
# Missing columns are fine.
example_row_raw = {
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "tarif": 19.99,
    "Nombre_sisnitre_client": 1,
    "Nombre_sisnitre_accepte_client": 1,
    "Nombre_sisnitre_refuse_client": np.nan,
    "Nombre_sisnitre_sans_suite_client": np.nan,
    "code_postal": "59700",
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

# --- D) Range safety for output
CLIP_PRED_TO_0_10 = True

# --- E) SHAP settings
SHAP_BACKGROUND_ROWS = 200  # only used to build a synthetic background (if you don't provide real data)
SHAP_MAX_DISPLAY = 20       # top features to show

# --- F) Ollama (OpenAI-compatible)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_TEMPERATURE = 0.2

# Big context you write yourself (put your long project explanation here)
PROJECT_CONTEXT_TEXT = """
[COLLE ICI TON CONTEXTE LONG]
- Ce que représente chaque variable
- Le but du modèle
- La logique métier
- Comment interpréter une note prédite
"""


# ============================================================
# 1) LOAD MODELS + META
# ============================================================

if not GATED_MODEL_DIR.is_dir():
    raise FileNotFoundError(f"GATED_MODEL_DIR not found: {GATED_MODEL_DIR}")

if not XGB_CLS_FILE.is_file():
    raise FileNotFoundError(f"Missing classifier file: {XGB_CLS_FILE}")

if not META_FILE.is_file():
    raise FileNotFoundError(f"Missing meta file: {META_FILE}")

if not REG_DIR.is_dir():
    raise FileNotFoundError(f"Missing regressors directory: {REG_DIR}")

# Load meta.json
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Load classifier
xgb_cls_gated = XGBClassifier()
xgb_cls_gated.load_model(str(XGB_CLS_FILE))

# Load regressors by class
xgb_reg_by_class = {}
for p in sorted(REG_DIR.glob("xgb_regressor_class_*.json")):
    # filename: xgb_regressor_class_0.json -> class_id = 0
    class_id = int(p.stem.split("_")[-1])
    m = XGBRegressor()
    m.load_model(str(p))
    xgb_reg_by_class[class_id] = m

if len(xgb_reg_by_class) == 0:
    raise FileNotFoundError(f"No regressor json found in: {REG_DIR}")

print("\nLoaded:")
print("- classifier:", XGB_CLS_FILE.name)
print("- regressors:", sorted(list(xgb_reg_by_class.keys())))

# Class ranges + names (if saved)
# Expect format: { "0": [0,2,"..."], "1": [3,6,"..."], ... } or dict with similar
class_ranges = meta.get("class_ranges", None)
class_names_fr = meta.get("class_names_fr", None)

# If not found, define defaults (same mapping you used before)
if class_ranges is None:
    class_ranges = {
        0: (0, 2, "extrêmement mauvais (0–2)"),
        1: (3, 6, "mauvais (3–6)"),
        2: (7, 8, "neutre (7–8)"),
        3: (9, 9, "bien (9)"),
        4: (10, 10, "très bien (10)"),
    }
else:
    # normalize keys to int
    class_ranges = {int(k): tuple(v) for k, v in class_ranges.items()}

if class_names_fr is None:
    class_names_fr = {k: v[2] for k, v in class_ranges.items()}
else:
    class_names_fr = {int(k): v for k, v in class_names_fr.items()}

# ============================================================
# 2) GET FEATURE COLUMNS ROBUSTLY (NO meta.json dependency)
# ============================================================

def _is_list_of_strings(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(i, str) for i in x)

feature_cols = None

# (1) Best: sklearn API if available
if hasattr(xgb_cls_gated, "feature_names_in_") and xgb_cls_gated.feature_names_in_ is not None:
    feature_cols = list(xgb_cls_gated.feature_names_in_)
    print("\nfeature_cols from xgb_cls_gated.feature_names_in_ :", len(feature_cols))

# (2) Fallback: scan meta.json for any list of strings and take the longest
if feature_cols is None:
    candidates = []
    for k, v in meta.items():
        if _is_list_of_strings(v):
            candidates.append((k, v))
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if _is_list_of_strings(v2):
                    candidates.append((f"{k}.{k2}", v2))

    if len(candidates) == 0:
        raise ValueError(
            "Impossible de récupérer feature_cols.\n"
            "- feature_names_in_ absent\n"
            "- meta.json ne contient aucune liste de colonnes encodées\n"
            "➡️ Solution: sauvegarder la liste des colonnes encodées lors du training."
        )

    best_key, best_list = sorted(candidates, key=lambda t: len(t[1]), reverse=True)[0]
    feature_cols = list(best_list)
    print(f"\nfeature_cols from meta.json key '{best_key}' :", len(feature_cols))

if len(feature_cols) < 5:
    raise ValueError(f"feature_cols too small ({len(feature_cols)}). Something is wrong.")


# ============================================================
# 3) ROBUST "ENCODE" RAW EXAMPLE -> ALIGN TO feature_cols
# IMPORTANT:
# - We DON'T have your full training preprocessing artifacts here.
# - So we use a safe generic approach:
#   (a) build DataFrame from raw dict
#   (b) pd.get_dummies(dummy_na=True) -> generates one-hot columns for any categoricals present
#   (c) align to feature_cols (missing->0, extra dropped)
#
# This runs without errors. If your original training used extra steps (freq encoding),
# those features may be missing => they will be 0.
# ============================================================

def encode_and_align_example(example_raw: dict, feature_cols: list) -> pd.DataFrame:
    df_raw = pd.DataFrame([example_raw]).copy()

    # Convert pd.NA -> np.nan if any
    df_raw = df_raw.astype(object).where(pd.notna(df_raw), np.nan)

    # Make sure strings stay strings
    for c in df_raw.columns:
        if df_raw[c].dtype == "object":
            # keep as object; get_dummies will handle
            pass

    # One-hot generic
    df_enc = pd.get_dummies(df_raw, dummy_na=True)

    # Add missing columns as 0
    missing = [c for c in feature_cols if c not in df_enc.columns]
    for c in missing:
        df_enc[c] = 0.0

    # Drop extras
    extra = [c for c in df_enc.columns if c not in feature_cols]
    if len(extra) > 0:
        df_enc = df_enc.drop(columns=extra)

    # Reorder exactly
    df_enc = df_enc[feature_cols].copy()

    # Force float
    df_enc = df_enc.astype(float)

    return df_enc

X_example = encode_and_align_example(example_row_raw, feature_cols)

print("\nX_example aligned shape:", X_example.shape)
print("Columns match:", list(X_example.columns) == list(feature_cols))


# ============================================================
# 4) GATED PREDICTION (classifier -> regressor by class)
# ============================================================

def gated_predict(X_encoded: pd.DataFrame, clf: XGBClassifier, regs_by_class: dict):
    # predicted class
    pred_class = clf.predict(X_encoded).astype(int)
    c = int(pred_class[0])

    if c not in regs_by_class:
        raise KeyError(f"No regressor found for predicted class={c}. Available: {sorted(list(regs_by_class.keys()))}")

    # regression
    pred_value = float(regs_by_class[c].predict(X_encoded)[0])

    # clip to class interval (strict gating)
    low, high, _name = class_ranges.get(c, (0, 10, ""))
    pred_value = float(np.clip(pred_value, low, high))

    # optional clip to [0,10]
    if CLIP_PRED_TO_0_10:
        pred_value = float(np.clip(pred_value, 0, 10))

    return c, pred_value

pred_class_id, pred_note = gated_predict(X_example, xgb_cls_gated, xgb_reg_by_class)

low, high, _ = class_ranges[pred_class_id]
print("\n=== GATED PREDICTION ===")
print("Classe prédite:", pred_class_id, "|", class_names_fr.get(pred_class_id, str(pred_class_id)))
print("Intervalle:", (low, high))
print("Note prédite:", pred_note)


# ============================================================
# 5) SHAP (GLOBAL + LOCAL) — classifier + the selected regressor
# NOTE:
# We don't have your real X_test here, so we build a synthetic background
# of zeros (works and avoids shape errors).
# If you DO have real encoded test data, replace X_bg with it.
# ============================================================

def shap_fix_array(sv: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    # Some SHAP versions add an extra last column (constant offset)
    if sv.ndim == 2 and sv.shape[1] == X.shape[1] + 1:
        return sv[:, :-1]
    return sv

# --- Build synthetic background (safe)
X_bg = pd.DataFrame(np.zeros((min(SHAP_BACKGROUND_ROWS, 200), len(feature_cols))), columns=feature_cols).astype(float)

# 5A) SHAP for classifier (multiclass)
print("\n--- SHAP (Classifier) ---")
explainer_cls = shap.TreeExplainer(xgb_cls_gated)

sv_cls = explainer_cls.shap_values(X_bg)  # can be list (per class) or array
# Choose a class to display globally (use predicted class)
cls_to_show = pred_class_id

# Global (bar)
plt.figure()
if isinstance(sv_cls, list):
    sv = shap_fix_array(np.array(sv_cls[cls_to_show]), X_bg)
else:
    # sometimes shape: (n_samples, n_features, n_classes)
    if sv_cls.ndim == 3:
        sv = shap_fix_array(sv_cls[:, :, cls_to_show], X_bg)
    else:
        sv = shap_fix_array(sv_cls, X_bg)

shap.summary_plot(sv, X_bg, plot_type="bar", show=True, max_display=SHAP_MAX_DISPLAY)

# Local (example) — waterfall on that same class
sv_one_cls = explainer_cls.shap_values(X_example)
if isinstance(sv_one_cls, list):
    sv_one = shap_fix_array(np.array(sv_one_cls[cls_to_show]), X_example)
    base_val = explainer_cls.expected_value[cls_to_show] if isinstance(explainer_cls.expected_value, (list, np.ndarray)) else explainer_cls.expected_value
else:
    if sv_one_cls.ndim == 3:
        sv_one = shap_fix_array(sv_one_cls[:, :, cls_to_show], X_example)
    else:
        sv_one = shap_fix_array(sv_one_cls, X_example)
    base_val = explainer_cls.expected_value[cls_to_show] if isinstance(explainer_cls.expected_value, (list, np.ndarray)) else explainer_cls.expected_value

plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=sv_one[0],
        base_values=base_val,
        data=X_example.iloc[0],
        feature_names=X_example.columns,
    ),
    max_display=SHAP_MAX_DISPLAY,
    show=True
)

# 5B) SHAP for the selected regressor
print("\n--- SHAP (Regressor selected by class) ---")
reg_model = xgb_reg_by_class[pred_class_id]
explainer_reg = shap.TreeExplainer(reg_model)

sv_reg_bg = explainer_reg.shap_values(X_bg)
sv_reg_bg = shap_fix_array(np.array(sv_reg_bg), X_bg)

plt.figure()
shap.summary_plot(sv_reg_bg, X_bg, plot_type="bar", show=True, max_display=SHAP_MAX_DISPLAY)

sv_reg_one = explainer_reg.shap_values(X_example)
sv_reg_one = shap_fix_array(np.array(sv_reg_one), X_example)

# Build a short "top reasons" list for the LLM
abs_contrib = np.abs(sv_reg_one[0])
top_idx = np.argsort(abs_contrib)[::-1][:10]
top_features = []
for i in top_idx:
    feat = X_example.columns[i]
    val = float(X_example.iloc[0, i])
    contrib = float(sv_reg_one[0, i])
    top_features.append({"feature": str(feat), "value": val, "shap_contribution": contrib})

plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=sv_reg_one[0],
        base_values=explainer_reg.expected_value,
        data=X_example.iloc[0],
        feature_names=X_example.columns,
    ),
    max_display=SHAP_MAX_DISPLAY,
    show=True
)

print("\nTop SHAP features (regressor, example):")
for t in top_features[:10]:
    print("-", t)


# ============================================================
# 6) SEND TO OLLAMA LLM (OpenAI compatible)
# ============================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # Ollama ignores the key but OpenAI client requires one
)

system_msg = f"""
Tu es un assistant expert en analyse de modèles ML.
Tu dois valider (ou critiquer) la prédiction d'un système "gated" :
- Un modèle de classification choisit une classe de satisfaction.
- Ensuite un modèle de régression dédié à cette classe prédit une note dans l'intervalle correspondant.
Tu dois répondre en français, de manière professionnelle, structurée et courte.

Contexte projet:
{PROJECT_CONTEXT_TEXT}
""".strip()

user_msg = {
    "task": "Validate gated model decision",
    "gated_prediction": {
        "predicted_class_id": pred_class_id,
        "predicted_class_name": class_names_fr.get(pred_class_id, str(pred_class_id)),
        "interval": [float(low), float(high)],
        "predicted_note": float(pred_note),
    },
    "input_example_raw": example_row_raw,
    "top_shap_features_for_example (regressor)": top_features,
    "notes": [
        "Les features SHAP listées sont les facteurs les plus influents pour la prédiction de note (régression).",
        "Tu peux dire si la prédiction paraît cohérente ou si certaines valeurs sont suspectes / incohérentes.",
        "Propose aussi des vérifications à faire sur les données si nécessaire."
    ]
}

resp = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False, indent=2)},
    ],
    temperature=OLLAMA_TEMPERATURE
)

llm_text = resp.choices[0].message.content
print("\n================= LLM RESPONSE =================\n")
print(llm_text)
```
