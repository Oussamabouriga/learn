```
from pathlib import Path
import os

MODEL_SUBPATH = Path("models") / "assembled_models" / "xgb_gated_optuna_v1"

# Cherche le dossier dans cwd + parents
cwd = Path(os.getcwd()).resolve()
candidates = [cwd] + list(cwd.parents)

found_dir = None
for base in candidates:
    p = base / MODEL_SUBPATH
    if p.is_dir():
        found_dir = p
        break

if found_dir is None:
    raise FileNotFoundError(
        f"Impossible de trouver {MODEL_SUBPATH} depuis:\n" +
        "\n".join(str(c) for c in candidates[:6]) + "\n...\n" +
        "➡️ Solution: mets ton notebook au niveau du projet OU fais os.chdir('.../ton_projet')."
    )

GATED_MODEL_DIR = str(found_dir)
print("✅ GATED_MODEL_DIR =", GATED_MODEL_DIR)



from pathlib import Path

GATED_MODEL_DIR = Path(GATED_MODEL_DIR)

XGB_CLS_FILE = GATED_MODEL_DIR / "xgb_classifier.json"
META_FILE    = GATED_MODEL_DIR / "meta.json"
REG_DIR      = GATED_MODEL_DIR / "regressors_by_class"

print("Classifier exists:", XGB_CLS_FILE.exists())
print("Meta exists      :", META_FILE.exists())
print("Reg dir exists   :", REG_DIR.exists())


import os
os.chdir("/CHEMIN/ABSOLU/VERS/TON/PROJET")
print("CWD =", os.getcwd())

```
