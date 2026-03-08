```
import os
from pathlib import Path

nb_path = Path("/Users/oussama bouriga/Documents/nps_final/llm.ipynb")
os.chdir(nb_path.parent)  # le dossier contenant le notebook
print("CWD =", os.getcwd())

```
