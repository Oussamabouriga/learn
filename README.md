```

import pandas as pd

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\[\]<>]", "_", regex=True)        # replace [ ] < >
        .str.replace(r"\s+", "_", regex=True)            # spaces -> _
        .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True) # other weird chars -> _
        .str.replace(r"_+", "_", regex=True)             # collapse ___ -> _
        .str.strip("_")
    )
    return df

# ✅ sanitize once (AFTER encoding + align)
X_train_encoded_no_te = sanitize_columns(X_train_encoded_no_te)
X_test_encoded_no_te  = sanitize_columns(X_test_encoded_no_te)

# (optional) quick check
bad = [c for c in X_train_encoded_no_te.columns if any(ch in c for ch in ["[", "]", "<", ">"])]
print("Bad columns left:", bad[:5])
```
