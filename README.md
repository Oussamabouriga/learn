import pandas as pd
import numpy as np

# Create column "last_word" = last token in 'cadeau' split by commas
df["last_word"] = (
    df["cadeau"]
      .fillna("")                                   # handle NaN
      .astype(str)
      .str.split(",")                               # split into list of tokens
      .apply(lambda lst: [w.strip() for w in lst if w.strip()])  # trim + remove empties
      .apply(lambda lst: lst[-1] if lst else np.nan)            # take last token
)

# If last_word == "APPLE", replace it with the value from column "gifts"
mask = df["last_word"].astype(str).str.upper().eq("APPLE")
df.loc[mask, "last_word"] = df.loc[mask, "gifts"]

df[["cadeau", "gifts", "last_word"]].head()