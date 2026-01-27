import pandas as pd
import numpy as np

# Create column "last_word" = last token in 'cadeau' split by commas
df["last_word"] = (
    df["cadeau"]
      .fillna("")                      # handle NaN
      .astype(str)
      .str.split(",")                  # split into list of tokens
      .apply(lambda lst: [w.strip() for w in lst if w.strip()])  # trim + remove empties
      .apply(lambda lst: lst[-1] if lst else np.nan)            # take last token
)

# df now has a new column: last_word
df[["cadeau", "last_word"]].head()