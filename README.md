```


# 1) Convert selected 0 values to np.nan (so XGBoost sees them as missing values)

import numpy as np
import pandas as pd

# Example: column name is "col1"
# Replace only in this column (0 -> np.nan)
df["col1"] = pd.to_numeric(df["col1"], errors="coerce")  # make sure column is numeric
df["col1"] = df["col1"].replace(0, np.nan)

# If you want to do it for multiple columns:
cols_zero_as_nan = ["col1"]  # add more column names if needed
for col in cols_zero_as_nan:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace(0, np.nan)
```
