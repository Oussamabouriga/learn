missing_report = (
    df.isna()
      .sum()
      .to_frame("missing_count")
      .assign(missing_pct=lambda x: (x.missing_count / len(df) * 100).round(2))
      .sort_values("missing_pct", ascending=False)
)

missing_report



import numpy as np

missing_count = (
    df.replace(r"^\s*$", np.nan, regex=True)
      .isna()
      .sum()
)

missing_count


(df.select_dtypes(include="number") == 0).sum().sum()

(df.select_dtypes(include="number") == 0).sum()


import numpy as np

empty_count = (
    df.replace(r"^\s*$", np.nan, regex=True)
      .isna()
      .sum()
)

empty_count

