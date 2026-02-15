```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with at least column 'corr' (index = feature names)
            (optionally 'abs_corr' can exist; it's ignored for plotting)
    """

    # --- accept Series or DataFrame ---
    if isinstance(top_df, pd.Series):
        top_df = top_df.to_frame(name="corr")

    if not isinstance(top_df, pd.DataFrame):
        raise TypeError("top_df must be a pandas DataFrame or Series.")

    if "corr" not in top_df.columns:
        raise ValueError("top_df must contain a column named 'corr'.")

    # --- make sure corr is numeric ---
    dfp = top_df.copy()
    dfp["corr"] = pd.to_numeric(dfp["corr"], errors="coerce")
    dfp = dfp.dropna(subset=["corr"])

    if dfp.empty:
        print("plot_top_corr_bar: nothing to plot (empty after cleaning).")
        return

    # order negative -> positive (like your screenshot)
    dfp = dfp.sort_values("corr")

    # style like your screenshot: gray background + white grid
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    labels = dfp.index.astype(str).tolist()
    values = dfp["corr"].values.astype(float)

    ax.barh(labels, values)

    # correlation axis bounds
    ax.set_xlim(-1, 1)

    # vertical line at 0
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    # grid style (white grid on gray background, like ggplot)
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", linewidth=0.5, alpha=0.35)
    ax.minorticks_on()

    # IMPORTANT: remove numeric annotations completely
    # (we keep show_values param for compatibility, but we don't draw values)
    # If you still want something here later (like rank numbers), tell me.

    plt.tight_layout()
    plt.show()
```
