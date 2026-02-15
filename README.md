```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    Plot horizontal bars of correlations (negative -> positive), exactly like before.

    Parameters
    ----------
    top_df : pd.DataFrame or pd.Series
        - If DataFrame: must contain a column named 'corr' and index = feature names
        - If Series: values = corr, index = feature names
    title : str
    show_values : bool
    """

    # --- accept Series or DataFrame
    if isinstance(top_df, pd.Series):
        d = top_df.to_frame(name="corr")
    elif isinstance(top_df, pd.DataFrame):
        if "corr" in top_df.columns:
            d = top_df[["corr"]].copy()
        elif top_df.shape[1] == 1:
            d = top_df.copy()
            d.columns = ["corr"]
        else:
            raise ValueError("top_df must have a 'corr' column (or be a single-column DataFrame).")
    else:
        raise TypeError("top_df must be a pandas DataFrame or Series.")

    # --- ensure numeric corr (FIX: pd.to_numeric, not np.to_numeric)
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])

    if d.empty:
        print("No valid correlation values to plot.")
        return None

    # --- sort so bars are ordered (negative -> positive)
    d = d.sort_values("corr")

    # --- prepare labels (wrap + truncate to avoid gigantic figures)
    labels = d.index.astype(str).tolist()

    # hard limits to prevent "Image size ... is too large" errors
    max_label_len = 60
    wrap_width = 22

    clean_labels = []
    for s in labels:
        s = s.strip()
        if len(s) > max_label_len:
            s = s[: max_label_len - 1] + "…"
        s = fill(s, width=wrap_width)
        clean_labels.append(s)

    n = len(d)

    # dynamic height but capped (prevents huge pixel images)
    height = 2 + 0.35 * n
    height = max(4.5, min(height, 18))  # cap height
    width = 10

    fig, ax = plt.subplots(figsize=(width, height), dpi=110)

    ax.barh(clean_labels, d["corr"].values)

    # correlation scale like before
    ax.set_xlim(-1, 1)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    if show_values:
        for y, v in enumerate(d["corr"].values):
            # place text slightly to the right for positive, left for negative
            x = v + 0.03 if v >= 0 else v - 0.03
            ha = "left" if v >= 0 else "right"
            ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig, ax


```
