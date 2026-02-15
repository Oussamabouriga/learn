```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

def plot_top_corr_bar(
    top_df,
    title="Top correlations with target",
    show_values=True,
    wrap_width=18,
    max_label_len=40,
    dpi=90,
    figsize=(10, 6),
    xlim=(-1, 1),
    max_bars=30,
):
    """
    Plot a horizontal bar chart of correlations.

    Works with:
    - pd.DataFrame with a column named 'corr' (and optional 'abs_corr'), index = feature names
    - pd.Series where values are correlations, index = feature names
    - pd.DataFrame with ONE numeric column (will be used)

    Keeps the same way of calling:
        plot_top_corr_bar(top_num_delai_sinistre, title="...")

    Parameters
    ----------
    top_df : pd.DataFrame or pd.Series
    title : str
    show_values : bool
    wrap_width : int      wrap long labels to multiple lines
    max_label_len : int   truncate very long labels
    dpi : int             low dpi to avoid "image too large" in notebooks
    figsize : tuple       inches
    xlim : tuple or None  if None -> auto based on data
    max_bars : int        max number of bars to plot (prevents huge figures)
    """

    # -----------------------------
    # 1) Normalize input to a DataFrame with column 'corr'
    # -----------------------------
    if isinstance(top_df, pd.Series):
        d = top_df.to_frame(name="corr").copy()
    elif isinstance(top_df, pd.DataFrame):
        d = top_df.copy()

        if "corr" in d.columns:
            d = d[["corr"]].copy()
        else:
            # take first numeric column
            num_cols = d.select_dtypes(include="number").columns.tolist()
            if len(num_cols) == 0:
                raise ValueError("plot_top_corr_bar: no numeric column found.")
            d = d[[num_cols[0]]].rename(columns={num_cols[0]: "corr"}).copy()
    else:
        raise TypeError("plot_top_corr_bar: top_df must be a pandas DataFrame or Series.")

    # Make sure corr is numeric (FIX: pd.to_numeric, not np.to_numeric)
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])

    if d.empty:
        print("plot_top_corr_bar: no valid numeric values to plot.")
        return d

    # -----------------------------
    # 2) Safety: limit number of bars (avoid giant renders)
    # -----------------------------
    # If your input has many rows, keep only strongest absolute values
    if len(d) > max_bars:
        d = d.reindex(d["corr"].abs().sort_values(ascending=False).head(max_bars).index)

    # Sort so bars are ordered (negative -> positive)
    d = d.sort_values("corr")

    # -----------------------------
    # 3) Prepare labels (truncate + wrap)
    # -----------------------------
    labels = d.index.astype(str).tolist()

    def format_label(s: str) -> str:
        s = s.strip()
        if len(s) > max_label_len:
            s = s[: max_label_len - 3] + "..."
        if wrap_width is not None and wrap_width > 0:
            return "\n".join(wrap(s, width=wrap_width)) if len(s) > wrap_width else s
        return s

    labels = [format_label(s) for s in labels]

    # -----------------------------
    # 4) Decide x-limits
    # -----------------------------
    # If values are not in [-1, 1], don't force (-1, 1) (auto-scale instead)
    v = d["corr"].values
    max_abs = float(np.nanmax(np.abs(v)))

    if xlim is None:
        # Auto symmetrical range
        if max_abs == 0:
            lim = 1
        else:
            lim = max_abs * 1.1
        xlim_use = (-lim, lim)
    else:
        # If user passed (-1,1) but data clearly bigger, auto-scale to avoid squashing
        if (abs(xlim[0]) == 1 and abs(xlim[1]) == 1) and max_abs > 1.5:
            lim = max_abs * 1.1
            xlim_use = (-lim, lim)
        else:
            xlim_use = xlim

    # -----------------------------
    # 5) Plot (low DPI + fixed figsize to avoid "image too large")
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = np.arange(len(d))

    ax.barh(y, d["corr"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("correlation")

    # vertical line at 0 for readability
    ax.axvline(0, linewidth=1)

    # x-limits
    ax.set_xlim(xlim_use)

    # value labels
    if show_values:
        for yi, val in enumerate(d["corr"].values):
            if np.isnan(val):
                continue
            # offset depends on sign and range
            offset = 0.02 * (xlim_use[1] - xlim_use[0])
            x_text = val + (offset if val >= 0 else -offset)
            ha = "left" if val >= 0 else "right"
            ax.text(x_text, yi, f"{val:.3f}", va="center", ha=ha, fontsize=9)

    # Tight layout with padding to prevent huge canvas expansions
    fig.tight_layout(pad=1.0)
    plt.show()

    return d

```
