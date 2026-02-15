```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    Same call as before:
        plot_top_corr_bar(top_df, title="...", show_values=True)

    Input:
      - DataFrame with column 'corr' and index = feature names
      - or a Series (index=feature, values=corr)

    Output:
      - figure + axes

    Notes:
      - Always plots on [-1, 1] (correlation scale)
      - Adds square grid style (major + minor) like your screenshot
      - Prevents "Image size too large" by keeping annotations inside [-1,1]
    """

    # --- accept Series or DataFrame
    if isinstance(top_df, pd.Series):
        d = top_df.to_frame(name="corr").copy()
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

    # --- ensure numeric
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])
    if d.empty:
        print("No valid correlation values to plot.")
        return None

    # --- if values not in [-1, 1], scale just for plotting (keeps sign/ranking)
    max_abs = float(d["corr"].abs().max())
    corr_plot = d["corr"].astype(float).copy()
    if max_abs > 1.000001:
        corr_plot = corr_plot / max_abs
        print(f"[WARN] Values not in [-1,1] (max abs={max_abs:.2f}). Scaled to [-1,1] for plotting.")

    # --- sort negative -> positive
    d = d.assign(_corr_plot=corr_plot).sort_values("_corr_plot")

    # --- figure size (cap height)
    n = len(d)
    fig_h = min(18, max(4.5, 0.35 * n + 2))
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=110)

    y_labels = d.index.astype(str).tolist()
    y = np.arange(n)
    vals = d["_corr_plot"].values

    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)

    # correlation scale like before
    ax.set_xlim(-1, 1)
    ax.axvline(0, linewidth=1)

    # ---------------------------
    # ✅ GRID STYLE (square grid)
    # ---------------------------
    ax.set_axisbelow(True)

    # Major ticks on x for correlations
    ax.set_xticks(np.linspace(-1, 1, 9))             # -1, -0.75, ..., 1
    ax.xaxis.set_minor_locator(MultipleLocator(0.125))  # extra vertical lines

    # Major y ticks already set (one per bar)
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))    # extra horizontal lines

    # Major grid (visible)
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.8, alpha=0.35)
    # Minor grid (lighter)
    ax.grid(True, which="minor", axis="both", linestyle="-", linewidth=0.5, alpha=0.18)

    # Optional: background like seaborn whitegrid (comment out if you don't want it)
    ax.set_facecolor("#f7f7f7")

    ax.set_xlabel("correlation")
    ax.set_title(title)

    # --- value labels (clipped inside [-1,1] to avoid image-size crash)
    if show_values:
        for yi, v in enumerate(vals):
            if v >= 0:
                x_text = min(v + 0.03, 0.98)
                ha = "left"
            else:
                x_text = max(v - 0.03, -0.98)
                ha = "right"
            ax.text(x_text, yi, f"{v:.3f}", va="center", ha=ha, fontsize=9, clip_on=True)

    fig.subplots_adjust(left=0.32, right=0.98, top=0.90, bottom=0.08)
    plt.show()

    return fig, ax

```
