```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names
            OR a Series (index=feature, values=corr)
    """

    # ---------- accept Series or DataFrame ----------
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

    # FIX: use pandas.to_numeric (NOT numpy.to_numeric)
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])
    if d.empty:
        print("No valid correlation values to plot.")
        return None

    # If someone passed non-correlation magnitudes, normalize for plotting (keeps sign/order)
    max_abs = float(d["corr"].abs().max())
    corr_plot = d["corr"].astype(float).copy()
    if max_abs > 1.000001:
        corr_plot = corr_plot / max_abs

    d = d.assign(_corr_plot=corr_plot).sort_values("_corr_plot")

    # ---------- build plot ----------
    n = len(d)
    fig_h = min(12, max(4.5, 0.35 * n + 2))
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=110)

    # EXACT style like your screenshot (grey background + white grid)
    ax.set_facecolor("#E5E5E5")        # ggplot-like panel
    fig.patch.set_facecolor("white")   # outside area white

    # blue used in your screenshot
    BLUE = "#1f77b4"

    y = np.arange(n)
    y_labels = d.index.astype(str).tolist()
    v = d["_corr_plot"].values

    ax.barh(y, v, color=BLUE, edgecolor=BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)

    ax.set_xlim(-1, 1)

    # blue zero line (NOT black, NOT red)
    ax.axvline(0, color=BLUE, linewidth=1)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    # white grid like the image
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color="white", linewidth=1)

    # remove spines (matches screenshot feel)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # values in black (no red)
    if show_values:
        for yi, val in enumerate(v):
            if val >= 0:
                x_text = min(val + 0.03, 0.98)
                ha = "left"
            else:
                x_text = max(val - 0.03, -0.98)
                ha = "right"
            ax.text(x_text, yi, f"{val:.3f}", va="center", ha=ha,
                    fontsize=9, color="black", clip_on=True)

    plt.tight_layout()
    plt.show()
    return fig, ax
```
