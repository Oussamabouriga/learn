```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names
            (also accepts a Series of correlations)
    """
    # accept Series or DataFrame
    if isinstance(top_df, pd.Series):
        d = top_df.to_frame(name="corr")
    elif isinstance(top_df, pd.DataFrame):
        if "corr" in top_df.columns:
            d = top_df[["corr"]].copy()
        elif top_df.shape[1] == 1:
            d = top_df.copy()
            d.columns = ["corr"]
        else:
            raise ValueError("top_df must have a 'corr' column.")
    else:
        raise TypeError("top_df must be a pandas DataFrame or Series.")

    # ensure numeric
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])
    if d.empty:
        print("No valid correlation values to plot.")
        return None

    # scale if values are not in [-1, 1]
    max_abs = float(d["corr"].abs().max())
    corr_plot = d["corr"].astype(float).copy()
    if max_abs > 1.000001:
        corr_plot = corr_plot / max_abs
        print(f"[WARN] Values not in [-1,1] (max abs={max_abs:.2f}). Scaled for plotting.")

    # sort negative -> positive
    d = d.assign(_corr_plot=corr_plot).sort_values("_corr_plot")

    n = len(d)
    fig_h = min(18, max(4.5, 0.35 * n + 2))
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=110)

    y_labels = d.index.astype(str).tolist()
    ax.barh(y_labels, d["_corr_plot"].values)

    ax.set_xlim(-1, 1)
    ax.axvline(0, linewidth=1)

    # ✅ GRID (FORCED VISIBLE)
    ax.set_axisbelow(True)
    ax.set_xticks(np.linspace(-1, 1, 9))
    ax.grid(True, axis="x", which="major", linestyle="--", linewidth=0.8, alpha=0.8)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    if show_values:
        vals = d["_corr_plot"].values
        for y, v in enumerate(vals):
            if v >= 0:
                x = min(v + 0.03, 0.98)
                ha = "left"
            else:
                x = max(v - 0.03, -0.98)
                ha = "right"
            ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    fig.subplots_adjust(left=0.32, right=0.98, top=0.90, bottom=0.08)
    plt.show()
    return fig, ax

```
