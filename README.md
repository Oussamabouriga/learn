```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names (like before)
            (also accepts a Series of correlations)

    This version is SAFE against the 'Image size too large' error by:
    - forcing plotted values into [-1, 1]
    - ensuring text annotations never go outside axis limits
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

    # --- ensure numeric
    d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
    d = d.dropna(subset=["corr"])

    if d.empty:
        print("No valid correlation values to plot.")
        return None

    # --- IMPORTANT: protect against wrong inputs (values not in [-1, 1])
    max_abs = float(d["corr"].abs().max())
    corr_plot = d["corr"].astype(float).copy()

    if max_abs > 1.000001:
        # If user passed raw values (not correlations), scale to [-1, 1] to avoid rendering crash
        # (keeps signs and relative magnitude)
        corr_plot = corr_plot / max_abs
        print(
            f"[WARN] 'corr' values are not in [-1,1] (max abs={max_abs:.2f}). "
            "It looks like you passed raw values instead of correlations. "
            "I scaled them to [-1,1] for plotting to avoid the image-size error."
        )

    # --- sort negative -> positive (like before)
    d = d.assign(_corr_plot=corr_plot).sort_values("_corr_plot")

    # --- build figure size safely (cap height)
    n = len(d)
    fig_h = min(18, max(4.5, 0.35 * n + 2))
    fig_w = 10

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)

    y_labels = d.index.astype(str).tolist()
    ax.barh(y_labels, d["_corr_plot"].values)

    ax.set_xlim(-1, 1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("correlation")
    ax.set_title(title)

    if show_values:
        # annotate using SAFE x positions inside [-1,1]
        vals = d["_corr_plot"].values
        for y, v in enumerate(vals):
            # keep label inside axis range
            if v >= 0:
                x = min(v + 0.03, 0.98)
                ha = "left"
            else:
                x = max(v - 0.03, -0.98)
                ha = "right"
            ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    # Avoid tight_layout explosion if labels are long
    fig.subplots_adjust(left=0.32, right=0.98, top=0.90, bottom=0.08)

    plt.show()
    return fig, ax



corr_mat = df_num.corr(method="spearman")
top_num_delai_Sinistre = top_corr_with_target(corr_mat, "delai_Sinistre", top_n=15)
plot_top_corr_bar(top_num_delai_Sinistre, title="...", show_values=True)
```
