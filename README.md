```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names
            OR a Series (index=feature, values=corr)
    """

    # --- ggplot style like your screenshot (grey background + white grid)
    with plt.style.context("ggplot"):

        # accept Series or DataFrame
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

        # numeric conversion (THIS fixes your np.to_numeric error)
        d["corr"] = pd.to_numeric(d["corr"], errors="coerce")
        d = d.dropna(subset=["corr"])
        if d.empty:
            print("No valid correlation values to plot.")
            return None

        # scale if someone accidentally passes non-correlation values
        max_abs = float(d["corr"].abs().max())
        vals = d["corr"].astype(float).copy()
        if max_abs > 1.000001:
            vals = vals / max_abs  # keep sign/rank, just normalize for plot

        d = d.assign(_corr_plot=vals).sort_values("_corr_plot")

        # figure size (safe)
        n = len(d)
        fig_h = min(12, max(4.5, 0.35 * n + 2))
        fig, ax = plt.subplots(figsize=(12, fig_h), dpi=110)

        y_labels = d.index.astype(str).tolist()
        y = np.arange(n)
        v = d["_corr_plot"].values

        ax.barh(y, v)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)

        ax.set_xlim(-1, 1)
        ax.axvline(0, linewidth=1)

        ax.set_xlabel("correlation")
        ax.set_title(title)

        # IMPORTANT: ggplot-style grid like your screenshot
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both")  # ggplot already sets the look

        # values
        if show_values:
            for yi, val in enumerate(v):
                if val >= 0:
                    x_text = min(val + 0.03, 0.98)
                    ha = "left"
                else:
                    x_text = max(val - 0.03, -0.98)
                    ha = "right"
                ax.text(x_text, yi, f"{val:.3f}", va="center", ha=ha, fontsize=9, clip_on=True)

        plt.tight_layout()
        plt.show()

        return fig, ax
```
