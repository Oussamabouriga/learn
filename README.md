```
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    Compatible with your old call:
        plot_top_corr_bar(top_df, title="...", show_values=True)

    Expects:
      - DataFrame with column 'corr' and index = feature names
      - OR DataFrame with a single numeric column
      - OR Series
    """

    if top_df is None or len(top_df) == 0:
        print("No data to plot.")
        return None

    # ---- accept Series or DataFrame
    if hasattr(top_df, "to_frame") and not hasattr(top_df, "columns"):
        d = top_df.to_frame(name="corr").copy()
    else:
        d = top_df.copy()

    # ---- choose the column to plot
    if "corr" in d.columns:
        col = "corr"
    else:
        # fallback: first numeric column
        num_cols = [c for c in d.columns if np.issubdtype(d[c].dtype, np.number)]
        if len(num_cols) == 0:
            # try coerce the first column
            col = d.columns[0]
            d[col] = np.to_numeric(d[col], errors="coerce")
        else:
            col = num_cols[0]

    d = d[[col]].rename(columns={col: "corr"}).copy()
    d["corr"] = np.to_numeric(d["corr"], errors="coerce")
    d = d.dropna()

    if d.empty:
        print("No numeric values to plot.")
        return None

    # ---- sort for nice order
    d = d.sort_values("corr")

    # ---- safe labels (wrap + truncate)
    labels = d.index.astype(str).tolist()
    labels = ["\n".join(textwrap.wrap(s, width=16)) for s in labels]
    labels = [(s if len(s) <= 30 else s[:29] + "…") for s in labels]

    vals = d["corr"].to_numpy()
    y = np.arange(len(vals))

    # ---- choose xlim:
    # if it looks like real correlation -> fixed [-1,1]
    absmax = float(np.nanmax(np.abs(vals)))
    if absmax <= 1.2:
        xmin, xmax = -1.0, 1.0
    else:
        # not correlation values (like your millions) -> autoscale safely
        pad = 0.05 * absmax
        xmin, xmax = -absmax - pad, absmax + pad

    # ---- plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=90)
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("correlation")
    ax.set_title(title)

    # ---- values text: ALWAYS keep inside axes (prevents giant canvas)
    if show_values:
        rng = (xmax - xmin) if (xmax > xmin) else 1.0
        off = 0.02 * rng

        for yi, v in enumerate(vals):
            # desired position near bar end
            x_text = v + (off if v >= 0 else -off)

            # clamp inside axes
            x_text = min(max(x_text, xmin + off), xmax - off)

            ha = "left" if v >= 0 else "right"
            ax.text(
                x_text, yi, f"{v:.3f}",
                va="center", ha=ha, fontsize=9,
                clip_on=True  # critical: do not expand bbox
            )

    # Avoid tight_layout (can explode in notebooks)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.90, bottom=0.12)

    plt.show()
    return d
```
