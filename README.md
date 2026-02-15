```
import numpy as np
import matplotlib.pyplot as plt

def plot_top_corr_bar(
    top_df,
    title="Top correlations with target",
    show_values=True,
    figsize=(10, 6),
    max_label_len=40,     # truncate long feature names
    left_margin=0.38      # enough space for y labels (increase if needed)
):
    """
    top_df: DataFrame with column 'corr' and index = feature names
    """

    if top_df is None or len(top_df) == 0:
        print("No data to plot.")
        return

    # sort so bars are ordered (negative -> positive)
    d = top_df.sort_values("corr").copy()

    # make safe, short labels (prevents huge canvas in notebook rendering)
    raw_labels = d.index.astype(str).tolist()
    labels = [
        (s if len(s) <= max_label_len else s[: max_label_len - 1] + "…")
        for s in raw_labels
    ]

    vals = d["corr"].astype(float).values
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlim(-1, 1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("correlation")
    ax.set_title(title)

    if show_values:
        for yi, v in enumerate(vals):
            x = v + 0.03 if v >= 0 else v - 0.03
            ha = "left" if v >= 0 else "right"
            ax.text(x, yi, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    # IMPORTANT: avoid tight_layout() here (it can blow up image size in notebooks)
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.90, bottom=0.12)

    plt.show()

```
