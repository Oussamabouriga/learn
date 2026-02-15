```
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def plot_top_corr_bar(
    top_df,
    title="Top correlations with target",
    show_values=True,
    figsize=(10, 6),
    max_label_len=35,          # hard truncate
    wrap_width=None,           # e.g. 18 to wrap labels; None = no wrap
    dpi=100,                   # safe dpi
):
    """
    top_df: DataFrame with column 'corr' and index = feature names
    """

    if top_df is None or len(top_df) == 0:
        print("No data to plot.")
        return None

    d = top_df.sort_values("corr").copy()

    # ---- SAFE LABELS (prevents huge image)
    labels = d.index.astype(str).tolist()

    # optional wrap (useful if you don't want truncation)
    if wrap_width is not None:
        labels = ["\n".join(textwrap.wrap(s, width=wrap_width)) for s in labels]

    # hard truncate (always)
    labels = [
        (s if len(s) <= max_label_len else s[: max_label_len - 1] + "…")
        for s in labels
    ]

    vals = d["corr"].astype(float).to_numpy()
    y = np.arange(len(vals))

    # ---- SAFE FIGURE RENDERING
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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

    # DON'T use tight_layout (can explode size in notebooks)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.90, bottom=0.12)

    # Important: render without bbox_inches="tight" behavior
    fig.set_constrained_layout(False)

    plt.show()
    return fig



```
