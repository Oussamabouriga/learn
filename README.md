```
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    # ✅ fixed scale for correlation (-1..1) + diverging colors
    im = ax.imshow(
        corr_mat.values,
        aspect="auto",
        vmin=-1, vmax=1,
        cmap="coolwarm"   # try also: "RdBu_r" or "bwr"
    )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("correlation")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
```
