```
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns.tolist()
    data = corr_mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(12, 8))

    # viridis like your first plot + handle NaN
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")  # NaNs shown as light gray (not random white)

    im = ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        vmin=-1, vmax=1,              # fixed correlation range
        cmap=cmap,
        interpolation="nearest"       # keeps blocks clean, no artifacts
    )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    # ✅ remove grid (some styles add it)
    ax.grid(False)

    # also remove any cell borders if they appear from styling
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("correlation")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

```

