```
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns.tolist()

    # Convert to numpy, keep NaNs to handle them
    data = corr_mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Make NaNs visible instead of "random white blocks" ---
    # Option A (recommended): show NaNs as light gray
    cmap = plt.cm.RdBu_r.copy()   # diverging map good for correlations
    cmap.set_bad(color="lightgray")

    im = ax.imshow(
        np.ma.masked_invalid(data),   # mask NaNs => uses set_bad color
        aspect="auto",
        vmin=-1, vmax=1,
        cmap=cmap,
        interpolation="nearest"       # avoids weird smoothing artifacts
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

