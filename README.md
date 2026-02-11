```
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_heatmap(corr_mat, title="Correlation heatmap", annotate=True, fmt="{:.2f}"):
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

    ax.grid(False)
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("correlation")

    ax.set_title(title)

    # ✅ Add correlation value in each cell
    if annotate:
        # choose text color depending on cell brightness (for readability)
        norm = plt.Normalize(vmin=-1, vmax=1)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue

                rgba = cmap(norm(v))
                # perceived luminance (0..1) to decide black/white text
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"

                ax.text(
                    j, i,
                    fmt.format(v),
                    ha="center", va="center",
                    fontsize=8,
                    color=text_color
                )

    plt.tight_layout()
    plt.show()


plot_corr_heatmap(corr_mat, title="NUMERIC correlation heatmap (Spearman)", annotate=True, fmt="{:.2f}")

```
