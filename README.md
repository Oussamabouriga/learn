```


def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names
    """
    # sort so bars are ordered (negative -> positive)
    top_df = top_df.sort_values("corr")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df.index.astype(str), top_df["corr"])

    ax.set_xlim(-1, 1)

    # optional: vertical line at 0 for readability
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    if show_values:
        for y, v in enumerate(top_df["corr"].values):
            # place text slightly to the right for positive, left for negative
            x = v + 0.03 if v >= 0 else v - 0.03
            ha = "left" if v >= 0 else "right"
            ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.show()

```
