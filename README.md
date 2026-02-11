```
import matplotlib.pyplot as plt

def plot_top_mi_columns(mi_cols_df, top_n=15, title="Top MI (column-level, fair)", show_values=True):
    top = mi_cols_df.head(top_n).sort_values("mi")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top.index.astype(str), top["mi"])

    # ✅ force axis 0..1
    ax.set_xlim(0, 1)

    ax.set_xlabel("Mutual Information (higher = more informative)")
    ax.set_title(title)

    # ✅ add value label on each bar
    if show_values:
        for y, v in enumerate(top["mi"].values):
            # keep text inside the axis even near 1.0
            x = min(v + 0.02, 0.98)
            ax.text(x, y, f"{v:.3f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    plt.show()

```

