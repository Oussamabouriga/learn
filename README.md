```
import matplotlib.pyplot as plt

def plot_mi_levels_with_counts(mi_levels_df, title="Information mutuelle + nombre de valeurs",
                               show_mi_values=True, show_count_values=True):
    if mi_levels_df.empty:
        print("No levels to plot.")
        return

    d = mi_levels_df.sort_values("mi")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bars = MI
    bars = ax1.barh(d.index.astype(str), d["mi"])
    ax1.set_xlabel("l'information mutuelle")
    ax1.set_title(title)

    # Force MI axis 0..1
    ax1.set_xlim(0, 1)

    # MI labels
    if show_mi_values:
        for y, v in enumerate(d["mi"].values):
            x = min(v + 0.02, 0.98)
            ax1.text(x, y, f"{v:.3f}", va="center", ha="left", fontsize=9)

    # Second axis = counts
    ax2 = ax1.twiny()
    ax2.plot(d["count"], d.index.astype(str), marker="o")
    ax2.set_xlabel("nombre de valeurs")

    # Count labels
    if show_count_values:
        for y, c in enumerate(d["count"].values):
            ax2.annotate(f"{int(c)}",
                         (c, y),
                         textcoords="offset points",
                         xytext=(6, 0),
                         va="center",
                         fontsize=9)

    plt.tight_layout()
    plt.show()

```

