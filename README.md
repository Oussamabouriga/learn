```

import numpy as np
import matplotlib.pyplot as plt

def plot_ko_by_note_100pct(counts_df, title="Distribution de KO par Evaluation Note (100% stacked)"):
    # 1) Force correct shape: rows=KO (0..5), cols=note (0..10)
    counts = counts_df.copy()
    counts = counts.reindex(index=range(0, 6), columns=range(0, 11), fill_value=0).astype(int)

    # 2) Percentages per NOTE (column-wise): each note column sums to 100%
    col_sum = counts.sum(axis=0).replace(0, np.nan)  # avoid division by zero
    pct = counts.div(col_sum, axis=1) * 100
    pct = pct.fillna(0)

    # 3) Prepare plot data: rows=note, cols=KO
    notes = np.arange(0, 11)
    kos   = np.arange(0, 6)
    pct_note = pct.T      # (note x KO)
    cnt_note = counts.T   # (note x KO)

    # 4) Plot (perfect layout)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_axisbelow(True)

    bottom = np.zeros(len(notes))
    colors = ["#2E7D32", "#66BB6A", "#FBC02D", "#FB8C00", "#E53935", "#424242"]  # pro colors

    min_label_pct = 6  # show labels only if segment >= 6%

    for j, ko in enumerate(kos):
        vals = pct_note[ko].to_numpy()
        bars = ax.bar(
            notes, vals, bottom=bottom, width=0.8,
            color=colors[j], edgecolor="white", linewidth=0.8,
            label=f"KO={ko}"
        )

        # Labels: percentage + count inside each segment (only if big enough)
        for i, b in enumerate(bars):
            p = vals[i]
            if p >= min_label_pct:
                c = int(cnt_note.iloc[i, j])
                ax.text(
                    b.get_x() + b.get_width()/2,
                    bottom[i] + p/2,
                    f"{p:.1f}%\n({c})",
                    ha="center", va="center",
                    fontsize=9, color="black"
                )

        bottom += vals

    # 5) Styling
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Evaluation note (0–10)")
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xticks(notes)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Nombre KO", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

# ✅ Use it with your counts table:
plot_ko_by_note_100pct(cozzzunts)
```

    
       