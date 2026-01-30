```



import matplotlib.pyplot as plt

colors = ["green", "lightgreen", "gold", "orange", "red", "black"]  # KO 0..5

ax = cozzzunts_pct.T.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 7),
    color=colors,
    edgecolor="black",
    linewidth=0.3
)

plt.title("Distribution de KO (0..5) pour chaque Evaluation Note (0..10)")
plt.xlabel("Evaluation note")
plt.ylabel("Pourcentage (%)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(title="KO", bbox_to_anchor=(1.02, 1), loc="upper left")

# Add labels: count + %
for i, note in enumerate(cozzzunts_pct.columns):  # evaluate_note columns
    cumulative = 0
    for ko in cozzzunts_pct.index:  # KO rows
        pct = cozzzunts_pct.loc[ko, note]
        cnt = cozzzunts.loc[ko, note]
        if pct >= 3:  # avoid clutter
            y = cumulative + pct / 2
            ax.text(i, y, f"{int(cnt)}\n{pct:.1f}%", ha="center", va="center", fontsize=8)
        cumulative += pct

plt.tight_layout()
plt.show()

```

    
       