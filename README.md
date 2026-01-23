
```
dist_dossier_ai_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.ylabel("Percentage (%)")
plt.xlabel("Group")
plt.title("Influence of Dossier Completion and AI on Evaluation Distribution")
plt.legend(title="Evaluation note", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()




```
