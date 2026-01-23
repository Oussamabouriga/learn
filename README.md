
```
import matplotlib.pyplot as plt

# Use percentage table (numeric, not combined strings)
plot_df = dist_dossier_ai_pct.copy()

plt.figure(figsize=(10, 6))

plot_df.plot(
    kind="bar",
    stacked=True
)

plt.ylabel("Percentage (%)")
plt.xlabel("Group")
plt.title("Influence of Dossier Completion and AI on Evaluation Distribution")
plt.legend(title="Evaluation note", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.imshow(dist_dossier_ai_pct, aspect="auto")
plt.colorbar(label="Percentage (%)")

plt.yticks(
    range(len(dist_dossier_ai_pct.index)),
    dist_dossier_ai_pct.index
)
plt.xticks(
    range(len(dist_dossier_ai_pct.columns)),
    dist_dossier_ai_pct.columns
)

plt.xlabel("Evaluation note")
plt.ylabel("Group")
plt.title("Heatmap of Evaluation Distribution by Group")
plt.tight_layout()
plt.show()



```
