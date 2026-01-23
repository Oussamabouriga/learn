
```
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

bars = plt.bar(
    dist_evaluation_table["evaluation"],
    dist_evaluation_table["percentage"]
)

plt.xlabel("Evaluation note")
plt.ylabel("Percentage (%)")
plt.title("Distribution of Evaluation Notes")
plt.xticks(dist_evaluation_table["evaluation"])

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()




dist_gender_table = dist_gender_combined.reset_index()
dist_gender_table.rename(columns={"index": "Gender"}, inplace=True)

```
