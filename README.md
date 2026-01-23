
```
from tabulate import tabulate

# Build distribution
dist_gender_car = (
    df.groupby(["gender", "car"])["satisfaction"]
      .value_counts()
      .unstack(fill_value=0)
      .sort_index()
)

# Percentages within each (gender, car)
dist_gender_car_pct = (
    dist_gender_car.div(dist_gender_car.sum(axis=1), axis=0) * 100
).round(2)

# Combine count + percentage
dist_gender_car_combined = (
    dist_gender_car.astype(str) + " (" + dist_gender_car_pct.astype(str) + "%)"
)

# Human-readable labels
dist_gender_car_combined.index = dist_gender_car_combined.index.map(
    lambda x: f"{'Male' if x[0]==0 else 'Female'} - {'Car' if x[1]==1 else 'No Car'}"
)

# 👉 IMPORTANT: move index to a column for tabulate
dist_gender_car_table = dist_gender_car_combined.reset_index()
dist_gender_car_table.rename(columns={"index": "Group"}, inplace=True)

# Display with tabulate
print(
    tabulate(
        dist_gender_car_table,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)


```
