
```
df["satisfaction"].value_counts().sort_index()


(df["satisfaction"].value_counts(normalize=True)
   .sort_index()
   .mul(100)
   .round(2))


satisfaction_dist = (
    df["satisfaction"]
    .value_counts()
    .sort_index()
    .to_frame("count")
)

satisfaction_dist["percentage"] = (
    satisfaction_dist["count"] / satisfaction_dist["count"].sum() * 100
).round(2)

satisfaction_dist

dist_counts = (
    df.groupby("gender")["satisfaction"]
      .value_counts()
      .unstack(fill_value=0)
      .sort_index()
)

dist_counts.index = dist_counts.index.map({0: "Male", 1: "Female"})
dist_counts


dist_pct = (dist_counts.div(dist_counts.sum(axis=1), axis=0) * 100).round(2)
dist_pct



dist_gender = (
    df.groupby("gender")["satisfaction"]
      .value_counts()
      .unstack(fill_value=0)
      .sort_index()
)

# percentages per gender (row-wise)
dist_gender_pct = (dist_gender.div(dist_gender.sum(axis=1), axis=0) * 100).round(2)

# combine count + %
dist_gender_combined = (
    dist_gender.astype(str) + " (" + dist_gender_pct.astype(str) + "%)"
)

# rename gender labels
dist_gender_combined.index = dist_gender_combined.index.map({0: "Male", 1: "Female"})

dist_gender_combined


dist_gender_car = (
    df.groupby(["gender", "car"])["satisfaction"]
      .value_counts()
      .unstack(fill_value=0)
      .sort_index()
)

# percentages within each (gender, car) group
dist_gender_car_pct = (
    dist_gender_car.div(dist_gender_car.sum(axis=1), axis=0) * 100
).round(2)

# combine count + %
dist_gender_car_combined = (
    dist_gender_car.astype(str) + " (" + dist_gender_car_pct.astype(str) + "%)"
)

# readable labels
dist_gender_car_combined.index = dist_gender_car_combined.index.map(
    lambda x: f"{'Male' if x[0]==0 else 'Female'} - {'Car' if x[1]==1 else 'No Car'}"
)

dist_gender_car_combined





```
