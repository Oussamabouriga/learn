```
def count_distribution(df, group_col, value_col, value_min=0, value_max=10):
    values = range(value_min, value_max + 1)
    return (
        df.groupby([group_col, value_col])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=values, fill_value=0)
    )

table = count_distribution(df, 'prog', 'sat', 0, 10)
```
