```

def nan_summary_table(df, sort_by="nan_pct", ascending=False):


    total_rows = len(df)

    summary = pd.DataFrame({
        "total_rows": total_rows,
        "nan_count": df.isna().sum(),
        "nan_pct": df.isna().mean() * 100,
        "non_nan_count": df.notna().sum(),
        "non_nan_pct": df.notna().mean() * 100
    })

    summary = summary.round(2)
    summary = summary.sort_values(sort_by, ascending=ascending)

    return summary


nan_table = nan_summary_table(df)
display(nan_table)

```
