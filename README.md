missing_report = (
    df.isna()
      .sum()
      .to_frame("missing_count")
      .assign(missing_pct=lambda x: (x.missing_count / len(df) * 100).round(2))
      .sort_values("missing_pct", ascending=False)
)

missing_report
