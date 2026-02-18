```
# Example 1 — simplest call
plot_delay_vs_age(
    df=df,
    age_col="Age",
    delay_col="delai_Sinistre"   # must be in MINUTES in your dataframe
)

# Example 2 — full control (ranges + bins + formatting)
plot_delay_vs_age(
    df=df,
    age_col="Age",
    delay_col="delai_Sinistre",
    age_range=(18, 80),
    age_breaks=[18, 25, 35, 45, 55, 65, 80],                  # age bins
    delay_range=(0, 10080),                                   # 0 → 7 days (minutes)
    delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],
    delay_display="auto",                                     # show as m / h / d
    show_percent=False,                                       # True → Volume in %
    show_points=True,
    title="Delay vs Age (Volume + Avg delay)"
)

# Example 3 — show delay labels in hours only
plot_delay_vs_age(
    df=df,
    age_col="Age",
    delay_col="delai_Sinistre",
    delay_display="hours"
)


```
