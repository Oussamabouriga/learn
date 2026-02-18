```
    
out = plot_age_distribution_by_delay_100pct(
    df=df,
    delay_col="delai_Sinistre",   # in minutes
    age_col="Age",
    delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],
    age_breaks=[0, 25, 35, 45, 60, 120],
    title="Répartition des âges par délai de sinistre (100%)"
)
```
