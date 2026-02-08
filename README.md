```
def sat_count_by_prog(sat, prog):
    return (
        prog.to_frame('prog')
            .assign(sat=sat)
            .groupby(['prog', 'sat'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=range(11), fill_value=0)
    )


table = sat_count_by_prog(df["sat"], df["prog"])
print(table)

```
