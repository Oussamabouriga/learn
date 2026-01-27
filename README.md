df.loc[(df["testt"] == "apple") & (df["yyy"] == "repair"), "yyy"] = "a_rc"
df.loc[(df["testt"] == "hors_apple") & (df["yyy"] == "repair"), "yyy"] = "hors_a_rc"