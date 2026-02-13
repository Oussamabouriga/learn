```
# Exemple: NPS vs delai_declaration, binning en minutes
g = plot_volume_nps_filtered(
    df,
    delay_col="delai_declaration",
    score_col="evaluate_note",
    delay_breaks=[0, 10, 30, 60, 120, 300, 600, 1500, 5000, 9000, 2731258],
    delay_range=(0, 2731258),

    # filtre catégorie
    cat_col="PARCOURS_FINAL",
    cat_value="APPLE_CDE",   # valeur exacte

    show_percent=False       # True si tu veux volume en %
)




```
