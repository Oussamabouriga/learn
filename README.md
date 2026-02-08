```

target = "evaluate_note"
features = get_numeric_features(df, target_col=target, exclude_cols=["PARCOURS_FINAL","PARCOURS_INITIAL","list_prest"])




corr_df = corr_with_target(df, target_col=target, features=features, method="spearman")
corr_df.head(15)




plot_top_corr(corr_df, top_n=15)



plot_corr_heatmap(df, cols=features + [target], method="spearman")



mi_df = mutual_information(df, target_col=target, features=features, task="regression")
mi_df.head(15)



plot_top_mi(mi_df, top_n=15)



comp_df, explained, loadings = run_pca(df, features, n_components=2)





plot_pca_2d(comp_df, df[target], title="PCA colored by evaluate_note")


comp_df5, explained5, loadings5 = run_pca(df, features, n_components=5)
plot_pca_variance(explained5)




top_pca_loadings(loadings, pc="PC1", top_n=10)
top_pca_loadings(loadings, pc="PC2", top_n=10)




boxplot_feature_by_note(df, feature="delai_Sinistre", target_col=target)


df2 = add_note_bins(df, target_col=target)
boxplot_feature_by_bin(df2, feature="delai_Sinistre", bin_col="note_bin")

```
