```

def split_cols(df, target_col):
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]
    if target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]
    return num_cols, cat_cols

def corr_numeric(df, num_cols, method="spearman"):
    # Spearman is good for ordinal-ish targets and non-linear monotonic relations
    return df[num_cols].corr(method=method)

def plot_corr_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_mat, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def corr_with_target_numeric(df, target_col, num_cols, method="spearman"):
    corr = df[num_cols + [target_col]].corr(method=method)[target_col].drop(target_col)
    out = pd.DataFrame({"corr": corr, "abs_corr": corr.abs()}).sort_values("abs_corr", ascending=False)
    return out



def plot_top_target_corr(corr_df, top_n=15, title="Top correlations with target"):
    top = corr_df.head(top_n).sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index, top["corr"])
    plt.title(title)
    plt.xlabel("correlation")
    plt.tight_layout()
    plt.show()


def cramers_v(x, y):
    # drop missing pairs
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    ct = pd.crosstab(x, y).to_numpy()
    n = ct.sum()
    if n == 0:
        return np.nan

    # chi-square (computed manually)
    row_sum = ct.sum(axis=1, keepdims=True)
    col_sum = ct.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    expected = np.where(expected == 0, 1e-12, expected)
    chi2 = ((ct - expected) ** 2 / expected).sum()

    r, k = ct.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1) + 1e-12))


def cat_cat_matrix(df, cat_cols):
    M = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for a in cat_cols:
        for b in cat_cols:
            M.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
    return M



def correlation_ratio(categories, values):
    # drop missing pairs
    mask = categories.notna() & values.notna()
    categories, values = categories[mask], values[mask]
    if categories.nunique() < 2:
        return np.nan

    overall_mean = values.mean()
    ss_between = 0.0
    ss_total = ((values - overall_mean) ** 2).sum()

    for cat in categories.unique():
        group = values[categories == cat]
        ss_between += len(group) * (group.mean() - overall_mean) ** 2

    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)



def cat_num_matrix(df, cat_cols, num_cols):
    M = pd.DataFrame(index=cat_cols, columns=num_cols, dtype=float)
    for c in cat_cols:
        for n in num_cols:
            M.loc[c, n] = correlation_ratio(df[c], df[n])
    return M


def cat_target_strength(df, cat_cols, target_col):
    rows = []
    for c in cat_cols:
        strength = correlation_ratio(df[c], df[target_col])
        rows.append((c, strength))
    out = pd.DataFrame(rows, columns=["feature", "eta"]).sort_values("eta", ascending=False).set_index("feature")
    return out



def plot_cat_target_strength(cat_strength_df, top_n=15, title="Categorical impact on target (eta)"):
    top = cat_strength_df.head(top_n).sort_values("eta")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index, top["eta"])
    plt.title(title)
    plt.xlabel("eta (0..1)")
    plt.tight_layout()
    plt.show()


def onehot_for_corr(df, target_col, drop_first=False):
    X = df.drop(columns=[target_col], errors="ignore")
    X_enc = pd.get_dummies(X, dummy_na=False, drop_first=drop_first)
    # add target back
    out = X_enc.copy()
    out[target_col] = df[target_col]
    return out


def corr_all_onehot(df, target_col, method="spearman"):
    df_enc = onehot_for_corr(df, target_col)
    return df_enc.corr(method=method), df_enc


def plot_scatter_matrix(df, num_cols, target_col=None, max_cols=8):
    cols = num_cols[:max_cols]
    pd.plotting.scatter_matrix(df[cols], figsize=(10, 10), diagonal="hist")
    plt.suptitle("Scatter matrix (numeric features)")
    plt.tight_layout()
    plt.show()


def plot_target_by_category(df, cat_col, target_col="evaluate_note", top_k=12):
    tmp = df[[cat_col, target_col]].dropna(subset=[cat_col, target_col]).copy()
    top = tmp[cat_col].value_counts().head(top_k).index
    tmp = tmp[tmp[cat_col].isin(top)]

    plt.figure(figsize=(12, 4))
    tmp.boxplot(column=target_col, by=cat_col, grid=False)
    plt.title(f"{target_col} by {cat_col} (top {top_k} categories)")
    plt.suptitle("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




target_col = "evaluate_note"
num_cols, cat_cols = split_cols(df, target_col)

# 1) numeric correlation heatmap
corr_num = corr_numeric(df, num_cols, method="spearman")
plot_corr_heatmap(corr_num, title="Numeric-Numeric Spearman correlation")

# 2) numeric correlation with target + plot
corr_t = corr_with_target_numeric(df, target_col, num_cols, method="spearman")
plot_top_target_corr(corr_t, top_n=15)

# 3) categorical impact on target (eta) + plot
cat_imp = cat_target_strength(df, cat_cols, target_col)
plot_cat_target_strength(cat_imp, top_n=15)

# 4) category-category association heatmap (Cramér’s V)
if len(cat_cols) > 1:
    M_cat = cat_cat_matrix(df, cat_cols)
    plot_corr_heatmap(M_cat, title="Categorical-Categorical association (Cramér's V)")

# 5) category-numeric association heatmap (eta)
if len(cat_cols) > 0 and len(num_cols) > 0:
    M_cat_num = cat_num_matrix(df, cat_cols, num_cols + [target_col])
    plot_corr_heatmap(M_cat_num, title="Categorical-Numeric association (eta)")

# 6) one-hot correlation (all columns) + heatmap (optional)
corr_all, df_enc = corr_all_onehot(df, target_col, method="spearman")
# (careful: can be wide if many categories)
# plot_corr_heatmap(corr_all, title="All columns (one-hot) Spearman correlation")

# 7) extra plots
plot_scatter_matrix(df, num_cols, max_cols=8)
# boxplot target by one categorical column:
# plot_target_by_category(df, cat_col="PARCOURS_FINAL", target_col=target_col, top_k=12)





```
