```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


# ============================================================
# 1) Mutual Information (MI) pour UNE feature (num OU cat)
#    - si cat: on encode en codes + discrete_features=True
# ============================================================
def mi_single_feature(df, feature_col, target_col, random_state=42):
    x = df[feature_col]
    y = df[target_col].astype(float)

    mask = y.notna() & x.notna()
    x = x[mask]
    y = y[mask]

    # numeric ?
    if pd.api.types.is_numeric_dtype(x):
        X = x.astype(float).values.reshape(-1, 1)
        mi = mutual_info_regression(X, y.values, discrete_features=[False], random_state=random_state)[0]
        return float(mi)

    # categorical -> codes
    x_cat = pd.Categorical(x.astype("string"))
    X = x_cat.codes.reshape(-1, 1)  # -1 possible if missing; but we filtered notna above
    mi = mutual_info_regression(X, y.values, discrete_features=[True], random_state=random_state)[0]
    return float(mi)


# ============================================================
# 2) Eta Squared (η²) pour cat -> target numeric
#    Interprétation: part de variance de target expliquée par la catégorie
# ============================================================
def eta_squared(df, cat_col, target_col):
    x = df[cat_col].astype("string")
    y = df[target_col].astype(float)

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    # moyenne globale
    y_mean = y.mean()

    # somme des carrés totale
    sst = ((y - y_mean) ** 2).sum()
    if sst == 0:
        return 0.0

    # somme des carrés "between groups"
    ssb = 0.0
    for lvl, y_grp in y.groupby(x):
        ssb += len(y_grp) * (y_grp.mean() - y_mean) ** 2

    return float(ssb / sst)


# ============================================================
# 3) Corr Spearman pour numeric -> target numeric
# ============================================================
def spearman_corr(df, num_col, target_col):
    tmp = df[[num_col, target_col]].dropna()
    if tmp.empty:
        return np.nan
    return float(tmp.corr(method="spearman").iloc[0, 1])


# ============================================================
# 4) Table "pro" : importance par COLONNE (pas par one-hot)
#    - numeric: corr_spearman + mi
#    - categorical: eta2 + mi
# ============================================================
def feature_importance_by_column(df, target_col, categorical_cols, numeric_cols, random_state=42):
    rows = []

    # numeric features
    for col in numeric_cols:
        if col == target_col:
            continue
        rows.append({
            "feature": col,
            "type": "numeric",
            "spearman_corr": spearman_corr(df, col, target_col),
            "eta2": np.nan,
            "mi": mi_single_feature(df, col, target_col, random_state=random_state),
        })

    # categorical features (NO one-hot here)
    for col in categorical_cols:
        rows.append({
            "feature": col,
            "type": "categorical",
            "spearman_corr": np.nan,
            "eta2": eta_squared(df, col, target_col),
            "mi": mi_single_feature(df, col, target_col, random_state=random_state),
        })

    out = pd.DataFrame(rows).set_index("feature")

    # normalisations pour comparaison (0..1)
    out["mi_norm"] = out["mi"] / (out["mi"].max() + 1e-12)
    out["eta2_norm"] = out["eta2"] / (out["eta2"].max() + 1e-12)

    # score global simple (tu peux changer les weights)
    # - numeric: combine |corr| et MI
    # - categorical: combine eta2 et MI
    out["score"] = np.where(
        out["type"] == "numeric",
        0.5 * out["mi_norm"] + 0.5 * out["spearman_corr"].abs().fillna(0),
        0.5 * out["mi_norm"] + 0.5 * out["eta2_norm"].fillna(0)
    )

    out = out.sort_values("score", ascending=False)
    return out


# ============================================================
# 5) Plot : top features (group-level) par score / MI / eta2
# ============================================================
def plot_top_features_table(imp_df, top_n=15, title="Top feature groups (column-level)"):
    top = imp_df.head(top_n).copy()
    top = top.sort_values("score")  # for horizontal bar

    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["score"])
    plt.xlabel("Score (combined)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_categorical_eta2(imp_df, top_n=15, title="Categorical features: eta² (explained variance)"):
    cat = imp_df[imp_df["type"] == "categorical"].dropna(subset=["eta2"]).head(top_n).copy()
    if cat.empty:
        print("No categorical eta² to plot.")
        return

    cat = cat.sort_values("eta2")
    plt.figure(figsize=(10, 6))
    plt.barh(cat.index.astype(str), cat["eta2"])
    plt.xlabel("eta² (higher = more variance explained)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_numeric_corr(imp_df, top_n=15, title="Numeric features: Spearman corr with target"):
    num = imp_df[imp_df["type"] == "numeric"].dropna(subset=["spearman_corr"]).head(top_n).copy()
    if num.empty:
        print("No numeric correlation to plot.")
        return

    num = num.sort_values("spearman_corr")
    plt.figure(figsize=(10, 6))
    plt.barh(num.index.astype(str), num["spearman_corr"])
    plt.xlabel("Spearman correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Drill-down PRO pour une colonne catégorielle:
#    - mean target par modalité + count
#    - (option) filter rare categories (min_count)
# ============================================================
def categorical_drilldown(df, cat_col, target_col, min_count=30, top_levels=20):
    tmp = df[[cat_col, target_col]].dropna()
    tmp[cat_col] = tmp[cat_col].astype("string")

    stats = tmp.groupby(cat_col)[target_col].agg(["mean", "median", "count"]).sort_values("count", ascending=False)

    # garder seulement les modalités assez présentes
    stats_f = stats[stats["count"] >= min_count].copy()

    # si trop de modalités, on garde top par count
    stats_f = stats_f.head(top_levels).copy()

    # plot mean + count (2 axes)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(stats_f.index.astype(str), stats_f["mean"], marker="o")
    ax1.set_ylabel(f"Mean {target_col}")
    ax1.set_xlabel(cat_col)
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.bar(stats_f.index.astype(str), stats_f["count"], alpha=0.3)
    ax2.set_ylabel("Count")

    plt.title(f"{cat_col}: mean {target_col} + sample size (filtered min_count={min_count})")
    plt.tight_layout()
    plt.show()

    return stats


# ============================================================
# =================== CALL THEM ALL (END) ====================
# ============================================================

TARGET_COL = "evaluate_note"

CATEGORICAL_COLS = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "list_prest"]  # ajoute d'autres si besoin
NUMERIC_COLS = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()

# 1) Importance pro par colonne (pas par one-hot)
imp = feature_importance_by_column(
    df=df,
    target_col=TARGET_COL,
    categorical_cols=CATEGORICAL_COLS,
    numeric_cols=NUMERIC_COLS,
    random_state=42
)

print("\n=== Feature importance (COLUMN LEVEL) ===\n")
print(imp.head(30))

# 2) Plots group-level
plot_top_features_table(imp, top_n=15, title="Top feature groups (column-level) - combined score")
plot_numeric_corr(imp, top_n=15, title="Numeric features: Spearman correlation with evaluate_note")
plot_categorical_eta2(imp, top_n=15, title="Categorical features: eta² (explained variance)")

# 3) Drill-down فقط إذا تحب تفهم داخل العمود
# مثال: PARCOURS_FINAL
stats_pf = categorical_drilldown(df, "PARCOURS_FINAL", TARGET_COL, min_count=30, top_levels=20)
# stats_pi = categorical_drilldown(df, "PARCOURS_INITIAL", TARGET_COL, min_count=30, top_levels=20)
# stats_lp = categorical_drilldown(df, "list_prest", TARGET_COL, min_count=30, top_levels=20)
```
