```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Helper: simple correlation heatmap plot
# ============================================================
def plot_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns.tolist()
    plt.figure(figsize=(12, 8))
    plt.imshow(corr_mat, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 1) Heatmap for NUMERIC variables only (including target)
# ============================================================
def numeric_corr_matrix(df, target_col, method="spearman"):
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    cols = [c for c in num_cols if c != target_col] + [target_col]
    return df[cols].corr(method=method)


def numeric_heatmap(df, target_col, method="spearman"):
    corr_mat = numeric_corr_matrix(df, target_col, method=method)
    plot_heatmap(corr_mat, title=f"NUMERIC correlation heatmap ({method})")
    return corr_mat


# ============================================================
# 2) One-Hot Encoding (for categorical columns)
# ============================================================
def onehot_encode_column(df, col):
    """
    Encodes ONLY one categorical column into multiple one-hot columns.
    Keeps other columns untouched.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in df")

    dummies = pd.get_dummies(df[col], prefix=col)
    return dummies


# ============================================================
# 3) Top correlation with target (works on any numeric dataframe)
# ============================================================
def top_corr_with_target(df_numeric, target_col, top_n=20, method="spearman"):
    """
    df_numeric: dataframe with only numeric columns (including target).
    returns top_n columns most correlated to target.
    """
    corr = df_numeric.corr(method=method)[target_col].drop(target_col).dropna()
    out = pd.DataFrame({"corr": corr, "abs_corr": corr.abs()}).sort_values("abs_corr", ascending=False).head(top_n)
    return out


def plot_top_corr(top_df, title="Top correlations with target"):
    top_df = top_df.sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(top_df.index, top_df["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Heatmap + Top-corr for one categorical group alone (encoded)
#    Example: PARCOURS_FINAL alone
# ============================================================
def categorical_group_analysis(df, cat_col, target_col, top_n=20, method="spearman"):
    """
    Creates one-hot columns for cat_col, adds target,
    then:
      - plots correlation heatmap
      - plots top correlations with target (levels of that categorical)
    """
    dummies = onehot_encode_column(df, cat_col)
    tmp = dummies.copy()
    tmp[target_col] = df[target_col]

    corr_mat = tmp.corr(method=method)
    plot_heatmap(corr_mat, title=f"{cat_col} correlation heatmap ({method})")

    top_df = top_corr_with_target(tmp, target_col, top_n=top_n, method=method)
    plot_top_corr(top_df, title=f"Top {top_n} correlations with {target_col} ({cat_col} levels)")
    return corr_mat, top_df


# ============================================================
# 5) Correlation of each DELAI column with target
# ============================================================
def delays_corr_with_target(df, delay_cols, target_col, method="spearman"):
    rows = []
    for c in delay_cols:
        if c in df.columns:
            corr = df[[c, target_col]].corr(method=method).iloc[0, 1]
            rows.append((c, corr))
    out = pd.DataFrame(rows, columns=["delay_col", "corr"]).assign(abs_corr=lambda d: d["corr"].abs())
    out = out.sort_values("abs_corr", ascending=False).set_index("delay_col")
    return out


def plot_delays_corr(delay_corr_df, title="Delays correlation with target"):
    d = delay_corr_df.sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(d.index, d["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Correlation of KO count with target + plot
# ============================================================
def ko_corr_with_target(df, ko_col, target_col, method="spearman"):
    if ko_col not in df.columns:
        raise ValueError(f"Column '{ko_col}' not found in df")
    return df[[ko_col, target_col]].corr(method=method).iloc[0, 1]


def plot_ko_scatter(df, ko_col, target_col, title=None):
    tmp = df[[ko_col, target_col]].dropna()
    plt.figure(figsize=(6, 4))
    plt.scatter(tmp[ko_col], tmp[target_col], s=20)
    plt.xlabel(ko_col)
    plt.ylabel(target_col)
    plt.title(title or f"{target_col} vs {ko_col}")
    plt.tight_layout()
    plt.show()


# ============================================================
# =================== CALL THEM ALL (END) ====================
# ============================================================

TARGET_COL = "evaluate_note"

PARCOURS_FINAL_COL = "PARCOURS_FINAL"
PARCOURS_INITIAL_COL = "PARCOURS_INITIAL"
LIST_PREST_COL = "list_prest"

# Put your delay columns names here (minutes)
DELAY_COLS = [
    # "delai_sinistre",
    # "delai_indemnisation",
    # "delai_preparation",
]

KO_COL = "nombre_prestation_ko"   # change if different in your df


# 1) Heatmap for NUMERIC variables alone
corr_num = numeric_heatmap(df, TARGET_COL, method="spearman")

# 2) Top correlation for NUMERIC variables alone
num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
df_num = df[num_cols].copy()   # numeric-only dataframe (target included if numeric)
top_num = top_corr_with_target(df_num, TARGET_COL, top_n=20, method="spearman")
print("\nTop correlations (NUMERIC only):\n", top_num)
plot_top_corr(top_num, title="Top correlations with target (NUMERIC only)")

# 3) PARCOURS_FINAL alone: heatmap + top correlations
corr_pf, top_pf = categorical_group_analysis(df, PARCOURS_FINAL_COL, TARGET_COL, top_n=20, method="spearman")
print("\nTop correlations (PARCOURS_FINAL levels):\n", top_pf)

# 4) PARCOURS_INITIAL alone: heatmap + top correlations
corr_pi, top_pi = categorical_group_analysis(df, PARCOURS_INITIAL_COL, TARGET_COL, top_n=20, method="spearman")
print("\nTop correlations (PARCOURS_INITIAL levels):\n", top_pi)

# 5) list_prest alone: heatmap + top correlations
corr_lp, top_lp = categorical_group_analysis(df, LIST_PREST_COL, TARGET_COL, top_n=20, method="spearman")
print("\nTop correlations (list_prest levels):\n", top_lp)

# 6) Delays correlation with target + plot
delay_corr = delays_corr_with_target(df, DELAY_COLS, TARGET_COL, method="spearman")
print("\nDelay correlations with target:\n", delay_corr)
plot_delays_corr(delay_corr, title="Delays correlation with evaluate_note")

# 7) KO correlation with target + plot
ko_corr = ko_corr_with_target(df, KO_COL, TARGET_COL, method="spearman")
print(f"\nCorrelation({KO_COL}, {TARGET_COL}) = {ko_corr:.4f}")
plot_ko_scatter(df, KO_COL, TARGET_COL, title=f"{TARGET_COL} vs {KO_COL} (corr={ko_corr:.3f})")
```
