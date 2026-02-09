```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Generic plotting helpers (SEPARATED as you requested)
# ============================================================
def plot_corr_heatmap(corr_mat, title="Correlation heatmap"):
    cols = corr_mat.columns.tolist()
    plt.figure(figsize=(12, 8))
    plt.imshow(corr_mat, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_top_corr_bar(top_df, title="Top correlations with target"):
    # top_df must contain a "corr" column, index=feature name
    top_df = top_df.sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(top_df.index.astype(str), top_df["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 1) NUMERIC ONLY: correlation matrix + top correlations
# ============================================================
def get_numeric_df(df, target_col):
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    if target_col not in num_cols:
        raise ValueError(f"Target '{target_col}' must be numeric for correlation.")
    return df[num_cols].copy()


def corr_matrix(df_numeric, method="spearman"):
    return df_numeric.corr(method=method)


def top_corr_with_target_from_matrix(corr_mat, target_col, top_n=20):
    s = corr_mat[target_col].drop(target_col).dropna()
    out = pd.DataFrame({"corr": s, "abs_corr": s.abs()}).sort_values("abs_corr", ascending=False).head(top_n)
    return out


# ============================================================
# 2) One-hot encode ONE categorical column (kept separate)
# ============================================================
def onehot_encode_column(df, col):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in df.")
    return pd.get_dummies(df[col], prefix=col)


# ============================================================
# 3) Categorical group analysis (heatmap + top correlations)
#    Functions separated: one returns matrices, others plot them.
# ============================================================
def categorical_group_corr(df, cat_col, target_col, method="spearman"):
    dummies = onehot_encode_column(df, cat_col)
    tmp = dummies.copy()
    tmp[target_col] = df[target_col]
    corr_mat = tmp.corr(method=method)
    return corr_mat


def categorical_group_top_corr(df, cat_col, target_col, method="spearman", top_n=20):
    dummies = onehot_encode_column(df, cat_col)
    tmp = dummies.copy()
    tmp[target_col] = df[target_col]
    corr_mat = tmp.corr(method=method)
    top_df = top_corr_with_target_from_matrix(corr_mat, target_col, top_n=top_n)
    return top_df


# ============================================================
# 4) Special for list_prest: keep ONLY top 20 correlated levels
#    Then heatmap only among those top levels + target
# ============================================================
def list_prest_top20_heatmap_data(df, list_prest_col, target_col, method="spearman", top_n=20):
    dummies = onehot_encode_column(df, list_prest_col)
    tmp = dummies.copy()
    tmp[target_col] = df[target_col]

    corr_to_target = tmp.corr(method=method)[target_col].drop(target_col).dropna()
    top_cols = corr_to_target.abs().sort_values(ascending=False).head(top_n).index.tolist()

    # return a small correlation matrix: top 20 levels + target
    corr_small = tmp[top_cols + [target_col]].corr(method=method)
    return corr_small, top_cols


def list_prest_top20_table(df, list_prest_col, target_col, method="spearman", top_n=20):
    dummies = onehot_encode_column(df, list_prest_col)
    tmp = dummies.copy()
    tmp[target_col] = df[target_col]
    corr_mat = tmp.corr(method=method)
    top_df = top_corr_with_target_from_matrix(corr_mat, target_col, top_n=top_n)
    return top_df


# ============================================================
# 5) Parcours FINAL + INITIAL together (encoded together)
# ============================================================
def two_categorical_cols_corr(df, col_a, col_b, target_col, method="spearman"):
    da = onehot_encode_column(df, col_a)
    db = onehot_encode_column(df, col_b)
    tmp = pd.concat([da, db], axis=1)
    tmp[target_col] = df[target_col]
    return tmp.corr(method=method)


# ============================================================
# 6) Delays correlation with target (bar plot-friendly)
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
    plt.barh(d.index.astype(str), d["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) nombre_prestation_ko: more informative plot
#    - Boxplot of evaluate_note for each KO count (0..5)
#    - And mean+count line chart (two axes) for extra info
# ============================================================
def ko_corr_with_target(df, ko_col, target_col, method="spearman"):
    if ko_col not in df.columns:
        raise ValueError(f"Column '{ko_col}' not found in df")
    return df[[ko_col, target_col]].corr(method=method).iloc[0, 1]


def plot_ko_effect(df, ko_col, target_col, max_ko=5):
    tmp = df[[ko_col, target_col]].dropna().copy()
    tmp = tmp[tmp[ko_col].between(0, max_ko)]

    # (A) Boxplot (distribution)
    plt.figure(figsize=(10, 4))
    tmp.boxplot(column=target_col, by=ko_col, grid=False)
    plt.title(f"{target_col} distribution by {ko_col} (0..{max_ko})")
    plt.suptitle("")
    plt.xlabel(ko_col)
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.show()

    # (B) Mean + Count per KO value (more info)
    g = tmp.groupby(ko_col)[target_col].agg(["mean", "count"]).reindex(range(max_ko + 1))
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(g.index, g["mean"], marker="o")
    ax1.set_xlabel(ko_col)
    ax1.set_ylabel(f"Mean {target_col}")

    ax2 = ax1.twinx()
    ax2.bar(g.index, g["count"], alpha=0.3)
    ax2.set_ylabel("Count")

    plt.title(f"Mean {target_col} and sample size by {ko_col}")
    plt.tight_layout()
    plt.show()


# ============================================================
# =================== CALL THEM ALL (END) ====================
# ============================================================

TARGET_COL = "evaluate_note"

PARCOURS_FINAL_COL = "PARCOURS_FINAL"
PARCOURS_INITIAL_COL = "PARCOURS_INITIAL"
LIST_PREST_COL = "list_prest"

# Put your delay columns here (minutes)
DELAY_COLS = [
    # "delai_sinistre",
    # "delai_indemnisation",
    # "delai_preparation",
]

KO_COL = "nombre_prestation_ko"   # change if different


# 1) Heat correlation plot for NUMERIC variables alone
df_num = get_numeric_df(df, TARGET_COL)
corr_num = corr_matrix(df_num, method="spearman")
plot_corr_heatmap(corr_num, title="NUMERIC correlation heatmap (Spearman)")

# 2) One-hot encoding for categorical data
# (we keep it per-column; encoding is inside functions)

# 3) Top correlation for NUMERIC data alone
top_num = top_corr_with_target_from_matrix(corr_num, TARGET_COL, top_n=20)
print("\nTop correlations with target (NUMERIC only):\n", top_num)
plot_top_corr_bar(top_num, title="Top correlations with target (NUMERIC only)")

# 4) PARCOURS_FINAL alone: heatmap + top correlations
corr_pf = categorical_group_corr(df, PARCOURS_FINAL_COL, TARGET_COL, method="spearman")
plot_corr_heatmap(corr_pf, title="PARCOURS_FINAL correlation heatmap (Spearman)")
top_pf = categorical_group_top_corr(df, PARCOURS_FINAL_COL, TARGET_COL, method="spearman", top_n=20)
print("\nTop correlations (PARCOURS_FINAL levels):\n", top_pf)
plot_top_corr_bar(top_pf, title="Top correlations with target (PARCOURS_FINAL levels)")

# 5) PARCOURS_INITIAL alone: heatmap + top correlations
corr_pi = categorical_group_corr(df, PARCOURS_INITIAL_COL, TARGET_COL, method="spearman")
plot_corr_heatmap(corr_pi, title="PARCOURS_INITIAL correlation heatmap (Spearman)")
top_pi = categorical_group_top_corr(df, PARCOURS_INITIAL_COL, TARGET_COL, method="spearman", top_n=20)
print("\nTop correlations (PARCOURS_INITIAL levels):\n", top_pi)
plot_top_corr_bar(top_pi, title="Top correlations with target (PARCOURS_INITIAL levels)")

# 6) list_prest alone: ONLY top 20 correlated with target + heatmap + top corr plot
corr_lp_top20, lp_cols = list_prest_top20_heatmap_data(df, LIST_PREST_COL, TARGET_COL, method="spearman", top_n=20)
plot_corr_heatmap(corr_lp_top20, title="list_prest heatmap (TOP 20 levels by |corr| with target)")
top_lp = list_prest_top20_table(df, LIST_PREST_COL, TARGET_COL, method="spearman", top_n=20)
print("\nTop correlations (list_prest TOP 20 levels):\n", top_lp)
plot_top_corr_bar(top_lp, title="Top correlations with target (list_prest TOP 20 levels)")

# 7) Heatmap for PARCOURS_FINAL + PARCOURS_INITIAL together
corr_pf_pi = two_categorical_cols_corr(df, PARCOURS_FINAL_COL, PARCOURS_INITIAL_COL, TARGET_COL, method="spearman")
plot_corr_heatmap(corr_pf_pi, title="PARCOURS_FINAL + PARCOURS_INITIAL heatmap (encoded together)")

# 8) Each delay correlation with target (bar plot)
delay_corr = delays_corr_with_target(df, DELAY_COLS, TARGET_COL, method="spearman")
print("\nDelay correlations with target:\n", delay_corr)
plot_delays_corr(delay_corr, title="Delays correlation with evaluate_note")

# 9) nombre_prestation_ko correlation with target + more informative plots
ko_corr = ko_corr_with_target(df, KO_COL, TARGET_COL, method="spearman")
print(f"\nCorrelation({KO_COL}, {TARGET_COL}) = {ko_corr:.4f}")
plot_ko_effect(df, KO_COL, TARGET_COL, max_ko=5)


```
