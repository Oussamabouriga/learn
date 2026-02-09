import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1) One-Hot encode categorical columns (keep numeric as-is)
# ------------------------------------------------------------
def onehot_encode(df, target_col, drop_first=False):
    """
    Returns df_encoded: numeric columns untouched + categorical columns one-hot encoded + target included.
    """
    X = df.drop(columns=[target_col], errors="ignore")
    X_enc = pd.get_dummies(X, drop_first=drop_first)
    X_enc[target_col] = df[target_col]
    return X_enc


# ------------------------------------------------------------
# 2) Correlation matrix for ALL columns (after encoding)
# ------------------------------------------------------------
def corr_all(df_encoded, method="spearman"):
    """
    Computes correlation matrix for all columns in df_encoded.
    """
    return df_encoded.corr(method=method)


# ------------------------------------------------------------
# 3) Plot heatmap for correlation matrix (general)
# ------------------------------------------------------------
def plot_corr_heatmap(corr_mat, title="Correlation heatmap", max_cols=60, focus_col=None):
    """
    Plots a correlation heatmap.
    - If too many columns, it keeps the most related ones to focus_col (usually your target).
    """
    corr_plot = corr_mat.copy()

    # reduce size if too wide
    if focus_col is not None and corr_plot.shape[0] > max_cols and focus_col in corr_plot.columns:
        keep = corr_plot[focus_col].abs().sort_values(ascending=False).head(max_cols).index
        corr_plot = corr_plot.loc[keep, keep]
    elif corr_plot.shape[0] > max_cols:
        corr_plot = corr_plot.iloc[:max_cols, :max_cols]

    cols = corr_plot.columns.tolist()

    plt.figure(figsize=(12, 8))
    plt.imshow(corr_plot, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4) Top correlations with the target (includes categorical after one-hot)
# ------------------------------------------------------------
def top_corr_with_target(corr_mat, target_col, top_n=20):
    """
    Returns a dataframe of top correlated features with the target.
    """
    s = corr_mat[target_col].drop(target_col).dropna()
    out = pd.DataFrame({"corr": s, "abs_corr": s.abs()}).sort_values("abs_corr", ascending=False).head(top_n)
    return out


def plot_top_corr(top_df, title="Top correlations with target"):
    """
    Horizontal bar plot of top correlations (shows sign).
    """
    top_df = top_df.sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(top_df.index, top_df["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5) Correlation heatmap for NUMERIC columns only (original df)
# ------------------------------------------------------------
def get_numeric_cols(df, target_col):
    """
    Returns numeric columns excluding target.
    """
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    return [c for c in num_cols if c != target_col]


def corr_numeric_only(df, target_col, method="spearman"):
    """
    Correlation matrix of numeric features + target (no encoding).
    """
    num_cols = get_numeric_cols(df, target_col)
    return df[num_cols + [target_col]].corr(method=method)


# ------------------------------------------------------------
# 6) Correlation between each delay column and the target
# ------------------------------------------------------------
def corr_delays_with_target(df, delay_cols, target_col, method="spearman"):
    """
    Returns correlation of each delay col with the target.
    Ignores NaN pairwise (keeps NaNs in df).
    """
    rows = []
    for c in delay_cols:
        if c in df.columns:
            corr = df[[c, target_col]].corr(method=method).iloc[0, 1]
            rows.append((c, corr))
    out = pd.DataFrame(rows, columns=["delay_col", "corr"]).assign(abs_corr=lambda d: d["corr"].abs())
    out = out.sort_values("abs_corr", ascending=False).set_index("delay_col")
    return out


def plot_delay_corr(delay_corr_df, title="Delay columns correlation with target"):
    """
    Bar plot of delay correlations with target.
    """
    d = delay_corr_df.sort_values("corr")
    plt.figure(figsize=(10, 6))
    plt.barh(d.index, d["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7) Correlation between KO count (prestation KO) and the target
# ------------------------------------------------------------
def corr_ko_with_target(df, ko_col, target_col, method="spearman"):
    """
    Returns single correlation value between KO column and target.
    """
    if ko_col not in df.columns:
        raise ValueError(f"KO column '{ko_col}' not found in df.")
    return df[[ko_col, target_col]].corr(method=method).iloc[0, 1]


def plot_ko_vs_target(df, ko_col, target_col, title=None):
    """
    Simple scatter plot: KO vs target (good to visualize trend).
    """
    tmp = df[[ko_col, target_col]].dropna()
    plt.figure(figsize=(6, 4))
    plt.scatter(tmp[ko_col], tmp[target_col], s=20)
    plt.xlabel(ko_col)
    plt.ylabel(target_col)
    plt.title(title or f"{target_col} vs {ko_col}")
    plt.tight_layout()
    plt.show()


# ============================================================
# CALL THEM ONE BY ONE (EDIT THESE NAMES)
# ============================================================

TARGET_COL = "evaluate_note"

# Put here all your delay columns (minutes). Example:
DELAY_COLS = [
    # "delai_decision",
    # "delai_reparation",
    # "delai_validation",
]

# Put your KO count column name here (example: "numbreko" or "nbr_ko"):
KO_COL = "numbreko"   # <-- change to your real column name


# 1) One-hot encode
df_enc = onehot_encode(df, target_col=TARGET_COL, drop_first=False)

# 2) Correlation for ALL columns (encoded)
corr_all_mat = corr_all(df_enc, method="spearman")

# 3) Heatmap for ALL columns (encoded)
plot_corr_heatmap(
    corr_all_mat,
    title="Correlation heatmap (ALL columns: numeric + categorical encoded)",
    max_cols=60,
    focus_col=TARGET_COL
)

# 4) Top correlations with target (includes categorical levels after encoding)
top_df = top_corr_with_target(corr_all_mat, TARGET_COL, top_n=20)
print(top_df)
plot_top_corr(top_df, title="Top correlations with target (includes encoded categorical levels)")

# 5) Heatmap numeric only (original df)
corr_num_mat = corr_numeric_only(df, TARGET_COL, method="spearman")
plot_corr_heatmap(
    corr_num_mat,
    title="Correlation heatmap (NUMERIC only)",
    max_cols=60
)

# 6) Correlation each delay with target + plot
delay_corr_df = corr_delays_with_target(df, DELAY_COLS, TARGET_COL, method="spearman")
print(delay_corr_df)
plot_delay_corr(delay_corr_df, title="Delays correlation with evaluate_note")

# 7) Correlation KO with target + plot
ko_corr = corr_ko_with_target(df, KO_COL, TARGET_COL, method="spearman")
print(f"Correlation({KO_COL}, {TARGET_COL}) = {ko_corr:.4f}")
plot_ko_vs_target(df, KO_COL, TARGET_COL, title=f"{TARGET_COL} vs {KO_COL} (corr={ko_corr:.3f})")
