```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer


# ============================================================
# Helpers: build X (features) and y (target) for MI
# ============================================================
def build_mi_dataset(df, target_col):
    """
    Returns:
      X_enc: One-hot encoded + numeric, NaNs filled (median for numeric, "MISSING" for cats)
      y: target as numeric
    """
    y = df[target_col].astype(float)

    X = df.drop(columns=[target_col]).copy()

    # split numeric / categorical
    num_cols = X.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill missing
    for c in num_cols:
        X[c] = X[c].astype(float)
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("MISSING")

    # One-hot for categoricals
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # keep only rows where target is not NaN
    mask = y.notna()
    return X_enc.loc[mask], y.loc[mask]


# ============================================================
# 1) Mutual Information for each feature with the target
# ============================================================
def mi_with_target(df, target_col, random_state=42):
    """
    Computes mutual information (MI) for each feature in df against target_col.
    Returns a sorted DataFrame: mi, normalized_mi
    """
    X_enc, y = build_mi_dataset(df, target_col)

    # MI for regression-style target (evaluate_note is numeric 0..10)
    mi = mutual_info_regression(X_enc.values, y.values, random_state=random_state)

    mi_df = pd.DataFrame({"feature": X_enc.columns, "mi": mi})
    mi_df["mi_norm"] = mi_df["mi"] / (mi_df["mi"].max() + 1e-12)
    mi_df = mi_df.sort_values("mi", ascending=False).set_index("feature")
    return mi_df


# ============================================================
# 2) Bar plot of top MI features
# ============================================================
def plot_top_mi(mi_df, top_n=20, title="Top Mutual Information with target"):
    top = mi_df.head(top_n).sort_values("mi")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["mi"])
    plt.xlabel("Mutual Information (higher = more informative)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Compare MI vs Correlation (Spearman) to see overlap
# ============================================================
def corr_with_target_encoded(df, target_col, method="spearman"):
    """
    Encodes categoricals (one-hot) + keeps numeric; computes correlation with target.
    Returns a Series indexed by feature.
    """
    X_enc, y = build_mi_dataset(df, target_col)
    tmp = X_enc.copy()
    tmp[target_col] = y.values
    corr = tmp.corr(method=method)[target_col].drop(target_col).dropna()
    return corr


def compare_mi_vs_corr(df, target_col, top_n=30, corr_method="spearman"):
    """
    Returns a DataFrame containing MI + corr + ranks.
    Also returns overlap list (features in both top lists).
    """
    mi_df = mi_with_target(df, target_col)
    corr_s = corr_with_target_encoded(df, target_col, method=corr_method)

    # align
    comp = pd.DataFrame({
        "mi": mi_df["mi"],
        "corr": corr_s
    }).dropna()

    comp["abs_corr"] = comp["corr"].abs()

    # top lists
    top_mi = comp.sort_values("mi", ascending=False).head(top_n).index
    top_corr = comp.sort_values("abs_corr", ascending=False).head(top_n).index
    overlap = list(set(top_mi).intersection(set(top_corr)))

    comp["mi_rank"] = comp["mi"].rank(ascending=False, method="dense")
    comp["corr_rank"] = comp["abs_corr"].rank(ascending=False, method="dense")
    comp = comp.sort_values(["mi_rank", "corr_rank"])

    return comp, overlap


def plot_mi_vs_corr_scatter(comp_df, title="MI vs |Correlation|"):
    """
    Scatter plot: each point is a feature.
    x = |corr|, y = MI
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(comp_df["abs_corr"], comp_df["mi"], s=20)
    plt.xlabel("|Correlation| (Spearman)")
    plt.ylabel("Mutual Information")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) 2D plots for top MI features (easy to understand)
#    - If feature is binary (0/1): boxplot-like using jitter scatter
#    - If numeric: scatter plot
# ============================================================
def is_binary_series(s):
    vals = pd.Series(s).dropna().unique()
    return len(vals) <= 2 and set(vals).issubset({0, 1})


def plot_top_mi_features_2d(df, target_col, top_n=6, random_state=42):
    """
    Picks top MI features (from encoded set) and draws a simple plot per feature.
    """
    mi_df = mi_with_target(df, target_col, random_state=random_state)
    X_enc, y = build_mi_dataset(df, target_col)

    top_feats = mi_df.head(top_n).index.tolist()

    for f in top_feats:
        x = X_enc[f].values
        yy = y.values

        plt.figure(figsize=(6, 4))

        if is_binary_series(x):
            # jitter scatter for 0/1
            jitter = (np.random.RandomState(0).rand(len(x)) - 0.5) * 0.08
            plt.scatter(x + jitter, yy, s=15)
            plt.xticks([0, 1], [f"{f}=0", f"{f}=1"])
            plt.xlabel("Binary feature")
        else:
            plt.scatter(x, yy, s=15)
            plt.xlabel(f)

        plt.ylabel(target_col)
        plt.title(f"2D view: {f} vs {target_col} (high MI)")
        plt.tight_layout()
        plt.show()


# ============================================================
# 5) Joint analysis (MI of pairs): see how TWO features together help
#    Simple approach:
#      - pick top_k features by MI
#      - discretize both features into bins
#      - build a combined "pair" feature
#      - compute MI(pair, target)
# ============================================================
def mi_joint_pairs(df, target_col, top_k=12, bins=6, random_state=42):
    """
    Returns MI for feature pairs (top_k features).
    Uses discretization and a combined pair label to measure joint information.
    """
    mi_df = mi_with_target(df, target_col, random_state=random_state)
    X_enc, y = build_mi_dataset(df, target_col)

    feats = mi_df.head(top_k).index.tolist()
    X = X_enc[feats].copy()

    # discretize (works for numeric; for one-hot 0/1 it stays basically same)
    disc = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
    Xb = disc.fit_transform(X.values)

    pair_scores = []
    n = len(feats)

    for i in range(n):
        for j in range(i + 1, n):
            a = Xb[:, i].astype(int)
            b = Xb[:, j].astype(int)

            # combine into one label: a * big + b
            pair = a * 1000 + b

            # MI between pair and target (treat pair as a discrete feature)
            # mutual_info_regression can handle discrete-ish numeric, but we help by casting float
            score = mutual_info_regression(pair.reshape(-1, 1), y.values, random_state=random_state)[0]
            pair_scores.append((feats[i], feats[j], score))

    out = pd.DataFrame(pair_scores, columns=["feat1", "feat2", "mi_pair"])
    out["mi_pair_norm"] = out["mi_pair"] / (out["mi_pair"].max() + 1e-12)
    out = out.sort_values("mi_pair", ascending=False)
    return out


def plot_top_joint_pairs(joint_df, top_n=15, title="Top joint MI feature pairs"):
    top = joint_df.head(top_n).copy()
    labels = (top["feat1"] + " + " + top["feat2"]).tolist()

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], top["mi_pair"].values[::-1])
    plt.xlabel("Mutual Information (pair)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# =================== CALL THEM ALL (END) ====================
# ============================================================

TARGET_COL = "evaluate_note"

# 1) MI for every feature
mi_df = mi_with_target(df, TARGET_COL)
print("\n[1] Mutual Information with target (top 30):\n", mi_df.head(30))

# 2) Bar plot top MI
plot_top_mi(mi_df, top_n=20, title="[2] Top MI features with evaluate_note")

# 3) Compare MI vs correlation + overlap + scatter
comp_df, overlap = compare_mi_vs_corr(df, TARGET_COL, top_n=30, corr_method="spearman")
print("\n[3] MI vs |corr| comparison (top rows):\n", comp_df.head(30))
print("\n[3] Overlap features (in both top MI and top |corr|):\n", overlap)
plot_mi_vs_corr_scatter(comp_df, title="[3] MI vs |Correlation| (Spearman)")

# 4) 2D plots for top MI features
plot_top_mi_features_2d(df, TARGET_COL, top_n=6)

# 5) Joint analysis (pairs)
joint_df = mi_joint_pairs(df, TARGET_COL, top_k=12, bins=6)
print("\n[5] Joint MI pairs (top 20):\n", joint_df.head(20))
plot_top_joint_pairs(joint_df, top_n=15, title="[5] Top joint MI pairs")

```
