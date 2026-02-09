```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer


# ============================================================
# 0) Helpers
# ============================================================
def _mask_xy(x, y):
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna() & x.notna()
    return x[mask], y[mask]


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _mi_regression(X, y, discrete_features, random_state=42):
    # X must be 2D
    mi = mutual_info_regression(
        X, y,
        discrete_features=discrete_features,
        random_state=random_state
    )
    return float(mi[0]) if len(mi) == 1 else mi


# ============================================================
# 1) MI for ONE COLUMN (FAIR column-level MI)
#    - numeric: treat as continuous
#    - categorical: treat as discrete (codes)
# ============================================================
def mi_column(df, col, target_col, random_state=42):
    x = df[col]
    y = df[target_col]

    x, y = _mask_xy(x, y)

    if x.empty:
        return np.nan

    if _is_numeric(x):
        X = pd.to_numeric(x, errors="coerce").fillna(pd.to_numeric(x, errors="coerce").median()).values.reshape(-1, 1)
        return _mi_regression(X, y.values, discrete_features=[False], random_state=random_state)

    # categorical -> codes, treat as discrete
    x_cat = pd.Categorical(x.astype("string"))
    codes = x_cat.codes.reshape(-1, 1)  # 0..K-1
    return _mi_regression(codes, y.values, discrete_features=[True], random_state=random_state)


# ============================================================
# 2) Column-level MI for ALL columns (main table)
# ============================================================
def mi_all_columns(df, target_col, numeric_cols, categorical_cols, random_state=42):
    rows = []

    for c in numeric_cols:
        if c == target_col:
            continue
        rows.append({"feature": c, "type": "numeric", "mi": mi_column(df, c, target_col, random_state)})

    for c in categorical_cols:
        rows.append({"feature": c, "type": "categorical", "mi": mi_column(df, c, target_col, random_state)})

    out = pd.DataFrame(rows).set_index("feature")
    out["mi_norm"] = out["mi"] / (out["mi"].max() + 1e-12)
    out = out.sort_values("mi", ascending=False)
    return out


def plot_top_mi_columns(mi_cols_df, top_n=15, title="Top MI (column-level, fair)"):
    top = mi_cols_df.head(top_n).sort_values("mi")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["mi"])
    plt.xlabel("Mutual Information (higher = more informative)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Drill-down MI inside ONE categorical column (levels)
#    - one-hot levels MI against target (for interpretation)
#    - we also show counts to avoid being fooled by rare levels
# ============================================================
def mi_categorical_levels(df, cat_col, target_col, top_n=20, min_count=20, random_state=42):
    x = df[cat_col].astype("string")
    y = df[target_col]
    x, y = _mask_xy(x, y)

    # one-hot for levels
    dummies = pd.get_dummies(x, prefix=cat_col, drop_first=False)

    # counts per level (sum of 1s)
    counts = dummies.sum(axis=0)

    # filter rare levels (important for unbalanced data)
    keep = counts[counts >= min_count].index
    dummies = dummies[keep]
    counts = counts[keep]

    if dummies.shape[1] == 0:
        return pd.DataFrame(columns=["mi", "mi_norm", "count"])

    mi_vals = mutual_info_regression(
        dummies.values, y.values,
        discrete_features=[True] * dummies.shape[1],
        random_state=random_state
    )

    out = pd.DataFrame({
        "mi": mi_vals,
        "count": counts.values
    }, index=dummies.columns)

    out["mi_norm"] = out["mi"] / (out["mi"].max() + 1e-12)
    out = out.sort_values("mi", ascending=False).head(top_n)
    return out


def plot_mi_levels(mi_levels_df, title="Top MI levels (categorical drill-down)"):
    if mi_levels_df.empty:
        print("No levels to plot (maybe all levels are rare under min_count).")
        return

    top = mi_levels_df.sort_values("mi")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["mi"])
    plt.xlabel("Mutual Information")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mi_levels_with_counts(mi_levels_df, title="MI levels + sample size"):
    if mi_levels_df.empty:
        print("No levels to plot.")
        return

    d = mi_levels_df.sort_values("mi")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(d.index.astype(str), d["mi"])
    ax1.set_xlabel("Mutual Information")
    ax1.set_title(title)

    ax2 = ax1.twiny()
    ax2.plot(d["count"], d.index.astype(str), marker="o")
    ax2.set_xlabel("Count (sample size)")

    plt.tight_layout()
    plt.show()


# ============================================================
# 4) 2D "MI-friendly" plots for top numeric columns
#    Since MI can be non-linear, we show binned target means:
#    - bin numeric feature into quantiles
#    - plot mean target per bin + count per bin
# ============================================================
def plot_binned_target_by_numeric(df, num_col, target_col, bins=8):
    x = pd.to_numeric(df[num_col], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 20:
        print(f"Not enough data to bin for {num_col}.")
        return

    # quantile bins
    binner = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
    xb = binner.fit_transform(x.values.reshape(-1, 1)).astype(int).ravel()

    tmp = pd.DataFrame({"bin": xb, "y": y.values})
    stats = tmp.groupby("bin")["y"].agg(["mean", "count"]).sort_index()

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(stats.index, stats["mean"], marker="o")
    ax1.set_xlabel(f"{num_col} (quantile bins)")
    ax1.set_ylabel(f"Mean {target_col}")

    ax2 = ax1.twinx()
    ax2.bar(stats.index, stats["count"], alpha=0.3)
    ax2.set_ylabel("Count")

    plt.title(f"{num_col}: binned mean {target_col} (non-linear view)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5) Joint MI (pairs) at COLUMN LEVEL (logical)
#    We compute MI of pair(feature1, feature2) with target
#    by discretizing each feature into bins and combining them.
# ============================================================
def joint_mi_pairs_columns(df, target_col, features, bins=6, top_n=15, random_state=42):
    y = pd.to_numeric(df[target_col], errors="coerce")
    pair_rows = []

    # prepare discretized versions of each feature
    disc_map = {}
    for f in features:
        x = df[f]
        x, yy = _mask_xy(x, y)

        if x.empty:
            continue

        if _is_numeric(x):
            x_num = pd.to_numeric(x, errors="coerce")
            # discretize numeric into quantile bins
            binner = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
            xb = binner.fit_transform(x_num.values.reshape(-1, 1)).astype(int).ravel()
            disc_map[f] = (xb, yy.values)
        else:
            # categorical -> codes
            x_cat = pd.Categorical(x.astype("string"))
            disc_map[f] = (x_cat.codes, yy.values)

    feats = [f for f in features if f in disc_map]

    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            f1, f2 = feats[i], feats[j]

            x1, y1 = disc_map[f1]
            x2, y2 = disc_map[f2]

            # align by using common length only if same mask isn't guaranteed
            # easiest: rebuild mask together from original df for exact alignment
            x1o = df[f1]
            x2o = df[f2]
            yo = pd.to_numeric(df[target_col], errors="coerce")
            mask = x1o.notna() & x2o.notna() & yo.notna()

            a = x1o[mask]
            b = x2o[mask]
            yy = yo[mask]

            # discretize again consistently for this pair
            if _is_numeric(a):
                a = pd.to_numeric(a, errors="coerce")
                ba = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile").fit_transform(a.values.reshape(-1,1)).astype(int).ravel()
            else:
                ba = pd.Categorical(a.astype("string")).codes

            if _is_numeric(b):
                b = pd.to_numeric(b, errors="coerce")
                bb = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile").fit_transform(b.values.reshape(-1,1)).astype(int).ravel()
            else:
                bb = pd.Categorical(b.astype("string")).codes

            pair = (ba.astype(int) * 1000 + bb.astype(int)).reshape(-1, 1)

            mi_pair = mutual_info_regression(pair, yy.values, discrete_features=[True], random_state=random_state)[0]
            pair_rows.append((f1, f2, float(mi_pair), len(yy)))

    out = pd.DataFrame(pair_rows, columns=["feat1", "feat2", "mi_pair", "n"])
    if out.empty:
        return out

    out["mi_pair_norm"] = out["mi_pair"] / (out["mi_pair"].max() + 1e-12)
    out = out.sort_values("mi_pair", ascending=False).head(top_n)
    return out


def plot_joint_pairs(joint_df, title="Top Joint MI pairs"):
    if joint_df.empty:
        print("No joint pairs computed.")
        return

    labels = (joint_df["feat1"] + " + " + joint_df["feat2"]).tolist()[::-1]
    vals = joint_df["mi_pair"].values[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals)
    plt.xlabel("Joint Mutual Information")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# =================== CALL THEM ALL (END) ====================
# ============================================================

TARGET_COL = "evaluate_note"

# --- define your columns (fair)
CATEGORICAL_COLS = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "list_prest"]  # add more if you want
NUMERIC_COLS = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()

# 1) Column-level MI (FAIR ranking)
mi_cols = mi_all_columns(df, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, random_state=42)
print("\n[1] Column-level MI (fair ranking):\n", mi_cols.head(30))

# 2) Plot top columns by MI
plot_top_mi_columns(mi_cols, top_n=15, title="[2] Top MI (COLUMN LEVEL, fair)")

# 3) Drill-down levels (optional but useful)
#    We only drill down for categorical columns you care about
for cat in CATEGORICAL_COLS:
    levels = mi_categorical_levels(df, cat, TARGET_COL, top_n=20, min_count=20, random_state=42)
    print(f"\n[3] {cat} top MI levels (min_count=20):\n", levels)
    plot_mi_levels_with_counts(levels, title=f"[3] {cat}: top MI levels + counts (min_count=20)")

# 4) 2D non-linear view for top numeric columns
top_numeric = mi_cols[mi_cols["type"] == "numeric"].head(5).index.tolist()
for col in top_numeric:
    plot_binned_target_by_numeric(df, col, TARGET_COL, bins=8)

# 5) Joint MI pairs at COLUMN LEVEL (logical)
#    Use top 8 columns (by MI) for pair search (keeps it readable)
top_cols_for_pairs = mi_cols.head(8).index.tolist()
joint = joint_mi_pairs_columns(df, TARGET_COL, top_cols_for_pairs, bins=6, top_n=15, random_state=42)
print("\n[5] Top joint MI pairs (column-level):\n", joint)
plot_joint_pairs(joint, title="[5] Top Joint MI pairs (COLUMN LEVEL)")
```
