```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


# ============================================================
# 0) Helpers
# ============================================================
def _to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _mask_xy(x, y):
    y = _to_numeric_safe(y)
    mask = x.notna() & y.notna()
    return x[mask], y[mask]


def _mi_regression_1d(X_2d, y_1d, discrete_features, random_state=42):
    mi = mutual_info_regression(
        X_2d, y_1d,
        discrete_features=discrete_features,
        random_state=random_state
    )
    return float(mi[0])


# ============================================================
# 1) MI for ONE COLUMN + full missingness stats (PRO)
# ============================================================
def mi_column_with_stats(df, col, target_col, random_state=42):
    n_total = len(df)

    x_raw = df[col]
    y_raw = df[target_col]

    # Missingness (for x only)
    n_nan_x = int(x_raw.isna().sum())
    pct_nan_x = (n_nan_x / n_total) * 100 if n_total else np.nan

    # Used rows (need x and y present)
    x, y = _mask_xy(x_raw, y_raw)
    n_used = int(len(x))

    # If not enough usable rows, MI is not reliable
    if n_used < 30:
        return {
            "feature": col,
            "type": "numeric" if _is_numeric(x_raw) else "categorical",
            "mi": np.nan,
            "mi_norm": np.nan,
            "n_total": n_total,
            "n_used": n_used,
            "n_nan": n_nan_x,
            "pct_nan": pct_nan_x,
            "note": "too_few_rows_for_mi(<30)"
        }

    # Compute MI
    if _is_numeric(x_raw):
        x_num = _to_numeric_safe(x).astype(float)
        x_num = x_num.fillna(x_num.median())  # safety

        X = x_num.values.reshape(-1, 1)
        mi = _mi_regression_1d(X, y.values, discrete_features=[False], random_state=random_state)
        ftype = "numeric"
    else:
        x_cat = pd.Categorical(x.astype("string"))
        codes = x_cat.codes.reshape(-1, 1)
        mi = _mi_regression_1d(codes, y.values, discrete_features=[True], random_state=random_state)
        ftype = "categorical"

    return {
        "feature": col,
        "type": ftype,
        "mi": float(mi),
        "mi_norm": np.nan,
        "n_total": n_total,
        "n_used": n_used,
        "n_nan": n_nan_x,
        "pct_nan": pct_nan_x,
        "note": ""
    }


# ============================================================
# 2) MI for ALL columns (COLUMN LEVEL, FAIR) + stats
# ============================================================
def mi_all_columns_with_stats(df, target_col, numeric_cols, categorical_cols, random_state=42):
    rows = []

    for c in numeric_cols:
        if c == target_col:
            continue
        rows.append(mi_column_with_stats(df, c, target_col, random_state=random_state))

    for c in categorical_cols:
        rows.append(mi_column_with_stats(df, c, target_col, random_state=random_state))

    out = pd.DataFrame(rows).set_index("feature")

    # Normalize MI for easy comparison (0..1)
    max_mi = out["mi"].max()
    out["mi_norm"] = out["mi"] / (max_mi + 1e-12)

    # Extra: quality / reliability flags
    out["used_pct"] = 100 * out["n_used"] / (out["n_total"] + 1e-12)
    out["reliability"] = np.where(
        out["n_used"] < 100, "LOW (few rows)",
        np.where(out["n_used"] < 300, "MEDIUM", "HIGH")
    )

    # Sort by MI descending
    out = out.sort_values("mi", ascending=False)
    return out


# ============================================================
# 3) Plots (more plots for better understanding)
# ============================================================
def plot_top_mi_columns(mi_df, top_n=15, title="Top MI (column-level, fair)"):
    top = mi_df.head(top_n).sort_values("mi")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["mi"])
    plt.xlabel("Mutual Information (higher = more informative)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_missingness_bar(mi_df, top_n=30, title="Missingness (% NaN) for top features"):
    """
    Shows %NaN for the same top features (by MI) so you can judge reliability.
    """
    top = mi_df.head(top_n).copy()
    top = top.sort_values("pct_nan")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["pct_nan"])
    plt.xlabel("% NaN in feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_used_rows_bar(mi_df, top_n=30, title="Used rows (%) for MI computation"):
    """
    Shows how much data was actually used for MI (n_used / n_total).
    """
    top = mi_df.head(top_n).copy()
    top = top.sort_values("used_pct")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["used_pct"])
    plt.xlabel("% rows used (feature & target not NaN)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mi_vs_missingness(mi_df, title="MI vs Missingness (%NaN)"):
    """
    Scatter: each point is a feature.
    Helps spot features with high MI but huge missingness (danger).
    """
    tmp = mi_df.dropna(subset=["mi"]).copy()
    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["pct_nan"], tmp["mi"], s=20)
    plt.xlabel("% NaN in feature")
    plt.ylabel("MI")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mi_vs_n_used(mi_df, title="MI vs n_used (how much data supported it)"):
    """
    Scatter: MI vs n_used to see if a feature looks important just from few rows.
    """
    tmp = mi_df.dropna(subset=["mi"]).copy()
    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["n_used"], tmp["mi"], s=20)
    plt.xlabel("n_used (rows used to compute MI)")
    plt.ylabel("MI")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Quick report for specific columns (delays)
# ============================================================
def report_columns(mi_df, cols):
    cols = [c for c in cols if c in mi_df.index]
    if not cols:
        print("None of the requested columns exist in the MI table.")
        return
    display_cols = ["type", "mi", "mi_norm", "n_used", "used_pct", "n_nan", "pct_nan", "reliability", "note"]
    print("\n=== Requested columns MI + missingness ===\n")
    print(mi_df.loc[cols, display_cols].sort_values("mi", ascending=False))


def plot_focus_delays(mi_df, cols, title_prefix="[Delays]"):
    """
    Two small plots:
      - MI of delays
      - Missingness of delays
    """
    cols = [c for c in cols if c in mi_df.index]
    if not cols:
        print("No delay columns found in MI table.")
        return

    d = mi_df.loc[cols].copy()

    # MI plot
    d1 = d.sort_values("mi")
    plt.figure(figsize=(8, 4))
    plt.barh(d1.index.astype(str), d1["mi"])
    plt.xlabel("MI")
    plt.title(f"{title_prefix} MI with target")
    plt.tight_layout()
    plt.show()

    # Missingness plot
    d2 = d.sort_values("pct_nan")
    plt.figure(figsize=(8, 4))
    plt.barh(d2.index.astype(str), d2["pct_nan"])
    plt.xlabel("% NaN")
    plt.title(f"{title_prefix} Missingness (%NaN)")
    plt.tight_layout()
    plt.show()


# ============================================================
# =================== CALL ALL (END) =========================
# ============================================================

TARGET_COL = "evaluate_note"

CATEGORICAL_COLS = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "list_prest"]  # add more if needed
NUMERIC_COLS = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()

# 1) Compute MI table with missingness stats
mi_cols = mi_all_columns_with_stats(
    df=df,
    target_col=TARGET_COL,
    numeric_cols=NUMERIC_COLS,
    categorical_cols=CATEGORICAL_COLS,
    random_state=42
)

print("\n=== Column-level MI (FAIR) + missingness stats (top 30) ===\n")
print(mi_cols.head(30)[["type", "mi", "mi_norm", "n_used", "used_pct", "n_nan", "pct_nan", "reliability", "note"]])

# 2) Plot top MI columns
plot_top_mi_columns(mi_cols, top_n=15, title="[MI] Top columns (fair) with evaluate_note")

# 3) New plots to understand reliability / imbalance
plot_missingness_bar(mi_cols, top_n=30, title="[MI] %NaN for Top 30 MI features")
plot_used_rows_bar(mi_cols, top_n=30, title="[MI] %Rows used for Top 30 MI features")
plot_mi_vs_missingness(mi_cols, title="[MI] MI vs %NaN (watch for high MI but too missing)")
plot_mi_vs_n_used(mi_cols, title="[MI] MI vs n_used (watch for high MI with low support)")

# 4) Focus report: delays you mentioned
DELAY_FOCUS = ["delai_reparation", "delai_indemnisation", "delai_de_completude"]
report_columns(mi_cols, DELAY_FOCUS)
plot_focus_delays(mi_cols, DELAY_FOCUS, title_prefix="[Delays Focus]")


```
