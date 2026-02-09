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
#    Returns: mi + n_used + n_nan + pct_nan + etc.
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

        # Fill remaining NaNs after coercion (should be none after mask, but safe)
        x_num = x_num.fillna(x_num.median())

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
        "mi_norm": np.nan,  # fill later after we know max MI
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

    # Sort by MI descending
    out = out.sort_values("mi", ascending=False)
    return out


# ============================================================
# 3) Plot: top MI columns + show missingness context in table
# ============================================================
def plot_top_mi_columns(mi_df, top_n=15, title="Top MI (column-level, fair)"):
    top = mi_df.head(top_n).sort_values("mi")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["mi"])
    plt.xlabel("Mutual Information (higher = more informative)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Quick report for specific delay columns (requested)
# ============================================================
def report_columns(mi_df, cols):
    cols = [c for c in cols if c in mi_df.index]
    if not cols:
        print("None of the requested columns exist in the MI table.")
        return
    display_cols = ["type", "mi", "mi_norm", "n_used", "n_nan", "pct_nan", "note"]
    print("\n=== Requested columns MI + missingness ===\n")
    print(mi_df.loc[cols, display_cols].sort_values("mi", ascending=False))


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
print(mi_cols.head(30)[["type", "mi", "mi_norm", "n_used", "n_nan", "pct_nan", "note"]])

# 2) Plot top MI columns
plot_top_mi_columns(mi_cols, top_n=15, title="[MI] Top columns (fair) with evaluate_note")

# 3) Focus report: delays you mentioned
DELAY_FOCUS = ["delai_reparation", "delai_indemnisation", "delai_de_completude"]
report_columns(mi_cols, DELAY_FOCUS)

```
