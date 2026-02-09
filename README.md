```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 0) Helpers
# ============================================================
def get_numeric_cols(df, target_col):
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    return num_cols


def get_categorical_cols(df, target_col):
    num_cols = df.select_dtypes(include=["number", "int64", "float64", "Int64"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]
    return cat_cols


def missing_report(df, cols):
    rep = pd.DataFrame({
        "n_total": len(df),
        "n_nan": [int(df[c].isna().sum()) for c in cols],
    }, index=cols)
    rep["pct_nan"] = 100 * rep["n_nan"] / rep["n_total"]
    return rep.sort_values("pct_nan", ascending=False)


# ============================================================
# 1) Prepare matrices for PCA
#    PCA needs numeric matrix -> we impute + scale
# ============================================================
def prepare_numeric_matrix(df, target_col):
    """
    Returns:
      X_scaled (np.array), feature_names (list), y (Series)
    """
    y = pd.to_numeric(df[target_col], errors="coerce")
    num_cols = get_numeric_cols(df, target_col)

    X = df[num_cols].copy()
    # keep rows where target exists
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # impute numeric
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, num_cols, y


def onehot_encode_top_levels(series, min_count=20, top_k=None):
    """
    One-hot encode a categorical series but keep only:
      - levels with count >= min_count
      - optionally only top_k most frequent levels
    """
    s = series.astype("string").fillna("MISSING")
    vc = s.value_counts()

    keep = vc[vc >= min_count].index
    if top_k is not None:
        keep = vc.head(top_k).index.intersection(keep)

    s2 = s.where(s.isin(keep), other="OTHER")
    d = pd.get_dummies(s2, prefix=series.name, drop_first=False)
    return d


def prepare_mixed_matrix_onehot(df, target_col, cat_cols, min_count=20, top_k=None):
    """
    Mixed PCA approach (PCA on numeric + one-hot categoricals).
    Returns:
      X_scaled, feature_names, y
    """
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    y = y.loc[mask]

    # numeric part
    num_cols = get_numeric_cols(df, target_col)
    X_num = df.loc[mask, num_cols].copy()
    X_num_imp = SimpleImputer(strategy="median").fit_transform(X_num)

    # categorical part (one-hot but controlled)
    onehots = []
    for c in cat_cols:
        d = onehot_encode_top_levels(df.loc[mask, c], min_count=min_count, top_k=top_k)
        onehots.append(d)

    X_cat = pd.concat(onehots, axis=1) if onehots else pd.DataFrame(index=X_num.index)

    # combine
    X_all = np.concatenate([X_num_imp, X_cat.values], axis=1)
    feature_names = num_cols + X_cat.columns.tolist()

    # scale
    X_scaled = StandardScaler().fit_transform(X_all)

    return X_scaled, feature_names, y


# ============================================================
# 2) Run PCA + extract explained variance + scores + loadings
# ============================================================
def run_pca(X_scaled, feature_names, n_components=5):
    """
    Returns:
      pca (model), scores (DataFrame), loadings (DataFrame), explained_variance_ratio (array)
    """
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(X_scaled)

    scores = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])])

    # loadings: contribution of each original feature to each PC
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return pca, scores, loadings, pca.explained_variance_ratio_


# ============================================================
# 3) Plots: explained variance, PC scatter, PC-target correlation
# ============================================================
def plot_explained_variance(ev_ratio, title="PCA explained variance"):
    x = np.arange(1, len(ev_ratio) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, ev_ratio, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(x, np.cumsum(ev_ratio), marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative explained variance")
    plt.title(title + " (cumulative)")
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


def plot_pc_scatter(scores, y, title="PC1 vs PC2 colored by target"):
    if "PC1" not in scores.columns or "PC2" not in scores.columns:
        print("Need at least 2 components for PC1 vs PC2 scatter.")
        return

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(scores["PC1"], scores["PC2"], c=y.values, s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(sc, label="evaluate_note")
    plt.tight_layout()
    plt.show()


def pc_target_correlations(scores, y, method="spearman"):
    """
    Correlate each PC with target to find which components relate to the note.
    """
    tmp = scores.copy()
    tmp["target"] = y.values
    corr = tmp.corr(method=method)["target"].drop("target")
    out = pd.DataFrame({"corr": corr, "abs_corr": corr.abs()}).sort_values("abs_corr", ascending=False)
    return out


def plot_pc_target_corr(pc_corr_df, title="Correlation between PCs and target"):
    d = pc_corr_df.sort_values("corr")
    plt.figure(figsize=(7, 4))
    plt.barh(d.index.astype(str), d["corr"])
    plt.xlabel("correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Explain "what affects note" using PCA loadings
#    -> pick PCs most correlated with target, then show top loadings
# ============================================================
def top_loadings_for_pc(loadings, pc_name="PC1", top_n=15):
    s = loadings[pc_name].copy()
    out = pd.DataFrame({"loading": s, "abs_loading": s.abs()}).sort_values("abs_loading", ascending=False).head(top_n)
    return out


def summarize_key_drivers(loadings, pc_corr_df, top_pcs=2, top_features=12):
    """
    Returns a dict: for the most target-related PCs, the strongest contributing features.
    """
    result = {}
    pcs = pc_corr_df.head(top_pcs).index.tolist()
    for pc in pcs:
        result[pc] = top_loadings_for_pc(loadings, pc, top_n=top_features)
    return result


# ============================================================
# =================== CALL EVERYTHING (END) ===================
# ============================================================

TARGET_COL = "evaluate_note"

# ---- (A) PCA on NUMERIC ONLY ----
X_num, num_features, y = prepare_numeric_matrix(df, TARGET_COL)

pca_num, scores_num, loadings_num, ev_num = run_pca(X_num, num_features, n_components=5)

plot_explained_variance(ev_num, title="[NUMERIC PCA] explained variance")
plot_pc_scatter(scores_num, y, title="[NUMERIC PCA] PC1 vs PC2 (colored by evaluate_note)")

pc_corr_num = pc_target_correlations(scores_num, y, method="spearman")
print("\n[NUMERIC PCA] PC-target correlations:\n", pc_corr_num)
plot_pc_target_corr(pc_corr_num, title="[NUMERIC PCA] PCs correlation with evaluate_note")

drivers_num = summarize_key_drivers(loadings_num, pc_corr_num, top_pcs=2, top_features=12)
print("\n[NUMERIC PCA] Key drivers for target-related PCs:")
for pc, table in drivers_num.items():
    print(f"\n--- {pc} (most related to target) top loadings ---\n{table}")


# ---- (B) PCA on MIXED DATA (numeric + categorical via controlled one-hot) ----
CAT_COLS = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "list_prest"]  # adjust/add
# Control rare categories to avoid noise/explosion:
MIN_COUNT = 20   # filter rare levels
TOP_K_LEVELS = None  # or put 30 to keep only top 30 levels per categorical column

X_mix, mix_features, y2 = prepare_mixed_matrix_onehot(df, TARGET_COL, CAT_COLS, min_count=MIN_COUNT, top_k=TOP_K_LEVELS)

pca_mix, scores_mix, loadings_mix, ev_mix = run_pca(X_mix, mix_features, n_components=5)

plot_explained_variance(ev_mix, title="[MIXED PCA] explained variance (numeric + one-hot)")
plot_pc_scatter(scores_mix, y2, title="[MIXED PCA] PC1 vs PC2 (colored by evaluate_note)")

pc_corr_mix = pc_target_correlations(scores_mix, y2, method="spearman")
print("\n[MIXED PCA] PC-target correlations:\n", pc_corr_mix)
plot_pc_target_corr(pc_corr_mix, title="[MIXED PCA] PCs correlation with evaluate_note")

drivers_mix = summarize_key_drivers(loadings_mix, pc_corr_mix, top_pcs=2, top_features=15)
print("\n[MIXED PCA] Key drivers for target-related PCs:")
for pc, table in drivers_mix.items():
    print(f"\n--- {pc} (most related to target) top loadings ---\n{table}")


# ---- (C) Optional: Missingness report for your key delay cols ----
FOCUS_COLS = ["delai_reparation", "delai_indemnisation", "delai_de_completude"]
exists = [c for c in FOCUS_COLS if c in df.columns]
if exists:
    print("\n[Missingness report] for key delay columns:\n", missing_report(df, exists))
```
