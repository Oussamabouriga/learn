```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def get_numeric_features(df, target_col, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    exclude_cols.add(target_col)
    
    numeric_cols = df.select_dtypes(include=["number", "Int64", "float64", "int64"]).columns
    features = [c for c in numeric_cols if c not in exclude_cols]
    return features

def make_xy(df, target_col, features):
    X = df[features].copy()       # keep NaN here
    y = df[target_col].copy()
    return X, y

def corr_with_target(df, target_col, features, method="spearman"):
    # Spearman is often better when target is ordinal (0..10)
    corr = df[features + [target_col]].corr(method=method)[target_col].drop(target_col)
    out = corr.to_frame("corr").assign(abs_corr=corr.abs()).sort_values("abs_corr", ascending=False)
    return out

def plot_top_corr(corr_df, top_n=15, title=None):
    top = corr_df.head(top_n).sort_values("corr")
    ax = top["corr"].plot(kind="barh", figsize=(10, 6))
    ax.set_title(title or f"Top {top_n} correlations with target")
    ax.set_xlabel("Correlation")
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df, cols, method="spearman", title="Correlation heatmap"):
    corr = df[cols].corr(method=method)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="corr")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def mutual_information(df, target_col, features, task="regression", random_state=0):
    X = df[features].copy()
    y = df[target_col].copy()

    # Impute X only for MI calculation (keep df untouched)
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=features,
        index=df.index
    )

    if task == "classification":
        mi = mutual_info_classif(X_imp, y, random_state=random_state)
    else:
        mi = mutual_info_regression(X_imp, y, random_state=random_state)

    mi_df = pd.DataFrame({"feature": features, "mi": mi}).sort_values("mi", ascending=False).set_index("feature")
    return mi_df


def plot_top_mi(mi_df, top_n=15, title=None):
    top = mi_df.head(top_n).sort_values("mi")
    ax = top["mi"].plot(kind="barh", figsize=(10, 6))
    ax.set_title(title or f"Top {top_n} mutual information")
    ax.set_xlabel("MI score")
    plt.tight_layout()
    plt.show()


def run_pca(df, features, n_components=2):
    X = df[features].copy()

    # impute + scale
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(X_scaled)

    comp_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)
    explained = pca.explained_variance_ratio_

    # loadings (how features contribute to PCs)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return comp_df, explained, loadings



def plot_pca_variance(explained):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(explained) + 1), np.cumsum(explained), marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.show()



def plot_pca_2d(comp_df, y, title="PCA (PC1 vs PC2)"):
    plt.figure(figsize=(7, 5))
    plt.scatter(comp_df["PC1"], comp_df["PC2"], c=y, s=20)
    plt.colorbar(label="evaluate_note")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def top_pca_loadings(loadings, pc="PC1", top_n=10):
    s = loadings[pc].abs().sort_values(ascending=False).head(top_n)
    return s.to_frame("abs_loading")


def boxplot_feature_by_note(df, feature, target_col="evaluate_note"):
    tmp = df[[feature, target_col]].dropna(subset=[target_col])  # keep feature NaN (boxplot ignores NaN)
    plt.figure(figsize=(12, 4))
    tmp.boxplot(column=feature, by=target_col, grid=False)
    plt.title(f"{feature} distribution by {target_col}")
    plt.suptitle("")
    plt.xlabel(target_col)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()




def add_note_bins(df, target_col="evaluate_note"):
    # change thresholds if you want
    bins = [-0.1, 4, 7, 10.1]
    labels = ["bad(0-4)", "mid(5-7)", "good(8-10)"]
    out = df.copy()
    out["note_bin"] = pd.cut(out[target_col], bins=bins, labels=labels)
    return out

def boxplot_feature_by_bin(df, feature, bin_col="note_bin"):
    tmp = df[[feature, bin_col]].dropna(subset=[bin_col])
    plt.figure(figsize=(8, 4))
    tmp.boxplot(column=feature, by=bin_col, grid=False)
    plt.title(f"{feature} distribution by {bin_col}")
    plt.suptitle("")
    plt.xlabel(bin_col)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()




def quick_feature_analysis(df, target_col="evaluate_note", exclude_cols=None, top_n=15):
    features = get_numeric_features(df, target_col, exclude_cols=exclude_cols)

    # 1) correlation
    corr_df = corr_with_target(df, target_col, features, method="spearman")
    plot_top_corr(corr_df, top_n=top_n, title="Top correlations (Spearman)")

    # 2) mutual info
    mi_df = mutual_information(df, target_col, features, task="regression")
    plot_top_mi(mi_df, top_n=top_n, title="Top mutual information")

    # 3) PCA
    comp_df, explained, loadings = run_pca(df, features, n_components=2)
    plot_pca_2d(comp_df, df[target_col], title="PCA colored by evaluate_note")

    # return tables for inspection
    return {
        "features": features,
        "corr": corr_df,
        "mi": mi_df,
        "pca_components": comp_df,
        "pca_explained": explained,
        "pca_loadings": loadings
    }



results = quick_feature_analysis(df, target_col="evaluate_note", exclude_cols=["PARCOURS_FINAL","PARCOURS_INITIAL","list_prest"])

```
