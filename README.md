```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) Show what builds each PCA component (PC1..PC5)
#    -> loadings = weights of original variables in each PC
# ============================================================
def get_top_loadings(loadings_df, pc_name="PC1", top_n=15):
    """
    Returns a table of the strongest contributors (by absolute loading)
    for a given principal component.
    """
    s = loadings_df[pc_name].copy()
    out = pd.DataFrame({
        "loading": s,
        "abs_loading": s.abs()
    }).sort_values("abs_loading", ascending=False).head(top_n)
    return out


def plot_top_loadings(loadings_df, pc_name="PC1", top_n=15, title=None):
    """
    Bar plot of top absolute loadings for one PC.
    Positive loading -> pushes PC up
    Negative loading -> pushes PC down
    """
    top = get_top_loadings(loadings_df, pc_name, top_n=top_n).sort_values("loading")
    plt.figure(figsize=(10, 6))
    plt.barh(top.index.astype(str), top["loading"])
    plt.xlabel("Loading (weight in the component)")
    plt.title(title or f"Top loadings for {pc_name}")
    plt.tight_layout()
    plt.show()


def explain_pcs(loadings_df, n_components=5, top_n=15, do_plot=True):
    """
    Prints + (optional) plots the top contributors for PC1..PCn.
    """
    for i in range(1, n_components + 1):
        pc = f"PC{i}"
        print(f"\n==================== {pc} ====================")
        tbl = get_top_loadings(loadings_df, pc_name=pc, top_n=top_n)
        print(tbl)

        if do_plot:
            plot_top_loadings(loadings_df, pc_name=pc, top_n=top_n,
                              title=f"{pc}: variables that build this component (top {top_n})")


# ============================================================
# 2) CALL (use the loadings you already computed)
#    - For numeric PCA: loadings_num
#    - For mixed PCA:  loadings_mix
# ============================================================

# Example: show PC1..PC5 for NUMERIC PCA
explain_pcs(loadings_num, n_components=5, top_n=15, do_plot=True)

# If you want also for MIXED PCA (numeric + one-hot):
# explain_pcs(loadings_mix, n_components=5, top_n=20, do_plot=True)


```
