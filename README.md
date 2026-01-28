```
import pandas as pd

def evalnote_by_binary_table(df, eval_col, bin_col):
    tmp = df[[eval_col, bin_col]].dropna()
    tmp[bin_col] = pd.to_numeric(tmp[bin_col], errors="coerce")
    tmp = tmp[tmp[bin_col].isin([0, 1])]

    counts = tmp.groupby([eval_col, bin_col]).size().unstack(fill_value=0).sort_index()
    pct = (counts.div(counts.sum(axis=0), axis=1) * 100).round(2)  # % within each binary group
    return counts, pct


import matplotlib.pyplot as plt

def evalnote_by_binary_plot(df, eval_col, bin_col, labels={0:"0", 1:"1"}, cmap="Set2", figsize=(12,6), min_pct=3):
    counts, pct = evalnote_by_binary_table(df, eval_col, bin_col)

    # rename legend labels
    pct = pct.rename(columns=labels)

    ax = pct.plot(kind="bar", stacked=True, figsize=figsize, colormap=cmap, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{eval_col} distribution by {bin_col}", fontweight="bold")
    ax.set_xlabel(eval_col)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # labels inside segments: count + %
    # (containers correspond to columns after rename)
    counts_for_labels = counts.rename(columns=labels)
    for j, g in enumerate(pct.columns):
        for i in range(len(pct.index)):
            p = float(pct.iloc[i, j])
            c = int(counts_for_labels.iloc[i, j])
            if c == 0 or p < min_pct:
                continue
            rect = ax.containers[j][i]
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_y() + rect.get_height()/2,
                    f"{c}\n{p:.1f}%",
                    ha="center", va="center", fontsize=8)

    ax.legend(title=bin_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()



evalnote_by_binary_plot(df, "evaluate_note", "decision_ai", labels={0:"Sans AI", 1:"Avec AI"})
evalnote_by_binary_plot(df, "evaluate_note", "dossier_complet", labels={0:"Non complet", 1:"Complet"})

```


```
def evalnote_by_category_table(df, eval_col, cat_col, top_groups=8, other="Other"):
    tmp = df[[eval_col, cat_col]].dropna()
    tmp[cat_col] = tmp[cat_col].astype(str).str.strip()

    # keep top groups
    top = tmp[cat_col].value_counts().head(top_groups).index
    tmp[cat_col] = tmp[cat_col].where(tmp[cat_col].isin(top), other)

    counts = tmp.groupby([eval_col, cat_col]).size().unstack(fill_value=0).sort_index()
    pct = (counts.div(counts.sum(axis=0), axis=1) * 100).round(2)
    return counts, pct


def evalnote_by_category_plot(df, eval_col, cat_col, top_groups=8, cmap="tab20", figsize=(14,6), min_pct=4):
    counts, pct = evalnote_by_category_table(df, eval_col, cat_col, top_groups=top_groups)

    ax = pct.plot(kind="bar", stacked=True, figsize=figsize, colormap=cmap, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{eval_col} distribution by {cat_col} (top {top_groups} + Other)", fontweight="bold")
    ax.set_xlabel(eval_col)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # labels
    for j in range(pct.shape[1]):
        for i in range(pct.shape[0]):
            p = float(pct.iloc[i, j])
            c = int(counts.iloc[i, j])
            if c == 0 or p < min_pct:
                continue
            rect = ax.containers[j][i]
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_y() + rect.get_height()/2,
                    f"{c}\n{p:.1f}%",
                    ha="center", va="center", fontsize=8)

    ax.legend(title=cat_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


evalnote_by_category_plot(df, "evaluate_note", "PARCOURS_FINAL", top_groups=8)
evalnote_by_category_plot(df, "evaluate_note", "PARCOURS_INITIAL", top_groups=8)


```

```
def evalnote_by_countbin_table(df, eval_col, count_col, bins, labels):
    tmp = df[[eval_col, count_col]].dropna()
    tmp[count_col] = pd.to_numeric(tmp[count_col], errors="coerce")
    tmp = tmp.dropna()
    tmp["bin"] = pd.cut(tmp[count_col], bins=bins, labels=labels, include_lowest=True, right=True)

    counts = tmp.groupby([eval_col, "bin"]).size().unstack(fill_value=0).sort_index()
    pct = (counts.div(counts.sum(axis=0), axis=1) * 100).round(2)
    return counts, pct


def evalnote_by_countbin_plot(df, eval_col, count_col, bins, labels, cmap="Set3", figsize=(14,6), min_pct=4):
    counts, pct = evalnote_by_countbin_table(df, eval_col, count_col, bins, labels)

    ax = pct.plot(kind="bar", stacked=True, figsize=figsize, colormap=cmap, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{eval_col} distribution by {count_col} (binned)", fontweight="bold")
    ax.set_xlabel(eval_col)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for j in range(pct.shape[1]):
        for i in range(pct.shape[0]):
            p = float(pct.iloc[i, j])
            c = int(counts.iloc[i, j])
            if c == 0 or p < min_pct:
                continue
            rect = ax.containers[j][i]
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_y() + rect.get_height()/2,
                    f"{c}\n{p:.1f}%",
                    ha="center", va="center", fontsize=8)

    ax.legend(title=f"{count_col} bin", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

bins_ko    = [-0.1, 0, 1, 2, 3, 4, 5]
labels_ko  = ["0","1","2","3","4","5"]
evalnote_by_countbin_plot(df, "evaluate_note", "nombre_prestation_ko", bins_ko, labels_ko)


bins_p     = [-0.1, 0, 1, 2, 3, 5, 8, 14, 10**9]
labels_p   = ["0","1","2","3-4","5-7","8-13","14","14+"]
evalnote_by_countbin_plot(df, "evaluate_note", "Nbr_ticket_pieces", bins_p, labels_p)


bins_i     = [-0.1, 0, 1, 2, 3, 5, 10, 20, 35, 10**9]
labels_i   = ["0","1","2","3-4","5-9","10-19","20-34","35","35+"]
evalnote_by_countbin_plot(df, "evaluate_note", "Nbr_ticket_information", bins_i, labels_i)


```
