```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_nps_and_volume(
    df: pd.DataFrame,
    delay_col: str,
    note_col: str,
    delay_breaks: list,  # Custom delay intervals (e.g. [0, 30, 60, 120, 300, 5000])
    promoters_min: int = 9,
    detractors_max: int = 6,
):
    d = df[[delay_col, note_col]].dropna().copy()

    # Assign each row into a delay bin
    d['delay_bin'] = pd.cut(d[delay_col], bins=delay_breaks, right=False, include_lowest=True)

    # Define a function to calculate NPS
    def calculate_nps(scores):
        promoters = (scores >= promoters_min).sum()
        detractors = (scores <= detractors_max).sum()
        total = len(scores)
        if total == 0:
            return np.nan
        return (promoters - detractors) / total * 100

    # Aggregate: calculate NPS and volume per bin
    agg = d.groupby('delay_bin').agg(
        nps=(note_col, calculate_nps),
        count=(note_col, 'size')
    ).reset_index()

    # Calculate total volume for percentages
    total_count = agg['count'].sum()
    agg['percent'] = (agg['count'] / total_count) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bars for volume (count)
    ax1.bar(agg['delay_bin'].astype(str), agg['count'], alpha=0.6, label='Volume (Count)', color='tab:blue')
    ax1.set_xlabel(f'{delay_col} bins')
    ax1.set_ylabel('Volume (Count)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Add a secondary y-axis for NPS
    ax2 = ax1.twinx()
    ax2.plot(agg['delay_bin'].astype(str), agg['nps'], marker='o', color='tab:red', label='NPS (Net Promoter Score)')
    ax2.set_ylabel('NPS', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-100, 100)

    # Add percentage labels on top of the bars
    for i, val in enumerate(agg['count']):
        ax1.text(i, val, f'{val}\n({agg["percent"].iloc[i]:.1f}%)',
                 ha='center', va='bottom', fontsize=9, color='black')

    # Final layout
    fig.tight_layout()
    fig.suptitle("NPS and Volume Distribution by Delay Intervals", y=1.02)
    plt.show()

    return agg

plot_nps_and_volume(
    df,
    delay_col="delai_declaration",
    note_col="evaluate_note",
    delay_breaks=[0, 30, 60, 120, 300, 1000, 5000]
)

import matplotlib.pyplot as plt
import numpy as np

def plot_top_corr_bar(top_df, title="Top correlations with target", show_values=True):
    """
    top_df: DataFrame with column 'corr' and index = feature names
    """
    # sort so bars are ordered (negative -> positive)
    top_df = top_df.sort_values("corr")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df.index.astype(str), top_df["corr"])

    # ✅ force full correlation range
    ax.set_xlim(-1, 1)

    # optional: vertical line at 0 for readability
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("correlation")
    ax.set_title(title)

    # ✅ show value labels on each bar
    if show_values:
        for y, v in enumerate(top_df["corr"].values):
            # place text slightly to the right for positive, left for negative
            x = v + 0.03 if v >= 0 else v - 0.03
            ha = "left" if v >= 0 else "right"
            ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.show()


def top_corr_with_target_from_matrix(corr_mat, target_col, top_n=20):
    s = corr_mat[target_col].drop(target_col).dropna()
    out = (s.to_frame("corr")
             .assign(abs_corr=s.abs())
             .sort_values("abs_corr", ascending=False)
             .head(top_n)
             .drop(columns="abs_corr"))
    return out


```
