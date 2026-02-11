```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_nps_and_volume(
    df: pd.DataFrame,
    delay_col: str,
    note_col: str,
    delay_breaks: list,
    promoters_min: int = 9,
    detractors_max: int = 6,
):
    d = df[[delay_col, note_col]].dropna().copy()

    # Assign each row into a delay bin
    d['delay_bin'] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    # NPS calculation
    def calculate_nps(scores):
        promoters = (scores >= promoters_min).sum()
        detractors = (scores <= detractors_max).sum()
        total = len(scores)
        if total == 0:
            return np.nan
        return (promoters - detractors) / total * 100

    # Aggregate
    agg = d.groupby('delay_bin').agg(
        nps=(note_col, calculate_nps),
        count=(note_col, 'size')
    ).reset_index()

    # Percent volume
    total_count = agg['count'].sum()
    agg['percent'] = (agg['count'] / total_count) * 100

    # Convert x labels once
    x_labels = agg['delay_bin'].astype(str)
    x = np.arange(len(agg))

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bars (volume)
    bars = ax1.bar(x, agg['count'], alpha=0.6, color='tab:blue')
    ax1.set_xlabel(f'{delay_col} bins')
    ax1.set_ylabel('Volume (Count)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha='right')

    # Volume labels (count + %)
    for i, val in enumerate(agg['count']):
        ax1.text(
            i,
            val,
            f'{val}\n({agg["percent"].iloc[i]:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # NPS line
    ax2 = ax1.twinx()
    ax2.plot(x, agg['nps'], marker='o', color='tab:red', linewidth=2)
    ax2.set_ylabel('NPS', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-100, 100)

    # ✅ NPS labels on points
    for i, val in enumerate(agg['nps']):
        if not np.isnan(val):
            ax2.annotate(
                f'{val:.1f}',
                (i, val),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=9,
                color='tab:red'
            )

    fig.tight_layout()
    fig.suptitle("NPS and Volume Distribution by Delay Intervals", y=1.02)
    plt.show()

    return agg

```
