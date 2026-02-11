```
import matplotlib.pyplot as plt
import pandas as pd

def plot_delay_scatter_colored_by_note(
    df: pd.DataFrame,
    delay_col: str,
    note_col: str,
    delay_range=None,
    jitter_y=False,     # True if many points overlap
    alpha=0.6,
    size=18
):
    d = df[[delay_col, note_col]].dropna().copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    # Optional jitter on Y to see overlapping points
    if jitter_y:
        import numpy as np
        d[note_col] = d[note_col] + np.random.uniform(-0.12, 0.12, size=len(d))

    # make note categorical (so same note => same color)
    d["note_cat"] = d[note_col].astype(int).astype(str)

    plt.figure(figsize=(12, 6))

    # plot each note separately => different color automatically
    for note_value in sorted(d["note_cat"].unique(), key=lambda x: int(x)):
        sub = d[d["note_cat"] == note_value]
        plt.scatter(
            sub[delay_col],
            sub[note_col],
            s=size,
            alpha=alpha,
            label=note_value
        )

    plt.xlabel(delay_col)
    plt.ylabel(note_col)
    plt.title(f"Scatter: {note_col} by {delay_col} (color = note)")
    plt.grid(True, alpha=0.3)

    # if your notes are 0..10, keep fixed y-limits
    plt.ylim(-0.5, 10.5)

    plt.legend(title="Note", ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

plot_delay_scatter_colored_by_note(
    df,
    delay_col="delai_declaration",
    note_col="evaluate_note",
    delay_range=(0, 5000),
    jitter_y=True
)

```
