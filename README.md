# 💼 Salary Segment NPS Analysis

Excellent 👌 You’re now describing a real-world data analytics task —  
this is much more meaningful than simple averages.

Let’s build a **fully commented Python function** that does exactly what you described:

We’ll calculate **salary segments**, then compute:

- total number of people in each segment  
- number of **Promoters** (evaluation score ≥ 9)  
- number of **Passives** (evaluation score = 7 or 8)  
- number of **Detractors** (evaluation score ≤ 6)  
- and finally the **NPS** (Net Promoter Score = %Promoters − %Detractors)

---

## ✅ Full Example (ready to copy into your `.ipynb` notebook)

```python
# Import the pandas library for data manipulation and analysis
import pandas as pd

# Define a function to calculate NPS (Net Promoter Score) by salary segment
def calculate_nps_by_salary_segment(df, salary_col, eval_col):
    """
    Calculate NPS and counts by salary segment.
    
    Args:
        df (pd.DataFrame): Input data containing salaries and evaluation scores.
        salary_col (str): Name of the numeric column for salary.
        eval_col (str): Name of the numeric column for evaluation scores (0-10 scale).
    
    Returns:
        pd.DataFrame: Table with segment, counts, and NPS results.
    """

    # ─────────────────────────────
    # 1️⃣ Define salary segments
    # ─────────────────────────────

    # Define salary range boundaries (from 0 to 1000, 1000 to 3000, and greater than 3000)
    bins = [0, 1000, 3000, float('inf')]  

    # Define labels (names) for each salary segment
    labels = ["< 1000", "1000–3000", "> 3000"]

    # Create a new column "salary_segment" in the DataFrame
    # The pd.cut() function assigns each row to a range based on the salary value
    df["salary_segment"] = pd.cut(df[salary_col], bins=bins, labels=labels, right=False)

    # ─────────────────────────────
    # 2️⃣ Define Promoter / Passive / Detractor
    # ─────────────────────────────

    # Promoters: evaluation score 9 or 10 → Boolean True/False
    df["promoter"] = df[eval_col] >= 9

    # Passives (neutral): evaluation score 7 or 8 → Boolean True/False
    df["passive"] = df[eval_col].between(7, 8, inclusive="both")

    # Detractors: evaluation score 0 to 6 → Boolean True/False
    df["detractor"] = df[eval_col] <= 6

    # ─────────────────────────────
    # 3️⃣ Group by salary segment and count values
    # ─────────────────────────────

    # Group the dataset by the column "salary_segment"
    # and aggregate different counts and sums
    summary = (
        df.groupby("salary_segment")  # Group by salary segment
          .agg(
              total_people=(salary_col, "count"),  # Count how many rows in each segment
              promoters=("promoter", "sum"),       # Count how many promoters (True = 1)
              passives=("passive", "sum"),         # Count how many passives (True = 1)
              detractors=("detractor", "sum")      # Count how many detractors (True = 1)
          )
          .reset_index()  # Reset the index to make "salary_segment" a normal column again
    )

    # ─────────────────────────────
    # 4️⃣ Compute percentages and NPS
    # ─────────────────────────────

    # Calculate % of promoters per segment
    summary["%_promoters"] = (summary["promoters"] / summary["total_people"]) * 100

    # Calculate % of detractors per segment
    summary["%_detractors"] = (summary["detractors"] / summary["total_people"]) * 100

    # NPS = % of promoters - % of detractors
    summary["NPS"] = summary["%_promoters"] - summary["%_detractors"]

    # ─────────────────────────────
    # 5️⃣ Return a clean summary table
    # ─────────────────────────────

    # Select and order the final columns to display in the result
    return summary[[
        "salary_segment",      # Salary range label
        "total_people",        # Number of people in that segment
        "promoters",           # Count of promoters
        "passives",            # Count of passives
        "detractors",          # Count of detractors
        "%_promoters",         # Percentage of promoters
        "%_detractors",        # Percentage of detractors
        "NPS"                  # Calculated Net Promoter Score
    ]]



Example Usage



# Create an example dataset (replace this with your SQL data)
data = {
    "salary": [900, 1200, 2500, 3100, 4000, 2800, 1500, 800, 5000, 1800],
    "evaluation_score": [6, 8, 9, 10, 5, 7, 9, 4, 10, 6]
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Call the function, specifying column names
results = calculate_nps_by_salary_segment(df, salary_col="salary", eval_col="evaluation_score")

# Display the resulting summary DataFrame
results


Tips

You can easily adjust the salary segmentation ranges:

# Example: adding more refined ranges
bins = [0, 1000, 2000, 4000, float('inf')]
labels = ["<1000", "1000–1999", "2000–3999", "≥4000"]

If your SQL data already has column names like salaire or evaluationScore, just pass them to the function arguments:


results = calculate_nps_by_salary_segment(df, salary_col="salaire", eval_col="evaluationScore")