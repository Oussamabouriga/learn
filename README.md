Here you go (macOS + venv + parquet).

1) Create the venv (in ~/Documents/environment/)

# Create folders (if not exist)
mkdir -p ~/Documents/environment
mkdir -p ~/Documents/ai_project

# Create venv
python3 -m venv ~/Documents/environment/ai_project_venv

# Activate it
source ~/Documents/environment/ai_project_venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

2) Install dependencies (Parquet support)

# Best: pyarrow for parquet
python -m pip install pandas pyarrow

3) Go to the project and run

cd ~/Documents/ai_project

Create a script file:

nano step1_cleaning.py

Paste this code, save (Ctrl+O, Enter) then exit (Ctrl+X):

import pandas as pd
import numpy as np

# ========= CONFIG =========
PARQUET_PATH = "data.parquet"        # <- change to your parquet file name/path
OUTPUT_CSV   = "data_clean.csv"
OUTPUT_PARQUET = "data_clean.parquet"
DICT_XLSX    = "data_dictionary_template.xlsx"

# ========= 1) Load parquet =========
df = pd.read_parquet(PARQUET_PATH)   # requires pyarrow
print("Shape (rows, cols):", df.shape)
print("\nHead:")
print(df.head(5))

# ========= 2) Understand variables (profiling) =========
profile = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum().values,
    "missing_pct": (df.isna().mean() * 100).round(2).values,
    "n_unique": df.nunique(dropna=True).values
}).sort_values("missing_pct", ascending=False)

print("\n--- PROFILE (top 30 by missing %) ---")
print(profile.head(30).to_string(index=False))

# Data dictionary template (you fill descriptions manually)
def sample_values(series, k=5):
    vals = series.dropna().unique()[:k]
    return ", ".join(map(str, vals))

data_dictionary = pd.DataFrame({
    "variable": df.columns,
    "type": df.dtypes.astype(str).values,
    "business_description": [""] * len(df.columns),
    "example_values": [sample_values(df[c]) for c in df.columns]
})

# ========= 3) Data quality checks =========
dup_count = df.duplicated().sum()
print("\nDuplicate rows:", dup_count)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if num_cols:
    print("\n--- NUMERIC DESCRIBE ---")
    print(df[num_cols].describe().T.to_string())

# IQR outlier report (optional)
def iqr_outlier_report(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = data[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((data[c] < lo) | (data[c] > hi)).sum()
        rows.append({
            "column": c,
            "lower_bound": lo,
            "upper_bound": hi,
            "outlier_count": int(outliers)
        })
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)

if num_cols:
    outlier_report = iqr_outlier_report(df, num_cols)
    print("\n--- OUTLIER REPORT (top 20) ---")
    print(outlier_report.head(20).to_string(index=False))

# ========= 4) Cleaning =========
df_clean = df.copy()

# 4.1 Drop exact duplicates
df_clean = df_clean.drop_duplicates()

# 4.2 Replace common missing markers in string columns
missing_markers = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "-", "--"]
for c in cat_cols:
    # keep non-string types safe: cast only object/string columns
    if df_clean[c].dtype == "object" or "string" in str(df_clean[c].dtype).lower():
        df_clean[c] = df_clean[c].replace(missing_markers, np.nan)

# 4.3 Fill missing values (simple baseline rules)
# Numeric -> median, Categorical -> "Unknown"
for c in num_cols:
    if df_clean[c].isna().any():
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

for c in cat_cols:
    if df_clean[c].isna().any():
        df_clean[c] = df_clean[c].fillna("Unknown")

# 4.4 Standardize categorical text
for c in cat_cols:
    if df_clean[c].dtype == "object" or "string" in str(df_clean[c].dtype).lower():
        df_clean[c] = df_clean[c].astype(str).str.strip()
        # optional unify casing:
        # df_clean[c] = df_clean[c].str.lower()

print("\nBefore:", df.shape, "| After:", df_clean.shape)

# ========= 5) Summary + export =========
summary_after = pd.DataFrame({
    "column": df_clean.columns,
    "dtype": df_clean.dtypes.astype(str),
    "missing_count": df_clean.isna().sum().values,
    "missing_pct": (df_clean.isna().mean() * 100).round(2).values,
    "n_unique": df_clean.nunique(dropna=True).values
}).sort_values("missing_pct", ascending=False)

print("\n--- SUMMARY AFTER (top 30 by missing %) ---")
print(summary_after.head(30).to_string(index=False))

# Save outputs
df_clean.to_csv(OUTPUT_CSV, index=False)
df_clean.to_parquet(OUTPUT_PARQUET, index=False)  # uses pyarrow
data_dictionary.to_excel(DICT_XLSX, index=False)  # pandas will use openpyxl engine if available

print(f"\nSaved -> {OUTPUT_CSV}")
print(f"Saved -> {OUTPUT_PARQUET}")
print(f"Saved -> {DICT_XLSX}")

Install Excel writer (optional but recommended):

python -m pip install openpyxl

Run the script:

python step1_cleaning.py

4) Next time: activate + run (quick)

source ~/Documents/environment/ai_project_venv/bin/activate
cd ~/Documents/ai_project
python step1_cleaning.py


Yes — but only for real missing values (NaN / None).

df.dropna(subset=["feature"]) will drop rows where feature is NaN/None.

It will NOT drop rows where feature is an empty string "" or just spaces "   " (common in CSV/Excel exports).


To cover both cases (NaN + empty/space strings), do this:

import numpy as np

col = "feature"

# Convert empty/whitespace strings to NaN, then drop
df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
df = df.dropna(subset=[col])

If your column is numeric but stored as text, this is also useful:

df[col] = pd.to_numeric(df[col], errors="coerce")  # invalid -> NaN
df = df.dropna(subset=[col])

