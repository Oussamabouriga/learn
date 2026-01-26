

0️⃣ Pandas Mental Model (Very Important)

Think of Pandas as Excel + SQL + Python:

DataFrame = table (rows × columns)

Series = one column

Operations are vectorized (fast, no loops)


Golden rule:

> If you can avoid a for loop, Pandas probably has a built-in function.




---

1️⃣ Loading & Saving Data

Why it matters

You will constantly move data between files, databases, and notebooks.

Load

pd.read_csv("file.csv")
pd.read_parquet("file.parquet")
pd.read_excel("file.xlsx")

Save

df.to_csv("out.csv", index=False)
df.to_parquet("out.parquet")
df.to_excel("out.xlsx", index=False)

👉 index=False avoids saving row numbers.


---

2️⃣ First Inspection (ALWAYS DO THIS)

df.head()
df.tail()
df.sample(5)
df.shape
df.columns
df.dtypes
df.info()
df.describe()

Why

Detect wrong types

Detect missing values

Understand scale & ranges



---

3️⃣ Selecting Data (Core Skill)

Columns

df["Age"]
df[["Age", "gender"]]

Rows

df.loc[0]      # label
df.iloc[0]     # position

Rows + columns

df.loc[df["Age"] >= 18, ["Age", "evaluate_note"]]


---

4️⃣ Filtering Rows (Most Used)

Why it matters

Filtering is how you answer most questions:

“People age 18–24?”

“Only rows where evaluate_note = 10?”

“cadeau contains virement?”


✅ Basic filter (keep rows that match)

df[df["Age"] > 18]
df[df["evaluate_note"] == 10]

✅ Multiple conditions (AND / OR)

# AND: both conditions must be true
subset = df[(df["Age"] >= 18) & (df["Age"] <= 24)]

# OR: at least one is true
subset2 = df[(df["gender"] == 1) | (df["car"] == 1)]

✅ Cleaner ranges with between()

df[df["Age"].between(18, 24)]

✅ Negation (NOT)

# keep rows where Age is NOT 0
filtered = df[df["Age"] != 0]

✅ Filter text contains word (substring)

# cadeau contains "virement" (case-insensitive)
df[df["cadeau"].astype(str).str.contains("virement", case=False, na=False)]

✅ query() (SQL-like filtering)

# easier to read for complex conditions
subset = df.query("Age >= 18 and Age <= 24 and evaluate_note == 3")


---

5️⃣ Deleting Rows & Columns (drop / delete)

A) Delete rows by condition (most common)

Keep what you want, everything else is removed.

# delete rows where Age == 0
df = df[df["Age"] != 0]

# delete rows where Age < 18
df = df[df["Age"] >= 18]

B) Delete rows with missing values

# delete rows with ANY missing value
clean = df.dropna()

# delete rows only if specific columns are missing
clean = df.dropna(subset=["Age", "evaluate_note"])

# delete rows only if ALL columns are missing
clean = df.dropna(how="all")

C) Delete duplicate rows

# remove exact duplicates
clean = df.drop_duplicates()

# remove duplicates based on a key
clean = df.drop_duplicates(subset=["person_id"], keep="first")

D) Delete columns

# drop one column
clean = df.drop(columns=["unused_col"])

# drop multiple columns
clean = df.drop(columns=["c1", "c2"])


---

6️⃣ Missing Values (NaN, None, empty)

Detect

df.isna().sum()           # missing per column
df.isna().sum().sum()     # total missing

Show rows with NaN

df[df["Age"].isna()]

Drop

# drop rows
clean = df.dropna()
clean = df.dropna(subset=["Age"])

# drop columns that are too empty
clean = df.dropna(axis=1, thresh=int(0.5 * len(df)))  # keep cols with >= 50% non-missing

Fill

df["Age"] = df["Age"].fillna(0)
df["Age"] = df["Age"].fillna(df["Age"].median())

Empty strings → NaN (important for text)

# matches "" and "   "
df = df.replace(r"^\s*$", np.nan, regex=True)


---

7️⃣ Updating Data (edit values / create rules)

A) Update a whole column

# convert type
df["Age"] = df["Age"].astype(int)

# set all missing to 0
df["Age"] = df["Age"].fillna(0)

B) Update values by condition (like SQL UPDATE WHERE)

# set Age=0 to NaN
import numpy as np

df.loc[df["Age"] == 0, "Age"] = np.nan

# cap values (clip)
df["Age"] = df["Age"].clip(lower=0, upper=100)

C) Replace categories / normalize text

# replace values

df["gender"] = df["gender"].replace({"M": 0, "F": 1})

# standardize text

df["city"] = df["city"].astype(str).str.strip().str.lower()

D) Create a new column based on rules

# simple rule

df["is_adult"] = df["Age"] >= 18

# multi-condition with np.where
import numpy as np

df["age_group"] = np.where(df["Age"] < 25, "18-24", "25+")


---

8️⃣ Counting & Distributions (counts, %)

A) Count rows

len(df)         # number of rows
df.shape[0]     # same

B) Count occurrences of each value (distribution)

# counts per score
counts = df["evaluate_note"].value_counts().sort_index()

# percentages per score
pct = df["evaluate_note"].value_counts(normalize=True).sort_index() * 100

C) Count a condition (how many rows match)

# how many Age == 0
n_zero = (df["Age"] == 0).sum()

# how many between 18 and 24
n_1824 = df["Age"].between(18, 24).sum()

D) Build a table with count + percentage

vc = df["evaluate_note"].value_counts().sort_index().to_frame("count")
vc["percentage"] = (vc["count"] / vc["count"].sum() * 100).round(2)
vc

E) Distributions by group (gender, car, etc.)

# distribution of evaluate_note by gender (counts)
dist = df.groupby("gender")["evaluate_note"].value_counts().unstack(fill_value=0)

# convert to % within gender
pct = (dist.div(dist.sum(axis=1), axis=0) * 100).round(2)


---

9️⃣ GroupBy (THE MOST IMPORTANT)

(NaN, None, empty)

Detect

df.isna().sum()
df.isna().sum().sum()

Show rows with NaN

df[df["Age"].isna()]

Drop

df.dropna()
df.dropna(subset=["Age"])

Fill

df["Age"].fillna(0)
df["Age"].fillna(df["Age"].median())

Empty strings → NaN

df.replace(r"^\\s*$", np.nan, regex=True)


---

6️⃣ Data Types (CRITICAL)

Convert

df["Age"] = df["Age"].astype(int)
df["gender"] = df["gender"].astype(int)
df["city"] = df["city"].astype("string")

Safe numeric conversion

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

Convert all object → string

df = df.convert_dtypes()


---

7️⃣ String Operations (Text Data)

df["cadeau"].str.lower()
df["cadeau"].str.strip()
df["cadeau"].str.contains("virement", case=False, na=False)
df["cadeau"].str.split(",")

Filter containing word

df[df["cadeau"].str.contains("virement", case=False, na=False)]


---

8️⃣ Counting & Distributions

Counts

df["evaluate_note"].value_counts()
df["evaluate_note"].value_counts(normalize=True)

Count condition

(df["Age"] == 0).sum()


---

9️⃣ GroupBy (THE MOST IMPORTANT)

Why

GroupBy answers questions like:

average score per group

distribution per category


Basic

df.groupby("gender")["evaluate_note"].mean()

Multiple columns

df.groupby(["gender", "car"])["evaluate_note"].mean()

Count

df.groupby("evaluate_note").size()

Multiple aggregations

df.groupby("nombre_ko")["evaluate_note"].agg(["mean", "count", "std"])


---

🔟 Pivot & Reshape

Pivot (wide)

df.pivot(index="Age", columns="gender", values="evaluate_note")

Melt (long)

df.melt(
    id_vars=["Age"],
    value_vars=["evaluate_note", "nombre_ko"],
    var_name="metric",
    value_name="value"
)


---

1️⃣1️⃣ Sorting

df.sort_values("Age")
df.sort_values("evaluate_note", ascending=False)
df.sort_index()


---

1️⃣2️⃣ Create New Columns

df["is_adult"] = df["Age"] >= 18
df["ko_ratio"] = df["nombre_ko"] / 5


---

1️⃣3️⃣ Apply vs Vectorization

Apply (slower)

df["Age_group"] = df["Age"].apply(lambda x: "young" if x < 30 else "old")

Vectorized (better)

df["Age_group"] = np.where(df["Age"] < 30, "young", "old")


---

1️⃣4️⃣ Map & Replace

df["gender_label"] = df["gender"].map({0: "Male", 1: "Female"})

df["gender"] = df["gender"].replace({"M": 0, "F": 1})


---

1️⃣5️⃣ Duplicates

df.duplicated().sum()
df.drop_duplicates()
df.drop_duplicates(subset=["person_id"])


---

1️⃣6️⃣ Joins & Concatenation

pd.merge(df1, df2, on="id", how="left")
pd.concat([df1, df2])


---

1️⃣7️⃣ Time Series Basics

df["date"] = pd.to_datetime(df["date"])
df.set_index("date").resample("M").mean()


---

1️⃣8️⃣ Performance Tips (IMPORTANT)

Avoid loops

Prefer vectorized ops

Filter early

Use category for low-cardinality columns


df["gender"] = df["gender"].astype("category")


---

1️⃣9️⃣ Deleting Rows & Columns (VERY COMMON)

Delete rows by condition

df = df[df["Age"] >= 18]
df = df[df["evaluate_note"] != 0]

Delete rows with missing values

df = df.dropna()
df = df.dropna(subset=["Age", "evaluate_note"])

Delete columns

df = df.drop(columns=["unwanted_column"])

Delete rows by index

df = df.drop(index=[0, 1, 2])


---

2️⃣0️⃣ Updating / Modifying Values

Update values conditionally

df.loc[df["Age"] < 18, "Age"] = 18

Replace values

df["gender"] = df["gender"].replace({"M": 0, "F": 1})

Update using another column

df["score_normalized"] = df["evaluate_note"] / 10


---

2️⃣1️⃣ Advanced Filtering Patterns

Filter with multiple conditions

df[(df["Age"] >= 18) & (df["gender"] == 1)]

Filter using isin

df[df["evaluate_note"].isin([8, 9, 10])]

Filter text contains

df[df["cadeau"].str.contains("virement", case=False, na=False)]


---

2️⃣2️⃣ Counting & Distribution Patterns (EXAM / REAL LIFE)

Count rows

len(df)
df.shape[0]

Distribution of a column

df["evaluate_note"].value_counts()
df["evaluate_note"].value_counts(normalize=True) * 100

Distribution by group

df.groupby("gender")["evaluate_note"].value_counts()

Count by condition

(df["Age"] < 18).sum()


---

2️⃣3️⃣ Building Clean Distribution Tables (CORE SKILL)

Counts matrix

df.groupby(["evaluate_note", "nombre_ko"]).size().unstack(fill_value=0)

Percentage matrix

tmp = df.groupby(["evaluate_note", "nombre_ko"]).size()
pct = tmp / tmp.groupby(level=0).sum() * 100
pct.unstack(fill_value=0)


---

2️⃣4️⃣ Updating Types After Cleaning

Convert after filtering

df["Age"] = df["Age"].astype(int)
df["evaluate_note"] = df["evaluate_note"].astype(int)

Convert to category (performance)

df["gender"] = df["gender"].astype("category")


---

2️⃣5️⃣ Debugging Pandas (VERY IMPORTANT)

Check what changed

df.shape
df.info()
df.head()

Check unexpected values

df["Age"].unique()
df["evaluate_note"].value_counts()


---

🧠 Golden Rules (Memorize These)

Inspect first (info, describe)

Clean before analysis

Filter early, not late

Percentages for comparison

Counts for volume

Tables → tabulate

Plots → numeric DataFrames only

Avoid loops, use vectorization


Qualitative (Categorical) and Quantitative (Numerical).
Here is the simple breakdown of the four types shown:
1. Qualitative Data (Categories)
This describes qualities or characteristics.
 * Nominal Data: Used for naming or labeling variables without any specific order.
   * Example: Colors, Gender, or Country names.
 * Ordinal Data: Data that follows a clear order or rank, but the difference between them isn't measured.
   * Example: Satisfaction ratings (Happy/Neutral/Sad) or finishing a race in 1st, 2nd, or 3rd place.
2. Quantitative Data (Numbers)
This describes measurable quantities.
 * Discrete Data: Countable numbers that are usually whole (you can't have half a person).
   * Example: The number of students in a class or the number of cars in a lot.
 * Continuous Data: Measurable values that can be broken down into infinite decimals.
   * Example: A person's height, weight, or the exact temperature outside.
In short:
 * Nominal: Labels only.
 * Ordinal: Labels in order.
 * Discrete: Counted numbers.
 * Continuous: Measured numbers.




