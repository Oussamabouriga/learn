```
This CatBoost error:

TypeError: must be real number, not NAType

means: you have pd.NA somewhere in your data (often in numeric columns).
CatBoost accepts np.nan, but doesn’t accept pd.NA inside the Pool.

So the fix is: convert every pd.NA to np.nan before creating the Pool, and make sure numeric columns are real numeric.

Put this block just before your Step 6 (Pool(...)):

# ==============================
# FIX for CatBoost Pool: remove pd.NA (NAType) -> convert to np.nan
# ==============================

# 1) Ensure code_postal is categorical string (as you want)
if "code_postal" in X_train.columns:
    X_train["code_postal"] = X_train["code_postal"].astype("string")
    X_test["code_postal"]  = X_test["code_postal"].astype("string")

# 2) Convert ALL pd.NA to np.nan (critical)
X_train = X_train.replace({pd.NA: np.nan})
X_test  = X_test.replace({pd.NA: np.nan})

# 3) Force numeric columns to be real numbers (NaN allowed)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c]  = pd.to_numeric(X_test[c], errors="coerce")

# 4) Convert any remaining missing tokens to np.nan safely
X_train = X_train.astype(object).where(pd.notna(X_train), np.nan)
X_test  = X_test.astype(object).where(pd.notna(X_test), np.nan)

print("Any pd.NA left in train?", (X_train.applymap(lambda v: v is pd.NA)).any().any())
print("Any pd.NA left in test ?", (X_test.applymap(lambda v: v is pd.NA)).any().any())

Then your Step 6 will work:

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool  = Pool(X_test, y_test, cat_features=cat_cols)

Why it happened (simple)
	•	Somewhere earlier you created missing values using pandas nullable types → they show as <NA> (pd.NA)
	•	CatBoost Pool cannot parse pd.NA as a number/string → crashes
	•	Converting to np.nan fixes it (CatBoost knows how to handle np.nan)

If it still errors after this, run this to find the exact column that contains pd.NA:

na_cols = [c for c in X_train.columns if (X_train[c].apply(lambda v: v is pd.NA)).any()]
print("Columns containing pd.NA:", na_cols)

```
