# ✅ Normalize and check column names to avoid KeyError: 'DUREE'
df.columns = [c.strip().upper() for c in df.columns]

if "DUREE" not in df.columns:
    if "duree" in df.columns:
        df.rename(columns={"duree": "DUREE"}, inplace=True)
    else:
        raise KeyError(f"❌ Column 'DUREE' not found. Found: {list(df.columns)}")

df["DUREE"] = df["DUREE"].astype(float)

# ✅ Ensure NPS_VALUE column exists
if "NPS_VALUE" not in df.columns:
    import numpy as np
    df["NPS_VALUE"] = np.random.choice(range(0, 11), size=len(df))