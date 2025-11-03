import connectorx as cx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# ────────────────────────────────
# 1️⃣  SQL Query
# ────────────────────────────────
query = """
SELECT 
    sin.Ticket_Id AS ticket,
    iddeclaratif,
    step_name,
    FLOOR(TIMESTAMPDIFF(MINUTE, date_debut, date_fin) / 60) AS DUREE,
    nps.evaluate_note AS nps_note,
    nps.evaluatestars AS nps_stars,
    COALESCE(nps.evaluate_note, nps.evaluatestars) AS nps_value
FROM karapass_v2_sysmik_all_v2 AS sin
INNER JOIN karapass_v2_sysmik_nps AS nps
    ON sin.Ticket_Id = nps.ticketid
WHERE DATE(sin.Closed_Time) BETWEEN '2025-01-01' AND DATE(CURRENT_DATE)
  AND (nps.evaluate_note IS NOT NULL OR nps.evaluatestars IS NOT NULL)
  AND sin.Type = 'Sinistre'
  AND sin.Sinistre_accepte = 1
  AND sin.idprogramme IN (30, 42, 81, 82, 83, 108, 109, 110)
  AND sin.Requester_Email NOT LIKE '%aptoriel.fr%'
  AND source_crm LIKE 'CRMK'
"""

# ────────────────────────────────
# 2️⃣  Load data from DB
# ────────────────────────────────
conn = "mysql://username:password@host:3306/database_name"
df = cx.read_sql(conn, query)

print(f"✅ Loaded {len(df):,} rows from DB")
print(tabulate(df.head(5), headers='keys', tablefmt='psql', showindex=False))

# ────────────────────────────────
# 3️⃣  Data preparation
# ────────────────────────────────
# Convert durée to hours
df["DUREE"] = df["DUREE"].astype(float)

# ────────────────────────────────
# 4️⃣  NPS calculation per step
# ────────────────────────────────
def compute_nps(sub_df):
    promoters = ((sub_df["nps_value"] >= 9).sum())
    detractors = ((sub_df["nps_value"] <= 6).sum())
    total = sub_df["nps_value"].notna().sum()
    return ((promoters - detractors) / total * 100) if total > 0 else None

nps_by_step = (
    df.groupby("step_name")
      .apply(compute_nps)
      .reset_index()
      .rename(columns={0: "NPS"})
)

# ────────────────────────────────
# 5️⃣  Delay statistics per step
# ────────────────────────────────
stats = (
    df.groupby("step_name")["DUREE"]
      .agg(["count", "mean", "median", "max"])
      .reset_index()
      .rename(columns={"count": "nombre", "mean": "moyenne", "median": "Mediane", "max": "Max"})
)

# Merge both (duration stats + NPS)
merged = stats.merge(nps_by_step, on="step_name", how="left")

print("\n📊 Moyenne et NPS par étape :")
print(tabulate(merged, headers='keys', tablefmt='psql', showindex=False))

# ────────────────────────────────
# 6️⃣  Plot NPS curve vs step_name
# ────────────────────────────────
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")

sns.lineplot(
    data=merged,
    x="step_name",
    y="NPS",
    marker="o",
    linewidth=2,
    color="blue"
)

for i, row in merged.iterrows():
    plt.text(i, row["NPS"], f"{row['NPS']:.1f}", ha="center", va="bottom", fontsize=9)

plt.title("📉 Évolution du NPS par étape (délai)", fontsize=13)
plt.xlabel("Nom de l’étape")
plt.ylabel("NPS (%)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ────────────────────────────────
# 7️⃣  Optional: Correlation between delay and NPS
# ────────────────────────────────
plt.figure(figsize=(7,5))
sns.regplot(data=merged, x="moyenne", y="NPS", ci=None, scatter_kws={"s":60})
plt.title("Relation entre la durée moyenne et le NPS")
plt.xlabel("Durée moyenne (heures)")
plt.ylabel("NPS (%)")
plt.grid(True)
plt.tight_layout()
plt.show()