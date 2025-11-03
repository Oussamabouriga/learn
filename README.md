import pandas as pd
from tabulate import tabulate

df_NPS_parcours_initial = cx.read_sql(query_NPS_parcours_initial, conn)

# Print a clean formatted table
print(tabulate(df_NPS_parcours_initial, headers='keys', tablefmt='psql', showindex=False))