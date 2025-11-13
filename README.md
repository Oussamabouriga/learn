import matplotlib.pyplot as plt

# ---------------------------
# 1. Replace these with your data
# ---------------------------
periods = ["1h", "2h", "3d", "4d"]  # X-axis labels
nps_values = [20, 35, 10, 45]       # Y-axis NPS values
# ---------------------------

plt.figure(figsize=(10, 6))

# Plot line + points
plt.plot(periods, nps_values, marker='o')

# Add text label next to each point
for x, y in zip(periods, nps_values):
    plt.text(x, y, f"{y}", ha='left', va='bottom')

# Axis labels & title
plt.xlabel("Period")
plt.ylabel("NPS")
plt.title("NPS Evolution Over Time")

# Optional: grid for readability
plt.grid(True)

plt.tight_layout()
plt.show()




22222222

import matplotlib.pyplot as plt

# ---------------------------
# Replace with your real data
# ---------------------------
programs = ["Program A", "Program B", "Program C", "Program D"]
nps_values = [45, 20, 10, 55]
# ---------------------------

plt.figure(figsize=(10, 6))

# Create bar chart
plt.bar(programs, nps_values)

# Add labels on top of each bar
for i, value in enumerate(nps_values):
    plt.text(i, value, f"{value}", ha='center', va='bottom')

# Labels & title
plt.xlabel("Program")
plt.ylabel("NPS")
plt.title("NPS Variation by Program")

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



3333333333
import matplotlib.pyplot as plt

# ---------------------------
# Replace with your real data
# ---------------------------
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
nps_values = [25, 30, 28, 40, 35, 45, 50, 47]
# ---------------------------

plt.figure(figsize=(12, 6))

# Line plot with markers
plt.plot(months, nps_values, marker='o')

# Add NPS labels next to each point
for x, y in zip(months, nps_values):
    plt.text(x, y, f"{y}", ha='left', va='bottom')

# Labels & title
plt.xlabel("Month")
plt.ylabel("NPS")
plt.title("NPS Evolution Over the Months")

# Add grid
plt.grid(True)

plt.tight_layout()
plt.show()


44444444444444444
import matplotlib.pyplot as plt

# ---------------------------
# Replace with your real data
# ---------------------------
periods = ["1h", "2h", "3d", "4d"]     # X-axis labels
nps_values = [20, 35, 10, 45]          # NPS curve
delta_sat_values = [5, 15, -5, 20]     # Delta SAT curve (example)
# ---------------------------

plt.figure(figsize=(10, 6))

# NPS curve
plt.plot(periods, nps_values, marker='o', label="NPS")

# Delta SAT curve (blue)
plt.plot(periods, delta_sat_values, marker='o', color='blue', label="Delta SAT")

# Add text labels for NPS
for x, y in zip(periods, nps_values):
    plt.text(x, y, f"{y}", ha='left', va='bottom')

# Add text labels for Delta SAT
for x, y in zip(periods, delta_sat_values):
    plt.text(x, y, f"{y}", ha='right', va='bottom', color='blue')

# Axis labels & title
plt.xlabel("Period")
plt.ylabel("Score")
plt.title("NPS & Delta SAT Evolution Over Time")

# Legend
plt.legend()

# Grid
plt.grid(True)

plt.tight_layout()
plt.show()
