import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("csv/dataset_volume.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove commas and convert counts to integers
df["Number of Images/Scans"] = df["Number of Images/Scans"].str.replace(",", "").astype(int)

# Extract values
models = df["Model"].tolist()
num_scans = df["Number of Images/Scans"].tolist()

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(models, num_scans, color='mediumseagreen')

# Annotate bars
for bar, count in zip(bars, num_scans):
    plt.text(bar.get_x() + bar.get_width()/2, count + 500, str(count), ha='center', va='bottom', fontsize=10)

plt.title("Comparative Analysis of Datasets Used for EUNetX and Baseline Models", fontsize=14)
plt.ylabel("Number of Images", fontsize=12)
plt.ylim(0, max(num_scans) + 5000)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("results/dataset_model_comparison_chart.png")
plt.show()
