import matplotlib.pyplot as plt

# Data for the chart
models = ["EUNetX (Internal)", "EUNetX (External)", "TransUNet", "UNet++", "Swin-PANet", "MedT"]
datasets = ["Internal Set", "External Set", "Synapse", "Public Lung", "MedSeg Dataset", "MedT MRI"]
num_scans = [6, 1000, 245, 500, 30000, 1500]

# Create the plot
plt.figure(figsize=(12, 6))
bars = plt.bar(models, num_scans, color='mediumseagreen')

# Annotate bars with number of images
for bar, count in zip(bars, num_scans):
    plt.text(bar.get_x() + bar.get_width()/2, count + 500, str(count), ha='center', va='bottom', fontsize=10)

# Labels and formatting
plt.title("Figure X: Comparative Analysis of Datasets Used for EUNetX and Baseline Models", fontsize=14)
plt.ylabel("Number of Images/Scans", fontsize=12)
plt.ylim(0, 35000)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save or show
plt.tight_layout()
plt.savefig("results/dataset_model_comparison_chart.png")
plt.show()
