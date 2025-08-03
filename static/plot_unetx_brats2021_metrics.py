import matplotlib.pyplot as plt

# Metric names and their corresponding values
metrics = [
    "DSC", "IoU", "Sensitivity", "Specificity", "PPV", "NPV"
]

# THIS MUST BE
values = [
    0.9200, 0.8518, 0.8968, 0.9829, 0.9444, 0.9671
]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color='skyblue', edgecolor='black')
plt.ylim(0, 1.05)
plt.title("Figure 4.7.2: EUNetX Performance Metrics on BraTS2021 Lung MRI Dataset")
plt.xlabel("Segmentation Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with exact values
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.4f}",
             ha='center', va='bottom', fontsize=10)

# Save and show
plt.tight_layout()
plt.savefig("results/EUNetX_brats2021_metrics_chart.png")
plt.show()
