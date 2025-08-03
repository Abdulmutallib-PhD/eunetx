import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV
df = pd.read_csv("csv/evaluation_metrics.csv")

# Ensure required columns exist and are named correctly
required_columns = ['Model', 'DSC', 'IoU', 'HD95']
df = df[required_columns]

# Replace NaN with 0 for plotting
dsc = df['DSC'].fillna(0).tolist()
iou = df['IoU'].fillna(0).tolist()
hd95 = df['HD95'].fillna(0).tolist()
models = df['Model'].tolist()

# Plotting
x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, dsc, width=width, label="DSC")
plt.bar(x, iou, width=width, label="IoU")
plt.bar(x + width, hd95, width=width, label="HD95")

plt.xticks(x, models, rotation=20)
plt.ylabel("Score / Distance (mm)")
plt.title("Comparative Metrics of EUNetX vs Baseline Models")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("results/comparative_metrics_with_hd95.png")
plt.show()
