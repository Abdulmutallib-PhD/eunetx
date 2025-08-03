import matplotlib.pyplot as plt
import numpy as np

# Metrics
metrics = [
    "DSC", "IoU", "Sensitivity", "Specificity", "PPV", "NPV"
]
internal = [0.9544, 0.9128, 0.9374, 0.9980, 0.9721, 0.9954]
external = [0.9232, 0.8573, 0.9276, 0.9692, 0.9188, 0.9727]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, internal, width, label="Internal (CT Brain)")
plt.bar(x + width/2, external, width, label="External (MRI Lung)")

plt.ylabel("Score")
plt.title("Figure 4.17: Comparative Performance of EUNetX on Internal and External Datasets")
plt.xticks(x, metrics)
plt.ylim(0.85, 1.05)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("comparative_EUNetX_metrics.png")
plt.show()
