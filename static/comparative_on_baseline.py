import matplotlib.pyplot as plt
import numpy as np

models = ["UNet++", "TransUNet", "Swin-PANet", "MedT", "EUNetX"]
dsc = [None, 0.8839, 0.9142, 0.8102, 0.9544]
iou = [0.9252, None, 0.8488, 0.6961, 0.9128]
hd95 = [None, None, None, None, 2.21]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, [v if v is not None else 0 for v in dsc], width=width, label="DSC")
plt.bar(x, [v if v is not None else 0 for v in iou], width=width, label="IoU")
plt.bar(x + width, [v if v is not None else 0 for v in hd95], width=width, label="HD95")

plt.xticks(x, models, rotation=20)
plt.ylabel("Score / Distance (mm)")
plt.title("Figure 4.18: Comparative Metrics of EUNetX vs Baseline Models")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("results/comparative_metrics_with_hd95.png")
plt.show()
