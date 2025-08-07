import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSVs
loss_df = pd.read_csv("csv/eunetx_epoch_loss.csv")
eval_df = pd.read_csv("csv/evaluation_metrics.csv")
expert_df = pd.read_csv("csv/expert_radiologist_metrics.csv")
perf_df = pd.read_csv("csv/system_performance_report.csv")

# 1. Plot Loss Curve ===
plt.figure(figsize=(10, 5))
plt.plot(loss_df["Epoch"], loss_df["Loss"], label="Training Loss", color="blue", marker='o')
plt.title("EUNetX Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/eunetx_loss_curve.png")
plt.close()


# # === 2. Bar Plot of Evaluation Metrics (excluding baseline NaNs) ===
# eval_clean = eval_df.dropna(subset=["DSC"])
# eval_clean.set_index("Model", inplace=True)
# eval_clean.plot(kind="bar", figsize=(12, 6))
# plt.title("Model Evaluation Metrics")
# plt.ylabel("Score")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.savefig("results/evaluation_metrics.png")
# plt.close()
#
# # === 3. Expert Metrics Plot ===
# expert_df.set_index("Model", inplace=True)
# expert_df.plot(kind="bar", figsize=(10, 5), color="green")
# plt.title("Expert Radiologist Metrics (Ground Truth)")
# plt.ylabel("Score")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.savefig("results/expert_metrics.png")
# plt.close()
#
# # === 4. Plot System Performance Summary (numeric values only) ===
# perf_numeric = {}
# for _, row in perf_df.iterrows():
#     try:
#         val = float(row["Value"])
#         perf_numeric[row["Metric"]] = val
#     except ValueError:
#         continue  # skip non-numeric
#
# # Plot
# plt.figure(figsize=(10, 5))
# plt.bar(perf_numeric.keys(), perf_numeric.values(), color='skyblue')
# plt.title("System Performance Summary")
# plt.ylabel("Value")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig("results/system_performance_summary.png")
# plt.close()

print(f"Plots and summaries saved")
