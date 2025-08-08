import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
loss_df = pd.read_csv("csv/eunetx_epoch_loss.csv")
eval_df = pd.read_csv("csv/evaluation_metrics.csv")
expert_df = pd.read_csv("csv/comparison_eunetx_vs_expert.csv")
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


# === 2. Bar Plot of Evaluation Metrics (excluding baseline NaNs) ===
eval_clean = eval_df.dropna(subset=["DSC"])
eval_clean.set_index("Model", inplace=True)
eval_clean.plot(kind="bar", figsize=(12, 6))
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/evaluation_metrics.png")
plt.close()

# === 3. Expert Metrics Plot ===
expert_df.set_index("Model", inplace=True)
expert_df.plot(kind="bar", figsize=(10, 5), color="green")
plt.title("Expert Radiologist Metrics (Ground Truth)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/expert_metrics.png")
plt.close()

# === 4. Plot System Performance Summary (numeric values only) ===
perf_numeric = {}
for _, row in perf_df.iterrows():
    try:
        val = float(row["Value"])
        perf_numeric[row["Metric"]] = val
    except ValueError:
        continue  # skip non-numeric

# Plot
plt.figure(figsize=(10, 5))
plt.bar(perf_numeric.keys(), perf_numeric.values(), color='skyblue')
plt.title("System Performance Summary")
plt.ylabel("Value")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/system_performance_summary.png")
plt.close()


# === Load CSV File ===
csv_path = "csv/eunetx_comparative_metrics_on_modalities.csv"
df = pd.read_csv(csv_path)

# === Clean and Convert Data ===
def clean_value(val):
    if isinstance(val, str):
        return float(val.replace('%', '').replace('mm', '').strip())
    return float(val)

df['CT'] = df['CT'].apply(clean_value)
df['MRI'] = df['MRI'].apply(clean_value)
df['Ultrasound'] = df['Ultrasound'].apply(clean_value)

# === Plotting Setup ===
metrics = df['Metric']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

# === Bar Charts ===
bars1 = ax.bar(x - width, df['CT'], width, label='CT', color='steelblue')
bars2 = ax.bar(x, df['MRI'], width, label='MRI', color='seagreen')
bars3 = ax.bar(x + width, df['Ultrasound'], width, label='Ultrasound', color='salmon')

# === Formatting ===
ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_ylabel('Score / Distance', fontsize=12)
ax.set_title('Comparative Evaluation Metrics of EUNetX on CT, MRI, and Ultrasound Datasets', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=30, ha='right')
ax.legend()
ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

# === Value Labels on Bars ===
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Show values on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


print(f"Plots and summaries saved")


# === Load CSV file ===
csv_path = "csv/EUNetX_Training_Loss_Comparison.csv"
df = pd.read_csv(csv_path)

# === Plot Setup ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary Y-axis (CT and MRI)
ax1.plot(df['Epoch'], df['CT'], label='CT Brain Tumor', linewidth=2, color='steelblue')
ax1.plot(df['Epoch'], df['MRI'], label='MRI Brain Tumor', linewidth=2, color='seagreen')
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("CT / MRI Training Loss", fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', linewidth=0.5)

# Secondary Y-axis (Ultrasound)
ax2 = ax1.twinx()
ax2.plot(df['Epoch'], df['Ultrasound'], label='Ultrasound Lung (COVID-19)', linewidth=2, color='salmon')
ax2.set_ylabel("Ultrasound Training Loss", fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Title
plt.title("Figure 23: Comparative Training Loss Curves of EUNetX Across Modalities", fontsize=14)
plt.tight_layout()
plt.show()


# Load data
csv_path = "csv/comparison_eunetx_vs_expert.csv"
df = pd.read_csv(csv_path)

# Setup
x = np.arange(len(df['Metric']))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Bars
bars1 = ax.bar(x - width/2, df['EUNetX'], width, label='EUNetX', color='steelblue')
bars2 = ax.bar(x + width/2, df['Expert'], width, label='Expert Radiologist', color='seagreen')

# Labels
ax.set_ylabel('Score / Distance', fontsize=12)
ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_title('Figure 27: Comparative Segmentation Performance of EUNetX vs. Expert Radiologist', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df['Metric'], rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# === Load CSV file ===
csv_path = "csv/EUNetX_vs_Baseline_Models.csv"  # Replace with your actual path
df = pd.read_csv(csv_path)

# === Convert numeric columns, handling missing values ===
metrics = df.columns[1:]
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Transpose for grouped bar chart ===
df_plot = df.set_index("Model").T

# === Plotting setup ===
fig, ax = plt.subplots(figsize=(14, 7))

bar_width = 0.13
index = np.arange(len(df_plot.index))
colors = ['steelblue', 'darkorange', 'seagreen', 'salmon', 'indigo']

# Plot each model
for i, model in enumerate(df_plot.columns):
    ax.bar(index + i * bar_width, df_plot[model], bar_width, label=model, color=colors[i % len(colors)])

# === Formatting ===
ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Score / Distance", fontsize=12)
ax.set_title("Comparative Performance of EUNetX and Baseline Models", fontsize=14)
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(df_plot.index, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Annotate values
for i, model in enumerate(df_plot.columns):
    for j, val in enumerate(df_plot[model]):
        if not np.isnan(val):
            ax.annotate(f'{val:.2f}',
                        xy=(j + i * bar_width, val),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.show()
