import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set paths
csv_path = 'csv/evaluation_metrics.csv'  # Path to your CSV
plots_dir = 'plots'                      # Output directory for plots

# Create output directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Clean data: replace empty strings with NaN and convert numerics
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')

# --- Plot 1: DSC, IoU, HD95 ---
df_plot1 = df.set_index('Model')[['DSC', 'IoU', 'HD95']]
df_plot1.plot(kind='bar', figsize=(10, 6), title='Segmentation Metrics: DSC, IoU, HD95')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'plot_dsc_iou_hd95.png'))
plt.close()

# --- Plot 2: Sensitivity & Specificity ---
df_plot2 = df.set_index('Model')[['Sensitivity', 'Specificity']]
df_plot2.plot(kind='bar', figsize=(10, 6), title='Sensitivity and Specificity')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'plot_sens_spec.png'))
plt.close()

# --- Plot 3: Predictive Values (PPV, NPV) ---
df_plot3 = df.set_index('Model')[['PPV', 'NPV']]
df_plot3.plot(kind='bar', figsize=(10, 6), title='Predictive Values (PPV, NPV)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'plot_predictive_values.png'))
plt.close()

# Filter only EUNetX row
EUNetX_row = df[df['Model'].str.lower() == 'EUNetX']

if not EUNetX_row.empty:
    # Drop 'Model' column to get only metrics
    EUNetX_metrics = EUNetX_row.drop(columns=['Model']).iloc[0]

    # Label mapping
    rename_map = {
        'DSC': 'Dice Similarity Coefficient (DSC)',
        'HD95': 'HD95 (mm)',
        'IoU': 'Jaccard Index (IoU)',
        'Sensitivity': 'Sensitivity',
        'Specificity': 'Specificity',
        'PPV': 'PPV',
        'NPV': 'NPV'
    }

    # Metric order and formatted labels
    ordered_keys = ['NPV', 'PPV', 'Specificity', 'Sensitivity', 'HD95', 'IoU', 'DSC']
    metrics = {rename_map[k]: EUNetX_metrics[k] for k in ordered_keys if k in EUNetX_metrics}

    # Create horizontal bar plot
    plt.figure(figsize=(8, 5))
    bars = plt.barh(list(metrics.keys()), list(metrics.values()), color='gray')

    for bar, label in zip(bars, metrics.keys()):
        val = metrics[label]
        label_text = f"{val:.2f} mm" if "HD95" in label else f"{val:.4f}"
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, label_text,
                 va='center', fontsize=9)

    plt.xlim(0, max(metrics.values()) + 0.5)
    plt.title("EUNetX Evaluation Metrics Summary")
    plt.xlabel("Score / Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'EUNetX_summary_horizontal.png'))
    plt.close()

    print("Saved to:", os.path.join(plots_dir, 'eunetx_summary_horizontal.png'))
else:
    print("EUNetX not found in the dataset.")

