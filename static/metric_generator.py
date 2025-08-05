import pandas as pd
import matplotlib.pyplot as plt
import os


# METRIC FOR EUNetX
csv_path = 'csv/evaluation_metrics.csv'
df = pd.read_csv(csv_path)
eunetx_df = df[df['Model'] == 'EUNetX']
metrics = ['DSC', 'HD95', 'IoU', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
values = eunetx_df[metrics].values.flatten()
plt.figure(figsize=(10, 6))
plt.bar(metrics, values)
plt.title('Evaluation Metrics for EUNetX')
plt.ylabel('Score')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# EUNETX EPOCH
def plot_training_loss(csv_file_path):
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    # Load CSV into DataFrame
    df = pd.read_csv(csv_file_path)

    # Assumes columns are: 'epoch', 'loss'
    if not {'epoch', 'loss'}.issubset(df.columns):
        raise ValueError("CSV must contain 'epoch' and 'loss' columns.")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='blue')
    plt.title('Training Loss per Epoch for EUNetX')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()

    # Annotate figure description
    plt.figtext(0.5, -0.1,
        "The Figure shows the training loss over epochs for the EUNetX model. "
        "A steady decline indicates effective learning. Spikes or flat trends may indicate instability or poor convergence.",
        wrap=True, horizontalalignment='center', fontsize=10)

    plt.show()

csv_path = 'csv/eunetx_epoch_loss.csv'
plot_training_loss(csv_path)
