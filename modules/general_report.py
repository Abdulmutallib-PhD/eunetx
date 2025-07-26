import pandas as pd
import matplotlib.pyplot as plt

# This can only be acquired after the testing and training training
train_losses_initial = []
val_losses_initial = []
train_losses_finetune = []
val_losses_finetune = []

# Create a DataFrame for CSV export
loss_df = pd.DataFrame({
    'Epoch': list(range(1, 6)) * 2,
    'Phase': ['Initial'] * 5 + ['Fine-tune'] * 5,
    'Train Loss': train_losses_initial + train_losses_finetune,
    'Validation Loss': val_losses_initial + val_losses_finetune
})

# Save to CSV
loss_csv_path = "/mnt/data/unetx_loss_curve.csv"
loss_df.to_csv(loss_csv_path, index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), train_losses_initial, 'r--', label='Train Loss (Initial)')
plt.plot(range(1, 6), val_losses_initial, 'b--', label='Val Loss (Initial)')
plt.plot(range(1, 6), train_losses_finetune, 'r-', label='Train Loss (Fine-tune)')
plt.plot(range(1, 6), val_losses_finetune, 'b-', label='Val Loss (Fine-tune)')
plt.xlabel('Epoch')
plt.ylabel('Dice Loss')
plt.title('UNetX Training and Fine-Tuning Loss Curve')
plt.legend()
plt.grid(True)

# Save plot
plot_path = "/mnt/data/unetx_loss_curve.png"
plt.savefig(plot_path)
plt.show()

(loss_csv_path, plot_path)
