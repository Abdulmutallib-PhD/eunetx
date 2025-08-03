import pandas as pd
import matplotlib.pyplot as plt

# Load training loss from CSV
df = pd.read_csv("csv/eunetx_epoch_loss.csv")

# Extract epoch and loss
epochs = df["Epoch"].tolist()
train_loss = df["Loss"].tolist()

# Use train_loss as val_loss if not available
val_loss = train_loss  # or load another column if you save it separately

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='x')  # optional
plt.title("EUNetX Loss Curve on External/External MRI Lung Dataset (Pls review!!!!!")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/eunetx_external_loss_curve.png")
plt.show()
