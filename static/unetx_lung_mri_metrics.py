import matplotlib.pyplot as plt

# Example simulated loss values over 50 epochs for internal dataset
epochs = list(range(1, 51))
train_loss = [0.5 - 0.003*i for i in range(50)]  # Example curve
val_loss = [0.52 - 0.0032*i for i in range(50)]  # Example curve

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
plt.title("Figure 4.15: UNetX Loss Curve on Internal CT Brain Dataset")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("unetx_internal_loss_curve.png")
plt.show()
