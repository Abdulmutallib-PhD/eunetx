import matplotlib.pyplot as plt

epochs = list(range(1, 51))
losses = [
    539.5341, 356.0265, 302.7677, 289.2611, 269.9644, 254.9091, 244.5755, 235.5515,
    226.9333, 225.1008, 214.9798, 212.4984, 206.9233, 202.6254, 198.2333, 192.7617,
    193.2018, 189.5327, 185.0002, 183.0245, 180.2390, 177.8745, 176.2935, 174.8098,
    173.5228, 170.7860, 169.6393, 167.7392, 166.9470, 165.0588, 162.1764, 161.7352,
    159.4165, 156.8854, 156.7957, 155.7956, 153.3522, 152.7173, 152.5957, 151.3846,
    148.1616, 148.0091, 148.4371, 146.1911, 144.6642, 145.4476, 143.2755, 143.0381,
    141.2187, 141.3301
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
plt.title("Figure 4.7.1: Training Loss Curve for EUNetX on BraTS2021 Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/EUNetX_brats2021_loss_curve.png")
plt.show()
