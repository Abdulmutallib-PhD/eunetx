import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_loader import DicomDataset
from model_unetx import UNetX
from utils import dice_loss
from evaluate_unetx import evaluate_full_metrics

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = DicomDataset('data/external_dataset/images', 'data/external_dataset/masks')
train_len = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# Model
model = UNetX().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = dice_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += dice_loss(model(x), y).item()
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}: Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")

# Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Training Curve"); plt.show()

# Evaluate
# results, csv_path = evaluate_full_metrics(model, val_loader)
metrics, evaluation_metrics, result_path, metrics_path = evaluate_full_metrics(model, val_loader)
print("Final evaluation results saved to:", evaluation_metrics)
