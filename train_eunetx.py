import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from torch.nn.functional import interpolate
from sklearn.metrics import jaccard_score, confusion_matrix
import time

from modules.dataset_loader import DicomAndPNGDataset
from modules.model_eunetx import EUNetX
from utils.utils import dice_loss

# -------------------------
# Evaluation Helper Function
# -------------------------
def evaluate_full_metrics(model, val_loader, threshold=0.5):
    model.eval()
    dsc_list, jaccard_list, sens_list, spec_list = [], [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(next(model.parameters()).device)
            pred = model(x)

            if pred.shape != y.shape:
                pred = torch.nn.functional.interpolate(pred, size=y.shape[2:], mode="trilinear", align_corners=False)

            pred_bin = (pred.cpu().numpy() > threshold).astype(np.uint8)
            y_np = y.cpu().numpy().astype(np.uint8)

            for p, t in zip(pred_bin, y_np):
                p_flat = p.flatten()
                t_flat = t.flatten()

                try:
                    dsc = (2 * np.sum(p_flat * t_flat)) / (np.sum(p_flat) + np.sum(t_flat) + 1e-8)
                    jacc = jaccard_score(t_flat, p_flat, average='binary', zero_division=1)
                    tn, fp, fn, tp = confusion_matrix(t_flat, p_flat, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn + 1e-8)
                    spec = tn / (tn + fp + 1e-8)

                except Exception as e:
                    print(f"Warning: Skipping one pair due to error: {e}")
                    continue

                dsc_list.append(dsc)
                jaccard_list.append(jacc)
                sens_list.append(sens)
                spec_list.append(spec)

    os.makedirs("results", exist_ok=True)

# Setup
print("Libraries loaded.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
images_path = 'collection/nonannotated/images'
masks_path = 'collection/annotated/images'
dataset = DicomAndPNGDataset(images_path, masks_path)

# Split dataset
train_len = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)

# Initialize model
model = EUNetX().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 2
train_losses, val_losses = [], []

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)

        if pred.shape != y.shape:
            pred = interpolate(pred, size=y.shape[2:], mode="trilinear", align_corners=False)

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
            pred = model(x)
            if pred.shape != y.shape:
                pred = interpolate(pred, size=y.shape[2:], mode="trilinear", align_corners=False)
            val_loss += dice_loss(pred, y).item()
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}: Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")

print("Validation process completed")
print("Please wait while initializing the next step...")
time.sleep(3)
# Execute the external plotting script
os.system("python modules/prediction.py")
