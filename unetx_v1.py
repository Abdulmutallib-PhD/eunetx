# -*- coding: utf-8 -*-
"""
UNetX Brain Tumor Segmentation using CT (PyTorch)

Note: This section is for testing the model using dummy data. You must have high end computer
with GPU. Also make sure all env are installed using the requirements.txt file
"""

# ==============================================================
# System and Data Handling Libraries
# ==============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

print("Libraries loaded.")

# ==============================================================
# Check device
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================
# Model Components — UNetX Blocks
# ==============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class LightweightFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(x)

class UNetX(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        self.lff = nn.ModuleList()

        for feat in features:
            self.encoder.append(ConvBlock(in_channels, feat))
            in_channels = feat

        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        for feat in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feat*2, feat))
            self.lff.append(LightweightFeatureFusion(feat))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = self.lff[idx//2](skips[idx//2])
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)

        return torch.sigmoid(self.final_conv(x))

# ==============================================================
# Data Preparation
# ==============================================================
class DummyCTDataset(Dataset):
    def __init__(self, n=200):
        self.X = [torch.randn(1,256,256) for _ in range(n)]
        self.Y = [torch.randint(0,2,(1,256,256),dtype=torch.float32) for _ in range(n)]

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

dataset = DummyCTDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ==============================================================
# Loss and Metrics
# ==============================================================
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2.*intersection + smooth)/(pred.sum() + target.sum() + smooth))

# ==============================================================
# Training Loop
# ==============================================================
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# ==============================================================
# Main Training
# ==============================================================
model = UNetX().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5

train_losses, val_losses = [], []
for epoch in range(epochs):
    t_loss = train(model, train_loader, optimizer, dice_loss)
    v_loss = validate(model, val_loader, dice_loss)
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

# ==============================================================
# Results Presentation
# ==============================================================
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.legend()
plt.title("UNetX Training & Validation Loss")
plt.show()

# ==============================================================
# Evaluate DSC, Sensitivity & Specificity on Validation Data
# ==============================================================
def evaluate_metrics(model, loader):
    dscs, sens, specs = [], [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = (model(x) > 0.5).float()
        y_flat, out_flat = y.view(-1), out.view(-1)

        TP = (y_flat*out_flat).sum()
        FP = ((1-y_flat)*out_flat).sum()
        FN = (y_flat*(1-out_flat)).sum()
        TN = ((1-y_flat)*(1-out_flat)).sum()

        dsc = 2*TP/(2*TP+FP+FN+1e-6)
        sensitivity = TP/(TP+FN+1e-6)
        specificity = TN/(TN+FP+1e-6)

        dscs.append(dsc.item())
        sens.append(sensitivity.item())
        specs.append(specificity.item())

    print(f"Mean DSC: {np.mean(dscs):.4f}")
    print(f"Mean Sensitivity: {np.mean(sens):.4f}")
    print(f"Mean Specificity: {np.mean(specs):.4f}")

evaluate_metrics(model, val_loader)

# ==============================================================
# Visualize Predictions
# ==============================================================
def show_predictions(model, loader, num=2):
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i >= num: break
            x,y = x.to(device), y.to(device)
            out = (model(x) > 0.5).float()

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(x[0,0].cpu(), cmap='gray')
            plt.title("Input Image")
            plt.subplot(1,3,2)
            plt.imshow(y[0,0].cpu(), cmap='gray')
            plt.title("Ground Truth")
            plt.subplot(1,3,3)
            plt.imshow(out[0,0].cpu(), cmap='gray')
            plt.title("Prediction")
            plt.show()

show_predictions(model, val_loader)

print("UNetX Pipeline Complete.")
