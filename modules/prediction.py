import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, confusion_matrix
import subprocess

class UNetX(nn.Module):
    def __init__(self):
        super(UNetX, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )

        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        up2 = self.upconv2(b)
        d2 = self.decoder2(up2)

        up1 = self.upconv1(d2)
        d1 = self.decoder1(up1)

        out = self.final(d1)
        return self.sigmoid(out)

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir) if fname.endswith('.png')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, fname)
            for fname in os.listdir(mask_dir) if fname.endswith('.png')
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (256, 256)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img), torch.tensor(mask)

# === Directory Paths ===
image_dir = "dataset/images"
mask_dir = "dataset/masks"

# === Load Dataset ===
dataset = Dataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Model Training ===
device = torch.device("cpu")
model = UNetX().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Training for more epochs
model.train()
for epoch in range(50):
    total_loss = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/50, Loss: {total_loss:.4f}")

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for img, mask in loader:
        pred = model(img.to(device)).cpu().numpy().flatten() > 0.5
        mask = mask.numpy().flatten()
        y_pred.extend(pred.astype(int))
        y_true.extend(mask.astype(int))

# === Metrics ===
y_true = np.array(y_true)
y_pred = np.array(y_pred)

intersection = np.sum(y_true * y_pred)
dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
jaccard = jaccard_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
ppv = tp / (tp + fp + 1e-8)
npv = tn / (tn + fn + 1e-8)

print(f"Jaccard Index (IoU): {jaccard:.4f}")
print(f"Dice Coefficient (DSC): {dice:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (PPV): {ppv:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")

# === Save model ===
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/unetx_model.pth")
print("Model saved to results/unetx_model.pth")

# === Auto-run visualization ===
print("\nWaiting 3 seconds before launching visualization...")
time.sleep(3)
print("Launching visualization...")
os.system("python modules/visualize_results.py")

print("\nWaiting a seconds while generating plots...")
time.sleep(3)
# Execute the external plotting script
os.system("python modules/generate_evaluation_plots.py")
print("All process completed")