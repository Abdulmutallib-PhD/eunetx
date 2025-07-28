import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32)
        )
        self.pool1 = torch.nn.MaxPool2d(2)
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(64)
        )
        self.pool2 = torch.nn.MaxPool2d(2)

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(128)
        )

        self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(64)
        )

        self.upconv1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32)
        )

        self.final = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

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

# Dataset loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (256, 256)).astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

# Load paths and model
image_dir = 'dataset/images'
mask_dir = 'dataset/masks'
dataset = Dataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = UNet()
model.load_state_dict(torch.load("results/unetx_model.pth", map_location='cpu'))
model.eval()

# Run one prediction
images, masks = next(iter(loader))
with torch.no_grad():
    preds = model(images)
pred_mask = preds.squeeze().numpy() > 0.5
original = images.squeeze().numpy()
truth = masks.squeeze().numpy()

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original, cmap='gray')
axs[0].set_title("Original Image")
axs[1].imshow(truth, cmap='gray')
axs[1].set_title("Ground Truth Mask")
axs[2].imshow(pred_mask, cmap='gray')
axs[2].set_title("UNetX Prediction")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
