import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

# ---------------------- UNetX Model ----------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class LightweightFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels//2, 1, bias=False),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels//2, 1, bias=False),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)

class UNetX(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.lff_modules = nn.ModuleList()
        self.pool = nn.MaxPool3d(2,2)

        for i, f in enumerate(features):
            in_feats = in_channels if i==0 else features[i-1]
            self.encoder_blocks.append(ConvBlock(in_feats, f))
            self.lff_modules.append(LightweightFeatureFusion(f,f))

        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        self.upsamples, self.decoder_convs, self.deep_supervision_outputs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        reversed_features = features[::-1]
        sum_lff = sum(features)

        for i in range(len(reversed_features)-1):
            up_in = features[-1]*2 if i==0 else reversed_features[i-1]
            up_out = reversed_features[i]
            self.upsamples.append(nn.ConvTranspose3d(up_in, up_out, 2,2))
            self.decoder_convs.append(ConvBlock(up_out+sum_lff, reversed_features[i]))
            self.deep_supervision_outputs.append(nn.Conv3d(reversed_features[i], out_channels, 1))

    def forward(self, x):
        enc_outs, lff_outs = [], []

        for i, blk in enumerate(self.encoder_blocks):
            x = blk(x)
            enc_outs.append(x)
            lff_outs.append(self.lff_modules[i](x))
            if i < len(self.encoder_blocks)-1:
                x = self.pool(x)

        x = self.bottleneck(x)
        outputs = []

        for i in range(len(self.decoder_convs)):
            x = self.upsamples[i](x)
            to_fuse = [x] + [F.interpolate(lff, size=x.shape[2:], mode='trilinear', align_corners=False) if lff.shape[2:]!=x.shape[2:] else lff for lff in lff_outs]
            x = self.decoder_convs[i](torch.cat(to_fuse, dim=1))
            outputs.append(torch.sigmoid(self.deep_supervision_outputs[i](x)))

        return outputs

# ---------------------- Dataset & Preprocessing ----------------------

class CTBrainDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not os.path.isfile(self.image_paths[idx]):
            raise FileNotFoundError(f"Image file not found: {self.image_paths[idx]}")

        ds = pydicom.dcmread(self.image_paths[idx])
        img_array = ds.pixel_array.astype(np.float32)

        img_array = np.clip(img_array, -100, 400)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        if self.mask_paths:
            if not os.path.isfile(self.mask_paths[idx]):
                raise FileNotFoundError(f"Mask file not found: {self.mask_paths[idx]}")
            ds_mask = pydicom.dcmread(self.mask_paths[idx])
            mask_array = ds_mask.pixel_array.astype(np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        else:
            mask_tensor = torch.zeros_like(img_tensor)

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

# ---------------------- Training & Validation ----------------------

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2.*intersection + smooth) / (union + smooth)

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_total = 0
    for imgs, masks in tqdm(loader, desc='Train'):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        out = model(imgs)[-1]
        loss = criterion(out, masks)
        loss.backward()
        opt.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    loss_total, dices = 0, []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Val'):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)[-1]
            loss_total += criterion(out, masks).item()
            pred = (out>0.5).float()
            dices.append(1 - dice_loss(pred, masks).item())
    return loss_total/len(loader), np.mean(dices)

# ---------------------- Semi-Supervised ----------------------

def pseudo_label(model, unlabelled_loader, device='cuda', threshold=0.8):
    model.eval()
    pseudo_imgs, pseudo_masks = [], []
    with torch.no_grad():
        for imgs,_ in tqdm(unlabelled_loader, desc='Pseudo-label'):
            imgs = imgs.to(device)
            out = model(imgs)[-1]
            preds = (out>threshold).float()
            pseudo_imgs.append(imgs.cpu())
            pseudo_masks.append(preds.cpu())
    return torch.cat(pseudo_imgs), torch.cat(pseudo_masks)

# ---------------------- Main ----------------------

if __name__ == '__main__':
    annotated_images = ["/path/to/real_image1.dcm", "/path/to/real_image2.dcm"]
    annotated_masks  = ["/path/to/real_mask1.dcm", "/path/to/real_mask2.dcm"]
    unannotated_images = ["/path/to/real_unlabeled1.dcm", "/path/to/real_unlabeled2.dcm"]

    for path in annotated_images + annotated_masks + unannotated_images:
        if not os.path.isfile(path):
            print(f"Warning: File does not exist — {path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = CTBrainDataset(annotated_images, annotated_masks)
    val_dataset = CTBrainDataset(annotated_images, annotated_masks)
    unlabelled_dataset = CTBrainDataset(unannotated_images, None)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=1)

    model = UNetX().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = lambda pred, tgt: dice_loss(pred, tgt) + nn.BCELoss()(pred, tgt)

    if len(unlabelled_dataset) > 0:
        pseudo_imgs, pseudo_masks = pseudo_label(model, unlabelled_loader, device)
        pseudo_dataset = TensorDataset(pseudo_imgs, pseudo_masks)
        combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
    else:
        combined_loader = train_loader

    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        train_loss = train_one_epoch(model, combined_loader, opt, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    torch.save(model.state_dict(), "unetx_final.pth")
