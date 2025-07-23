import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

# ---------------------- Dataset ----------------------

class CTBrainDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ds = pydicom.dcmread(self.image_paths[idx])
        img_array = ds.pixel_array.astype(np.float32)

        img_array = np.clip(img_array, -100, 400)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [C=1, H, W]
        return img_tensor, self.image_paths[idx]  # keep path for saving output

# ---------------------- Inference ----------------------

def infer_and_save(model, loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc='Inference'):
            imgs = imgs.to(device).unsqueeze(0)  # [B=1, C, H, W]
            pred = model(imgs)[-1]  # final prediction
            pred = (pred > 0.5).float().cpu().squeeze().numpy()

            base_name = os.path.basename(paths[0]).replace('.dcm', '_mask.npy')
            np.save(os.path.join(output_dir, base_name), pred)

            # Optionally save overlay
            img_orig = imgs.cpu().squeeze().numpy()
            plt.imshow(img_orig[0], cmap='gray')
            plt.imshow(pred[0], alpha=0.5, cmap='Reds')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, base_name.replace('.npy', '.png')))
            plt.close()

# ---------------------- Main ----------------------

if __name__ == '__main__':
    unannotated_folder = 'dataset/nonannotated/image'
    output_dir = 'predictions'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetX().to(device)

    if os.path.exists('unetx_final.pth'):
        model.load_state_dict(torch.load('unetx_final.pth', map_location=device))
    else:
        print("⚠️ Model weights not found! Please train first or provide weights.")
        exit()

    image_paths = [os.path.join(unannotated_folder, f) for f in os.listdir(unannotated_folder) if f.endswith('.dcm')]
    if not image_paths:
        print(f"No DICOM files found in {unannotated_folder}")
        exit()

    dataset = CTBrainDataset(image_paths)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    infer_and_save(model, loader, device, output_dir)
    print(f"Inference complete. Results saved in {output_dir}")
