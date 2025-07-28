import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image

class DicomAndPNGDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.dcm', '.png'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                                  if f.lower().endswith(('.dcm', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image (DICOM or PNG)
        if image_path.lower().endswith(".dcm"):
            image = self.load_dicom(image_path)
        else:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            image = np.array(image).astype(np.float32)

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Load mask (DICOM or PNG)
        if mask_path.lower().endswith(".dcm"):
            mask = self.load_dicom(mask_path)
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
            mask = np.array(mask).astype(np.float32)
            mask = (mask > 127).astype(np.float32)

        # Convert to torch tensor and add channel dimension [C, H, W]
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

    @staticmethod
    def load_dicom(path):
        ds = pydicom.dcmread(path)
        if 'PixelData' not in ds:
            raise ValueError("No pixel data found in DICOM file.")

        try:
            img = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to decode DICOM file: {path}\n{str(e)}")

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img