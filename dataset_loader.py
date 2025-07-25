import os
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset

class DicomDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.dcm')])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.dcm')])

    def __len__(self):
        return len(self.images)

    def load_dicom(self, path):
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

    def __getitem__(self, idx):
        img = np.expand_dims(self.load_dicom(self.images[idx]), 0)
        mask = np.expand_dims(self.load_dicom(self.masks[idx]), 0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
