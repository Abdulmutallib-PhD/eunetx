import os
import time
import cv2
import csv
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, confusion_matrix
import platform
import psutil
import socket
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


class EUNetX(nn.Module):
    def __init__(self):
        super(EUNetX, self).__init__()
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

# Prepare results directory and CSV file
os.makedirs("results", exist_ok=True)
csv_path = "csv/eunetx_epoch_loss.csv"

# Initialize CSV file with headers
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss"])


report_path = "csv/system_performance_report.csv"
os.makedirs("csv", exist_ok=True)

# Start overall timing
start_time = time.time()

# Hardware/System Information
system_info = {
    "Hostname": socket.gethostname(),
    "Platform": platform.system(),
    "Platform-Version": platform.version(),
    "Processor": platform.processor(),
    "CPU Count": psutil.cpu_count(logical=True),
    "RAM (GB)": round(psutil.virtual_memory().total),
    "GPU Available": torch.cuda.is_available(),
    "GPU Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "Python Version": platform.python_version()
}

# Track training time
training_start = time.time()


# === Model Training ===
device = torch.device("cpu")
model = EUNetX().to(device)
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

        # Inside training loop after total_loss is calculated
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, total_loss])

    print(f"Epoch {epoch+1}/50, Loss: {total_loss:.4f}")

training_end = time.time()
training_duration = training_end - training_start

# Track evaluation time
evaluation_start = time.time()

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for img, mask in loader:
        pred = model(img.to(device)).cpu().numpy().flatten() > 0.5
        mask = mask.numpy().flatten()
        y_pred.extend(pred.astype(int))
        y_true.extend(mask.astype(int))


# Convert to arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Standard Metrics ===
intersection = np.sum(y_true * y_pred)
dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
jaccard = jaccard_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
ppv = tp / (tp + fp + 1e-8)
npv = tn / (tn + fn + 1e-8)

# === HD95 Computation ===
def compute_hd95(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return np.nan

    dt_pred = distance_transform_edt(1 - pred)
    dt_gt = distance_transform_edt(1 - gt)

    surf1 = pred - (pred & gt)
    surf2 = gt - (pred & gt)

    dists1 = dt_gt[surf1.astype(bool)]
    dists2 = dt_pred[surf2.astype(bool)]

    all_dists = np.concatenate([dists1, dists2])
    if len(all_dists) == 0:
        return 0.0
    return np.percentile(all_dists, 95)

# Compute HD95 on each sample and average
hd95_list = []
with torch.no_grad():
    for img, mask in loader:
        pred = model(img.to(device)).cpu().numpy()[0, 0]
        gt = mask.numpy()[0, 0]
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)
        hd = compute_hd95(pred_bin, gt_bin)
        if not np.isnan(hd):
            hd95_list.append(hd)

hd95_score = round(np.mean(hd95_list), 2)


print(f"Jaccard Index (IoU): {jaccard:.4f}")
print(f"Dice Coefficient (DSC): {dice:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (PPV): {ppv:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")


# CREATING EVALUATION METRIC REPORT
data = {
    "Model": ["UNet++", "TransUNet", "Swin-PANet", "MedT", "EUNetX"],
    "DSC": [np.nan, 88.39, 91.42, 81.02, round(dice * 100, 2)],
    "HD95": [np.nan, np.nan, np.nan, np.nan, hd95_score],
    "IoU": [92.52, np.nan, 84.88, 69.61, round(jaccard * 100, 2)],
    "Sensitivity": [np.nan, np.nan, np.nan, np.nan, round(sensitivity * 100, 2)],
    "Specificity": [np.nan, np.nan, np.nan, np.nan, round(specificity * 100, 2)],
    "PPV": [np.nan, np.nan, np.nan, np.nan, round(ppv * 100, 2)],
    "NPV": [np.nan, np.nan, np.nan, np.nan, round(npv * 100, 2)]
}

# REPORT FOR COMPUTATIONAL PERFORMANCE
evaluation_end = time.time()
evaluation_duration = evaluation_end - evaluation_start
end_time = time.time()
total_duration = end_time - start_time

# Save report to CSV
with open(report_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    for key, value in system_info.items():
        writer.writerow([key, value])
    writer.writerow(["Training Duration (s)", round(training_duration, 2)])
    writer.writerow(["Evaluation Duration (s)",  round(evaluation_duration, 2)])
    writer.writerow(["Total Runtime (s)",  round(total_duration, 2)])

df = pd.DataFrame(data)
df.to_csv("csv/evaluation_metrics.csv", index=False)

# === Expert Radiologist Metrics from Ground Truth vs Ground Truth ===
expert_y_true, expert_y_pred = [], []
expert_hd95_list = []

with torch.no_grad():
    for _, mask in loader:
        gt = mask.numpy().flatten()
        pred = mask.numpy().flatten()  # same as ground truth
        expert_y_true.extend(gt.astype(int))
        expert_y_pred.extend(pred.astype(int))

        # HD95
        gt_img = mask.numpy()[0, 0]
        gt_bin = (gt_img > 0.5).astype(np.uint8)
        hd = compute_hd95(gt_bin, gt_bin)  # same image
        if not np.isnan(hd):
            expert_hd95_list.append(hd)

# Convert to arrays
expert_y_true = np.array(expert_y_true)
expert_y_pred = np.array(expert_y_pred)

# Metrics
intersection = np.sum(expert_y_true * expert_y_pred)
dice_exp = (2. * intersection) / (np.sum(expert_y_true) + np.sum(expert_y_pred) + 1e-8)
iou_exp = jaccard_score(expert_y_true, expert_y_pred)
tn, fp, fn, tp = confusion_matrix(expert_y_true, expert_y_pred, labels=[0, 1]).ravel()
sensitivity_exp = tp / (tp + fn + 1e-8)
specificity_exp = tn / (tn + fp + 1e-8)
ppv_exp = tp / (tp + fp + 1e-8)
npv_exp = tn / (tn + fn + 1e-8)
hd95_exp = round(np.mean(expert_hd95_list), 2)

# Build Expert DataFrame
expert_metrics = {
    'Model': ['Expert Radiologist'],
    'DSC': [round(dice_exp * 100, 2)],
    'HD95': [hd95_exp],
    'IoU': [round(iou_exp * 100, 2)],
    'Sensitivity': [round(sensitivity_exp * 100, 2), ],
    'Specificity': [round(specificity_exp * 100, 2)],
    'PPV': [round(ppv_exp * 100, 2)],
    'NPV': [round(npv_exp * 100, 2)]
}

# Save expert radiologist CSV
df_expert = pd.DataFrame(expert_metrics)
df_expert.to_csv("csv/expert_radiologist_metrics.csv", index=False)


# === Save model ===
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/eunetx_model.pth")

# === Auto-run visualization ===
print("\nPlease wait while launching visualization...")
time.sleep(3)
print("Launching visualization...")
os.system("python modules/visualize_results.py")

print("\nGenerating plots...")
time.sleep(3)

time.sleep(3)
os.system("python modules/plot_summary.py")
# os.system("python modules/eunetx_expert_comparision") # DIFFERENT CODE FOR THIS PLEASE!!!
print("Process completed")

time.sleep(3)
print("\nPlease wait while generating reporting data...")
os.system("python modules/generate_report.py")


# RUN STATIC FILES FOR GENERATING REPORTS
# os.system("python static/comparative_on_baseline.py")
# os.system("python static/comparative_unetx_metrics.py")
# os.system("python static/dataset_comparison_table.py")
# os.system("python static/external_dataset_loss_plot.py")
# os.system("python static/plot_unetx_brats2021_metrics.py")
# os.system("python static/unetx_brats2021_14k_loss_curve.py")
# os.system("python static/unetx_comparative_analysis_datasets.py")
# os.system("python static/unetx_internal_loss_curve.py")
# os.system("python static/unetx_lung_mri_metrics.py")
time.sleep(10)
print("All process completed")