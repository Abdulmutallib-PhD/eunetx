
import os
import numpy as np
import torch
import pandas as pd
from medpy.metric.binary import hd95  # Ensure you install with: pip install medpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_full_metrics(model, loader):
    dscs, hd95s, jaccards = [], [], []
    sens, specs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = (model(x) > 0.5).float()
            for i in range(x.size(0)):
                pred = out[i, 0].cpu().numpy()
                true = y[i, 0].cpu().numpy()

                inter = np.logical_and(pred, true).sum()
                union = np.logical_or(pred, true).sum()
                dsc = (2.0 * inter) / (pred.sum() + true.sum() + 1e-6)
                dsc = min(max(dsc, 0.0), 1.0)  # Clamp DSC to [0, 1]
                jaccard = inter / (union + 1e-6)
                jaccard = min(max(jaccard, 0.0), 1.0)

                try:
                    hd95_val = hd95(pred.astype(bool), true.astype(bool))
                    if np.isinf(hd95_val) or np.isnan(hd95_val):
                        hd95_val = 0.0
                except:
                    hd95_val = 0.0

                TP = np.sum((pred == 1) & (true == 1))
                FP = np.sum((pred == 1) & (true == 0))
                FN = np.sum((pred == 0) & (true == 1))
                TN = np.sum((pred == 0) & (true == 0))

                sensitivity = TP / (TP + FN + 1e-6)
                specificity = TN / (TN + FP + 1e-6)
                specificity = min(max(specificity, 0.0), 1.0)

                dscs.append(dsc)
                jaccards.append(jaccard)
                hd95s.append(hd95_val)
                sens.append(sensitivity)
                specs.append(specificity)