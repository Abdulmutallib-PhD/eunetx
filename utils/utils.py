import torch
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

def log_and_plot_losses(train_losses_initial, val_losses_initial, train_losses_finetune, val_losses_finetune, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    max_len = max(len(train_losses_initial), len(train_losses_finetune))
    data = {
        'Epoch': list(range(1, max_len + 1)),
        'Train_Loss_Initial': train_losses_initial + [''] * (max_len - len(train_losses_initial)),
        'Val_Loss_Initial': val_losses_initial + [''] * (max_len - len(val_losses_initial)),
        'Train_Loss_Finetune': train_losses_finetune + [''] * (max_len - len(train_losses_finetune)),
        'Val_Loss_Finetune': val_losses_finetune + [''] * (max_len - len(val_losses_finetune)),
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(output_dir, "unetx_loss_curve.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_initial, 'r--', label='Initial Train Loss')
    plt.plot(val_losses_initial, 'b--', label='Initial Val Loss')
    plt.plot(train_losses_finetune, 'r-', label='Fine-tune Train Loss')
    plt.plot(val_losses_finetune, 'b-', label='Fine-tune Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Dice Loss")
    plt.title("UNetX Training and Fine-Tuning Loss Curve")
    plt.legend()
    plot_path = os.path.join(output_dir, "unetx_loss_curve.png")
    plt.savefig(plot_path)
    plt.close()

    return csv_path, plot_path

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth))


