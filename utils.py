import torch

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth))
