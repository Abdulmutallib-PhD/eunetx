# Advanced Medical Image Segmentation Visualization for U-NetX
# Based on your code with significant improvements and PhD thesis integration

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, \
    Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
import warnings

warnings.filterwarnings("ignore")

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedSegmentationVisualizer:
    """Advanced visualization class for medical image segmentation results"""

    def __init__(self, save_dir='chapter4_figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Define color schemes for different models
        self.model_colors = {
            'FCN': '#3498db',
            'U-Net': '#2ecc71',
            'U-Net++': '#f39c12',
            'TransUNet': '#9b59b6',
            'Swin-UNet': '#1abc9c',
            'U-NetX': '#e74c3c'
        }

    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Calculate Dice coefficient"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return (2.0 * intersection + smooth) / (union + smooth)

    def iou_score(self, y_true, y_pred, smooth=1e-6):
        """Calculate Intersection over Union"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def sensitivity_specificity(self, y_true, y_pred):
        """Calculate sensitivity and specificity"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        true_positives = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        true_negatives = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        false_positives = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        false_negatives = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

        sensitivity = true_positives / (true_positives + false_negatives + 1e-6)
        specificity = true_negatives / (true_negatives + false_positives + 1e-6)

        return sensitivity, specificity

    def hausdorff_distance(self, y_true, y_pred):
        """Calculate Hausdorff distance"""
        # Convert to binary
        y_true = (y_true > 0.5).astype(np.uint8)
        y_pred = (y_pred > 0.5).astype(np.uint8)

        # Calculate distance transforms
        dist_true = distance_transform_edt(1 - y_true)
        dist_pred = distance_transform_edt(1 - y_pred)

        # Find surface points
        surface_true = (dist_true == 1)
        surface_pred = (dist_pred == 1)

        if not np.any(surface_true) or not np.any(surface_pred):
            return 0.0

        # Calculate distances
        dist_true_to_pred = dist_pred[surface_true]
        dist_pred_to_true = dist_true[surface_pred]

        # Hausdorff distance
        hd = max(np.max(dist_true_to_pred), np.max(dist_pred_to_true))
        return hd


class UNetXArchitecture:
    """Enhanced U-NetX architecture with all advanced components"""

    @staticmethod
    def nested_dense_block(x, filters, name):
        """Nested dense block component"""
        conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{name}_conv1')(x)
        conv1 = BatchNormalization()(conv1)

        concat1 = concatenate([x, conv1], name=f'{name}_concat1')
        conv2 = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{name}_conv2')(concat1)
        conv2 = BatchNormalization()(conv2)

        concat2 = concatenate([x, conv1, conv2], name=f'{name}_concat2')
        conv3 = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{name}_conv3')(concat2)
        conv3 = BatchNormalization()(conv3)

        # Final concatenation
        output = concatenate([x, conv1, conv2, conv3], name=f'{name}_output')
        output = Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{name}_bottleneck')(output)

        return output

    @staticmethod
    def build_unetx(input_size=(256, 256, 3), num_classes=1, dropout_rate=0.1):
        """Build U-NetX architecture"""
        inputs = Input(input_size)

        # Encoder path with nested dense blocks
        # Level 1
        c1 = UNetXArchitecture.nested_dense_block(inputs, 64, 'encoder1')
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout_rate)(p1)

        # Level 2
        c2 = UNetXArchitecture.nested_dense_block(p1, 128, 'encoder2')
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout_rate)(p2)

        # Level 3
        c3 = UNetXArchitecture.nested_dense_block(p2, 256, 'encoder3')
        p3 = Max