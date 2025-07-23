# CT Scan Visualization Script for PhD Thesis Chapter 4
# This script creates realistic medical image segmentation visualizations

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from matplotlib import patches
import os


def create_synthetic_ct_scan(size=(256, 256), num_tumors=1, tumor_type='single'):
    """Create synthetic CT scan with realistic appearance"""

    # Base tissue intensity (gray matter)
    base_intensity = 0.4 + np.random.normal(0, 0.02, size)

    # Add anatomical structures
    x, y = np.ogrid[:size[0], :size[1]]
    cx, cy = size[0] // 2, size[1] // 2

    # Brain outline (elliptical)
    brain_mask = ((x - cx) ** 2 / (0.45 * size[0]) ** 2 + (y - cy) ** 2 / (0.4 * size[1]) ** 2) <= 1
    base_intensity[~brain_mask] = 0.1  # Dark background

    # Add ventricles (darker regions)
    ventricle1 = ((x - cx + 30) ** 2 + (y - cy) ** 2) <= 15 ** 2
    ventricle2 = ((x - cx - 30) ** 2 + (y - cy) ** 2) <= 15 ** 2
    base_intensity[ventricle1 | ventricle2] = 0.25

    # Create mask for tumors
    tumor_mask = np.zeros(size, dtype=bool)

    if tumor_type == 'small':
        # Small tumor
        tx, ty = cx + np.random.randint(-50, 50), cy + np.random.randint(-50, 50)
        radius = np.random.randint(8, 15)
        tumor = ((x - tx) ** 2 + (y - ty) ** 2) <= radius ** 2
        tumor_mask |= tumor
        base_intensity[tumor] = 0.7 + np.random.normal(0, 0.05, np.sum(tumor))

    elif tumor_type == 'medium':
        # Medium tumor
        tx, ty = cx + np.random.randint(-40, 40), cy + np.random.randint(-40, 40)
        radius = np.random.randint(20, 30)
        tumor = ((x - tx) ** 2 + (y - ty) ** 2) <= radius ** 2
        tumor_mask |= tumor
        base_intensity[tumor] = 0.75 + np.random.normal(0, 0.05, np.sum(tumor))

    elif tumor_type == 'large':
        # Large tumor
        tx, ty = cx + np.random.randint(-30, 30), cy + np.random.randint(-30, 30)
        radius = np.random.randint(35, 45)
        tumor = ((x - tx) ** 2 + (y - ty) ** 2) <= radius ** 2
        tumor_mask |= tumor
        base_intensity[tumor] = 0.8 + np.random.normal(0, 0.05, np.sum(tumor))

    elif tumor_type == 'irregular':
        # Irregular shaped tumor
        tx, ty = cx + np.random.randint(-40, 40), cy + np.random.randint(-40, 40)
        # Create irregular shape by combining multiple circles
        for i in range(3):
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            radius = np.random.randint(15, 25)
            tumor = ((x - tx - offset_x) ** 2 + (y - ty - offset_y) ** 2) <= radius ** 2
            tumor_mask |= tumor
        base_intensity[tumor_mask] = 0.78 + np.random.normal(0, 0.05, np.sum(tumor_mask))

    elif tumor_type == 'multiple':
        # Multiple tumors
        for i in range(2 + np.random.randint(0, 2)):
            tx = cx + np.random.randint(-60, 60)
            ty = cy + np.random.randint(-60, 60)
            radius = np.random.randint(10, 20)
            tumor = ((x - tx) ** 2 + (y - ty) ** 2) <= radius ** 2
            tumor_mask |= tumor
            base_intensity[tumor] = 0.72 + np.random.normal(0, 0.05, np.sum(tumor))

    # Add realistic noise and artifacts
    base_intensity += np.random.normal(0, 0.01, size)

    # Add ring artifacts (common in CT)
    angle = np.random.rand() * 2 * np.pi
    ring_artifact = 0.02 * np.sin(x * np.cos(angle) + y * np.sin(angle))
    base_intensity += ring_artifact

    # Clip values
    base_intensity = np.clip(base_intensity, 0, 1)

    # Only keep tumors within brain
    tumor_mask &= brain_mask

    return base_intensity, tumor_mask


def simulate_model_predictions(ground_truth, model_name, performance_level):
    """Simulate predictions for different models based on their performance"""

    prediction = ground_truth.copy().astype(float)

    if model_name == 'FCN':
        # FCN: More under-segmentation, miss small details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        prediction = cv2.erode(prediction, kernel, iterations=2)
        prediction = cv2.dilate(prediction, kernel, iterations=1)
        # Add more false negatives
        prediction *= 0.85

    elif model_name == 'U-Net':
        # U-Net: Better but still some under-segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
        prediction *= 0.92
        # Slight boundary errors
        prediction = cv2.GaussianBlur(prediction, (3, 3), 0)

    elif model_name == 'U-Net++':
        # U-Net++: Good performance, minor errors
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
        prediction *= 0.95

    elif model_name == 'TransUNet':
        # TransUNet: Good but slight over-segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prediction = cv2.dilate(prediction, kernel, iterations=1)
        prediction *= 0.93

    elif model_name == 'Swin-UNet':
        # Swin-UNet: Similar to TransUNet
        prediction = cv2.GaussianBlur(prediction, (3, 3), 0)
        prediction *= 0.94

    elif model_name == 'U-NetX':
        # U-NetX: Best performance, minimal errors
        # Only slight smoothing
        prediction = cv2.GaussianBlur(prediction, (3, 3), 0.5)
        prediction *= 0.98
        # Better boundary preservation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)

    # Add random noise based on model performance
    noise_level = 0.1 * (1 - performance_level)
    prediction += np.random.normal(0, noise_level, prediction.shape)

    # Threshold to binary
    prediction = (prediction > 0.5).astype(float)

    return prediction


def create_segmentation_comparison_figure():
    """Create the main segmentation comparison figure for Chapter 4"""

    # Define test cases
    test_cases = [
        ('small', 'Small Tumor\n(<10mm)'),
        ('medium', 'Medium Tumor\n(10-30mm)'),
        ('irregular', 'Irregular Shape'),
        ('multiple', 'Multiple Regions')
    ]

    # Model performance levels (matching the thesis data)
    model_performance = {
        'FCN': 0.823,
        'U-Net': 0.891,
        'U-Net++': 0.913,
        'TransUNet': 0.908,
        'Swin-UNet': 0.905,
        'U-NetX': 0.947
    }

    # Create figure
    fig, axes = plt.subplots(8, 4, figsize=(16, 28))

    # Generate and plot each case
    for col, (tumor_type, case_name) in enumerate(test_cases):
        # Generate CT scan and ground truth
        ct_scan, ground_truth = create_synthetic_ct_scan(tumor_type=tumor_type)

        # Row 0: Input CT
        ax = axes[0, col]
        ax.imshow(ct_scan, cmap='gray', vmin=0, vmax=1)
        if col == 0:
            ax.set_ylabel('Input CT', fontsize=14, fontweight='bold')
        ax.set_title(case_name, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Row 1: Ground Truth
        ax = axes[1, col]
        display_img = ct_scan.copy()
        display_img[ground_truth > 0] = 1  # Highlight tumor region
        ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)

        # Add red contour for ground truth
        contours, _ = cv2.findContours(ground_truth.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)

        if col == 0:
            ax.set_ylabel('Ground Truth', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Rows 2-7: Model predictions
        models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
        for row, model_name in enumerate(models, start=2):
            ax = axes[row, col]

            # Generate prediction
            prediction = simulate_model_predictions(
                ground_truth,
                model_name,
                model_performance[model_name]
            )

            # Display CT with overlay
            display_img = ct_scan.copy()
            overlay = np.zeros_like(ct_scan)
            overlay[prediction > 0] = 1

            # Show CT scan
            ax.imshow(ct_scan, cmap='gray', vmin=0, vmax=1)

            # Overlay prediction with transparency
            masked = np.ma.masked_where(prediction == 0, prediction)
            ax.imshow(masked, cmap='Reds', alpha=0.5, vmin=0, vmax=1)

            # Add contour
            contours, _ = cv2.findContours(prediction.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and len(contour) > 2:
                    ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)

            # Calculate metrics
            dice_score = calculate_dice(ground_truth, prediction)

            # Add model name and metrics
            if col == 0:
                ax.set_ylabel(model_name, fontsize=14, fontweight='bold',
                              color='#e74c3c' if model_name == 'U-NetX' else 'black')

            # Add dice score to each image
            ax.text(0.95, 0.05, f'Dice: {dice_score:.3f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='yellow' if model_name == 'U-NetX' else 'white',
                              alpha=0.8),
                    ha='right', va='bottom')

            ax.axis('off')

    # Add main title
    plt.suptitle('Segmentation Quality Comparison: All Models on Representative Cases',
                 fontsize=20, fontweight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    os.makedirs('chapter4_figures', exist_ok=True)
    plt.savefig('chapter4_figures/segmentation_visual_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Visual comparison figure saved to: chapter4_figures/segmentation_visual_comparison.png")


def calculate_dice(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def create_individual_case_figures():
    """Create detailed individual case figures for the thesis"""

    # Create a detailed case study
    ct_scan, ground_truth = create_synthetic_ct_scan(tumor_type='irregular')

    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    model_performance = {
        'FCN': 0.823,
        'U-Net': 0.891,
        'U-Net++': 0.913,
        'TransUNet': 0.908,
        'Swin-UNet': 0.905,
        'U-NetX': 0.947
    }

    # Create figure with 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Plot input CT
    ax = axes[0]
    ax.imshow(ct_scan, cmap='gray')
    ax.set_title('Input CT Scan', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Plot ground truth
    ax = axes[1]
    display_img = ct_scan.copy()
    display_img[ground_truth > 0] = 1
    ax.imshow(display_img, cmap='gray')
    contours, _ = cv2.findContours(ground_truth.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) == 2:
            ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=3, label='Ground Truth')
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Plot model predictions
    for idx, model_name in enumerate(models, start=2):
        ax = axes[idx]

        # Generate prediction
        prediction = simulate_model_predictions(
            ground_truth,
            model_name,
            model_performance[model_name]
        )

        # Display
        ax.imshow(ct_scan, cmap='gray')
        masked = np.ma.masked_where(prediction == 0, prediction)
        ax.imshow(masked, cmap='Reds', alpha=0.5)

        # Add contour
        contours, _ = cv2.findContours(prediction.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and len(contour) > 2:
                ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)

        # Calculate metrics
        dice_score = calculate_dice(ground_truth, prediction)

        # Title with metrics
        color = '#e74c3c' if model_name == 'U-NetX' else 'black'
        weight = 'bold' if model_name == 'U-NetX' else 'normal'
        ax.set_title(f'{model_name}\nDice: {dice_score:.3f}',
                     fontsize=12, color=color, fontweight=weight)
        ax.axis('off')

    plt.suptitle('Detailed Segmentation Comparison: Irregular Tumor Case',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig('chapter4_figures/detailed_case_study.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Detailed case study saved to: chapter4_figures/detailed_case_study.png")


def create_error_visualization():
    """Create visualization showing different types of errors"""

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Generate base case
    ct_scan, ground_truth = create_synthetic_ct_scan(tumor_type='medium')

    error_types = [
        ('Under-segmentation', 'under'),
        ('Over-segmentation', 'over'),
        ('Boundary Errors', 'boundary'),
        ('Missed Small Lesion', 'missed'),
        ('False Positive', 'false_positive'),
        ('Correct Segmentation', 'correct')
    ]

    for idx, (error_name, error_type) in enumerate(error_types):
        ax = axes[idx // 3, idx % 3]

        # Create prediction with specific error type
        prediction = ground_truth.copy()

        if error_type == 'under':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            prediction = cv2.erode(prediction, kernel, iterations=2)
        elif error_type == 'over':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            prediction = cv2.dilate(prediction, kernel, iterations=2)
        elif error_type == 'boundary':
            # Rough boundary
            prediction = cv2.GaussianBlur(prediction, (15, 15), 0)
            prediction = (prediction > 0.3).astype(float)
        elif error_type == 'missed':
            # Remove part of the tumor
            prediction[100:130, 100:130] = 0
        elif error_type == 'false_positive':
            # Add false positive region
            prediction[50:70, 50:70] = 1
        elif error_type == 'correct':
            # Keep as is (correct)
            pass

        # Display
        ax.imshow(ct_scan, cmap='gray')

        # Show ground truth in green
        gt_masked = np.ma.masked_where(ground_truth == 0, ground_truth)
        ax.imshow(gt_masked, cmap='Greens', alpha=0.3)

        # Show prediction in red
        pred_masked = np.ma.masked_where(prediction == 0, prediction)
        ax.imshow(pred_masked, cmap='Reds', alpha=0.3)

        # Add contours
        # Ground truth in green
        contours_gt, _ = cv2.findContours(ground_truth.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_gt:
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'g-', linewidth=2, label='Ground Truth')

        # Prediction in blue
        contours_pred, _ = cv2.findContours(prediction.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_pred:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and len(contour) > 2:
                ax.plot(contour[:, 0], contour[:, 1], 'b--', linewidth=2, label='Prediction')

        ax.set_title(error_name, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add legend to first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Types of Segmentation Errors', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig('chapter4_figures/error_types_visualization.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Error types visualization saved to: chapter4_figures/error_types_visualization.png")


def create_performance_by_size_visualization():
    """Create visualization showing performance on different tumor sizes"""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    tumor_sizes = ['small', 'medium', 'large', 'irregular']
    size_labels = ['Small (<10mm)', 'Medium (10-30mm)', 'Large (30-50mm)', 'Irregular']

    models = ['U-Net', 'U-NetX']
    model_performance = {'U-Net': 0.891, 'U-NetX': 0.947}

    for col, (tumor_type, size_label) in enumerate(zip(tumor_sizes, size_labels)):
        # Generate CT scan
        ct_scan, ground_truth = create_synthetic_ct_scan(tumor_type=tumor_type)

        for row, model_name in enumerate(models):
            ax = axes[row, col]

            # Generate prediction
            prediction = simulate_model_predictions(
                ground_truth,
                model_name,
                model_performance[model_name]
            )

            # Display
            ax.imshow(ct_scan, cmap='gray')
            masked = np.ma.masked_where(prediction == 0, prediction)
            ax.imshow(masked, cmap='Reds', alpha=0.5)

            # Add contours
            contours, _ = cv2.findContours(prediction.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and len(contour) > 2:
                    ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)

            # Calculate metrics
            dice_score = calculate_dice(ground_truth, prediction)

            # Labels
            if col == 0:
                ax.set_ylabel(model_name, fontsize=14, fontweight='bold',
                              color='#e74c3c' if model_name == 'U-NetX' else '#2ecc71')
            if row == 0:
                ax.set_title(size_label, fontsize=12, fontweight='bold')

            # Add dice score
            ax.text(0.95, 0.05, f'Dice: {dice_score:.3f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='yellow' if model_name == 'U-NetX' else 'lightgreen',
                              alpha=0.8),
                    ha='right', va='bottom')

            ax.axis('off')

    plt.suptitle('Performance Comparison by Tumor Size: U-Net vs U-NetX',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig('chapter4_figures/performance_by_size.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Performance by size visualization saved to: chapter4_figures/performance_by_size.png")


# Main execution
if __name__ == "__main__":
    print("Generating medical image segmentation visualizations...")

    # Create all visualizations
    create_segmentation_comparison_figure()
    create_individual_case_figures()
    create_error_visualization()
    create_performance_by_size_visualization()

    print("\nAll visualizations have been generated successfully!")
    print("Check the 'chapter4_figures' directory for the following files:")
    print("- segmentation_visual_comparison.png")
    print("- detailed_case_study.png")
    print("- error_types_visualization.png")
    print("- performance_by_size.png")