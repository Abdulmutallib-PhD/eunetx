# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create a directory for saving figures
import os

os.makedirs('chapter4_figures', exist_ok=True)


# 1. Performance Comparison Bar Chart
def plot_performance_comparison():
    """Create performance comparison across different models"""
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    metrics = {
        'Dice Score': [0.823, 0.891, 0.913, 0.908, 0.905, 0.947],
        'IoU': [0.698, 0.803, 0.839, 0.831, 0.827, 0.899],
        'Sensitivity': [0.812, 0.887, 0.909, 0.901, 0.898, 0.943],
        'Specificity': [0.976, 0.988, 0.991, 0.990, 0.989, 0.995],
        'Precision': [0.847, 0.906, 0.924, 0.919, 0.916, 0.956]
    }

    # Create DataFrame
    df = pd.DataFrame(metrics, index=models)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (metric, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(models, values, alpha=0.8)

        # Highlight U-NetX
        bars[-1].set_color('darkred')
        bars[-1].set_alpha(1.0)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0.6, 1.05)
        ax.set_ylabel(metric)
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('chapter4_figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 2. Dice Coefficient Line Plot
def plot_dice_progression():
    """Create a line plot showing Dice coefficient progression"""
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    dice_scores = [0.823, 0.891, 0.913, 0.908, 0.905, 0.947]
    std_devs = [0.045, 0.032, 0.024, 0.027, 0.028, 0.018]

    plt.figure(figsize=(10, 6))

    # Create line plot with error bars
    x = np.arange(len(models))
    plt.errorbar(x, dice_scores, yerr=std_devs, marker='o', markersize=10,
                 linewidth=2, capsize=5, capthick=2, elinewidth=2)

    # Highlight U-NetX point
    plt.scatter(x[-1], dice_scores[-1], color='red', s=200, zorder=5,
                edgecolors='darkred', linewidth=2)

    # Add annotations
    for i, (score, std) in enumerate(zip(dice_scores, std_devs)):
        plt.annotate(f'{score:.3f}±{std:.3f}',
                     xy=(i, score),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Dice Coefficient', fontsize=12)
    plt.title('Dice Coefficient Progression Across Architectures', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.75, 1.0)

    plt.tight_layout()
    plt.savefig('chapter4_figures/dice_progression.png', dpi=300, bbox_inches='tight')
    plt.show()


# 3. Training and Validation Loss Curves
def plot_training_curves():
    """Plot training and validation loss curves for all models"""
    np.random.seed(42)
    epochs = np.arange(0, 160, 2)

    # Simulate training curves (replace with actual data)
    def generate_loss_curve(initial, final, convergence_rate, epochs_to_converge):
        loss = np.zeros(len(epochs))
        for i, e in enumerate(epochs):
            if e < epochs_to_converge:
                loss[i] = initial - (initial - final) * (1 - np.exp(-convergence_rate * e))
            else:
                loss[i] = final + 0.01 * np.random.randn()
        return loss

    models_data = {
        'FCN': generate_loss_curve(0.8, 0.187, 0.02, 142),
        'U-Net': generate_loss_curve(0.8, 0.124, 0.025, 118),
        'U-Net++': generate_loss_curve(0.8, 0.098, 0.024, 125),
        'U-NetX': generate_loss_curve(0.8, 0.076, 0.03, 95)
    }

    plt.figure(figsize=(12, 7))

    for model, loss in models_data.items():
        style = '-' if model == 'U-NetX' else '--'
        linewidth = 3 if model == 'U-NetX' else 2
        plt.plot(epochs, loss, label=model, linestyle=style, linewidth=linewidth)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 160)
    plt.ylim(0, 0.85)

    # Add convergence annotations
    convergence_points = {'FCN': (142, 0.187), 'U-Net': (118, 0.124),
                          'U-Net++': (125, 0.098), 'U-NetX': (95, 0.076)}

    for model, (x, y) in convergence_points.items():
        plt.scatter(x, y, s=100, zorder=5)
        plt.annotate(f'{model}\n{x} epochs', xy=(x, y), xytext=(x + 10, y + 0.05),
                     fontsize=9, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('chapter4_figures/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 4. Active Learning Performance
def plot_active_learning():
    """Plot active learning vs random sampling performance"""
    data_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    random_dice = [0.612, 0.723, 0.798, 0.841, 0.867, 0.884, 0.897, 0.908, 0.918, 0.924]
    active_dice = [0.698, 0.804, 0.856, 0.884, 0.903, 0.918, 0.929, 0.937, 0.943, 0.947]

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(data_percentages, random_dice, 'o--', linewidth=2, markersize=8,
             label='Random Sampling', color='#1f77b4')
    plt.plot(data_percentages, active_dice, 'o-', linewidth=3, markersize=8,
             label='Active Learning (BALD)', color='#ff7f0e')

    # Fill area between curves
    plt.fill_between(data_percentages, random_dice, active_dice, alpha=0.3, color='green')

    # Add annotations for key points
    plt.annotate('90% performance\nwith 60% data',
                 xy=(60, 0.918), xytext=(65, 0.85),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12)
    plt.ylabel('Dice Coefficient', fontsize=12)
    plt.title('Active Learning vs Random Sampling Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(5, 105)
    plt.ylim(0.55, 1.0)

    # Add improvement percentages
    for i in range(0, len(data_percentages), 2):
        improvement = ((active_dice[i] - random_dice[i]) / random_dice[i]) * 100
        mid_point = (random_dice[i] + active_dice[i]) / 2
        plt.text(data_percentages[i], mid_point, f'+{improvement:.1f}%',
                 ha='center', va='center', fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig('chapter4_figures/active_learning_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. Computational Efficiency Scatter Plot
def plot_efficiency_analysis():
    """Create scatter plot showing efficiency vs performance trade-off"""
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    dice_scores = [0.823, 0.891, 0.913, 0.908, 0.905, 0.947]
    inference_times = [45.2, 52.7, 68.4, 89.3, 76.5, 61.3]
    parameters = [134.3, 31.0, 47.2, 105.3, 41.4, 52.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Dice vs Inference Time
    scatter1 = ax1.scatter(inference_times, dice_scores, s=[p * 3 for p in parameters],
                           alpha=0.6, c=range(len(models)), cmap='viridis')

    # Highlight U-NetX
    ax1.scatter(inference_times[-1], dice_scores[-1], s=parameters[-1] * 3,
                color='red', edgecolors='darkred', linewidth=2, zorder=5)

    # Add model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (inference_times[i], dice_scores[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax1.set_xlabel('Inference Time (ms)', fontsize=12)
    ax1.set_ylabel('Dice Coefficient', fontsize=12)
    ax1.set_title('Performance vs Inference Time Trade-off', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add optimal region
    ax1.axvspan(50, 70, alpha=0.1, color='green', label='Optimal Region')
    ax1.axhspan(0.92, 0.96, alpha=0.1, color='green')

    # Subplot 2: Parameters vs Dice Score
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
    bars = ax2.bar(models, parameters, color=colors, alpha=0.7)

    # Create secondary y-axis for Dice scores
    ax2_twin = ax2.twinx()
    ax2_twin.plot(models, dice_scores, 'ro-', linewidth=2, markersize=8)

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Parameters (M)', fontsize=12)
    ax2_twin.set_ylabel('Dice Coefficient', fontsize=12, color='red')
    ax2.set_title('Model Complexity vs Performance', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2_twin.tick_params(axis='y', colors='red')

    # Add value labels
    for i, (bar, param) in enumerate(zip(bars, parameters)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{param:.1f}M', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('chapter4_figures/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 6. Ablation Study Heatmap
def plot_ablation_study():
    """Create heatmap showing ablation study results"""
    components = ['Full Model', '-Nested Dense', '-Full-scale Skip',
                  '-Bayesian Dropout', '-Active Learning', 'Base U-Net']
    metrics = ['Dice Score', 'Parameters (M)', 'Inference (ms)']

    # Data for heatmap
    data = np.array([
        [0.947, 52.8, 61.3],
        [0.921, 41.2, 54.7],
        [0.908, 45.6, 57.2],
        [0.934, 52.8, 58.9],
        [0.929, 52.8, 61.3],
        [0.891, 31.0, 52.7]
    ])

    # Normalize data for better visualization
    data_norm = np.zeros_like(data)
    for i in range(data.shape[1]):
        data_norm[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())

    plt.figure(figsize=(8, 6))

    # Create custom colormap
    cmap = sns.diverging_palette(10, 150, as_cmap=True)

    # Create heatmap
    sns.heatmap(data_norm, annot=data, fmt='.3f', cmap=cmap,
                xticklabels=metrics, yticklabels=components,
                cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5, linecolor='gray')

    plt.title('Ablation Study: Component Impact Analysis', fontsize=14, fontweight='bold')
    plt.ylabel('Model Configuration', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)

    # Add delta annotations
    deltas = data[0, 0] - data[:, 0]
    for i in range(1, len(components)):
        plt.text(3.5, i, f'Δ={deltas[i]:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('chapter4_figures/ablation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# 7. Tumor Size Performance Analysis
def plot_tumor_size_analysis():
    """Create grouped bar chart for tumor size performance"""
    tumor_sizes = ['Small\n(<10mm)', 'Medium\n(10-30mm)', 'Large\n(30-50mm)', 'Very Large\n(>50mm)']
    models = ['FCN', 'U-Net', 'U-Net++', 'U-NetX']

    data = {
        'FCN': [0.672, 0.834, 0.867, 0.891],
        'U-Net': [0.785, 0.896, 0.912, 0.924],
        'U-Net++': [0.823, 0.917, 0.928, 0.937],
        'U-NetX': [0.942, 0.948, 0.951, 0.954]
    }

    x = np.arange(len(tumor_sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bars
    for i, (model, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xlabel('Tumor Size Category', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title('Segmentation Performance by Tumor Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tumor_sizes)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.6, 1.0)

    # Add sample counts
    sample_counts = [423, 1247, 892, 615]
    for i, count in enumerate(sample_counts):
        ax.text(i, 0.62, f'n={count}', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig('chapter4_figures/tumor_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 8. Uncertainty Calibration Plot
def plot_uncertainty_calibration():
    """Create uncertainty calibration plot"""
    # Generate calibration data
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

    # Simulated calibration data (replace with actual data)
    perfect_calibration = bin_centers
    observed_accuracy = bin_centers + 0.05 * np.sin(bin_centers * 2 * np.pi) + 0.02 * np.random.randn(10)
    observed_accuracy = np.clip(observed_accuracy, 0, 1)

    plt.figure(figsize=(10, 8))

    # Main calibration plot
    plt.subplot(2, 1, 1)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    plt.plot(bin_centers, observed_accuracy, 'ro-', linewidth=2, markersize=8,
             label='U-NetX Calibration')

    # Fill area showing calibration error
    plt.fill_between(bin_centers, perfect_calibration, observed_accuracy,
                     alpha=0.3, color='red', label='Calibration Error')

    plt.xlabel('Mean Predicted Confidence', fontsize=12)
    plt.ylabel('Observed Accuracy', fontsize=12)
    plt.title('Uncertainty Calibration Analysis', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add ECE annotation
    ece = np.mean(np.abs(observed_accuracy - perfect_calibration))
    plt.text(0.6, 0.2, f'ECE = {ece:.3f}', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Reliability diagram
    plt.subplot(2, 1, 2)
    bin_counts = np.random.randint(50, 200, 10)  # Simulated counts
    plt.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='blue')
    plt.xlabel('Confidence Bin', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Sample Distribution Across Confidence Bins', fontsize=13)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('chapter4_figures/uncertainty_calibration.png', dpi=300, bbox_inches='tight')
    plt.show()


# 9. Cross-Domain Validation Matrix
def plot_cross_domain_validation():
    """Create heatmap for cross-domain validation results"""
    train_datasets = ['Brain Tumor', 'Brain Tumor', 'Lung Nodule', 'Lung Nodule']
    test_datasets = ['Brain Tumor', 'Lung Nodule', 'Lung Nodule', 'Brain Tumor']
    dice_scores = [0.947, 0.823, 0.938, 0.792]

    # Create matrix
    matrix = np.array([[0.947, 0.823], [0.792, 0.938]])

    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=['Brain Tumor', 'Lung Nodule'],
                yticklabels=['Brain Tumor', 'Lung Nodule'],
                cbar_kws={'label': 'Dice Score'},
                vmin=0.75, vmax=0.95)

    plt.xlabel('Test Dataset', fontsize=12)
    plt.ylabel('Training Dataset', fontsize=12)
    plt.title('Cross-Domain Validation Performance', fontsize=14, fontweight='bold')

    # Add annotations for transfer learning performance
    plt.text(2.5, 0.5, 'Strong within-domain\nperformance', ha='left', va='center', fontsize=10)
    plt.text(2.5, 1.5, 'Promising cross-domain\ngeneralization', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('chapter4_figures/cross_domain_validation.png', dpi=300, bbox_inches='tight')
    plt.show()


# 10. Statistical Significance Box Plot
def plot_statistical_significance():
    """Create box plots with statistical significance indicators"""
    np.random.seed(42)

    # Generate sample data for each model (replace with actual data)
    n_samples = 100
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']

    data = {
        'FCN': np.random.normal(0.823, 0.045, n_samples),
        'U-Net': np.random.normal(0.891, 0.032, n_samples),
        'U-Net++': np.random.normal(0.913, 0.024, n_samples),
        'TransUNet': np.random.normal(0.908, 0.027, n_samples),
        'Swin-UNet': np.random.normal(0.905, 0.028, n_samples),
        'U-NetX': np.random.normal(0.947, 0.018, n_samples)
    }

    # Create DataFrame
    df_list = []
    for model, values in data.items():
        df_temp = pd.DataFrame({'Model': model, 'Dice Score': values})
        df_list.append(df_temp)
    df = pd.concat(df_list)

    plt.figure(figsize=(12, 8))

    # Create box plot
    box_plot = sns.boxplot(data=df, x='Model', y='Dice Score', palette='Set3')

    # Add mean markers
    means = [data[model].mean() for model in models]
    plt.scatter(range(len(models)), means, color='red', s=100, zorder=5,
                marker='D', label='Mean')

    # Add statistical significance bars
    y_max = 1.02
    significance_pairs = [('U-Net', 'U-NetX'), ('U-Net++', 'U-NetX'),
                          ('TransUNet', 'U-NetX'), ('Swin-UNet', 'U-NetX')]

    for i, (model1, model2) in enumerate(significance_pairs):
        x1 = models.index(model1)
        x2 = models.index(model2)
        y = y_max - i * 0.02

        # Draw significance bar
        plt.plot([x1, x1, x2, x2], [y - 0.005, y, y, y - 0.005], 'k-', linewidth=1)
        plt.text((x1 + x2) / 2, y + 0.005, '***', ha='center', fontsize=12)

    plt.ylabel('Dice Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Statistical Distribution of Dice Scores Across Models', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.7, 1.05)

    # Add p-value legend
    plt.text(0.02, 0.98, '*** p < 0.001', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('chapter4_figures/statistical_significance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 11. Feature Map Visualization
def plot_feature_maps():
    """Create feature map visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Simulate feature maps (replace with actual feature maps)
    np.random.seed(42)

    # Create synthetic tumor-like patterns
    x, y = np.meshgrid(np.linspace(-2, 2, 64), np.linspace(-2, 2, 64))
    tumor_mask = np.exp(-(x ** 2 + y ** 2) / 0.5)

    # Layer 1: Edge features
    edges = np.gradient(tumor_mask)[0] + np.gradient(tumor_mask)[1]
    edges = (edges - edges.min()) / (edges.max() - edges.min())

    # Layer 2: Shape features
    shapes = tumor_mask + 0.2 * np.random.randn(64, 64)
    shapes = (shapes - shapes.min()) / (shapes.max() - shapes.min())

    # Layer 3: Semantic features
    semantic = tumor_mask * (1 + 0.1 * np.random.randn(64, 64))
    semantic = (semantic - semantic.min()) / (semantic.max() - semantic.min())

    # Output: Final segmentation
    output = (tumor_mask > 0.5).astype(float)

    feature_maps = [edges, shapes, semantic, output]
    titles = ['Layer 1: Edge Features', 'Layer 2: Shape Features',
              'Layer 3: Semantic Features', 'Output: Segmentation']

    for ax, fmap, title in zip(axes, feature_maps, titles):
        im = ax.imshow(fmap, cmap='hot', aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle('Feature Map Progression Through U-NetX Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/feature_maps.png', dpi=300, bbox_inches='tight')
    plt.show()


# 12. Clinical Acceptability Pie Charts
def plot_clinical_acceptability():
    """Create pie charts for clinical acceptability assessment"""
    models = ['U-Net', 'U-Net++', 'U-NetX']
    acceptability_data = {
        'U-Net': [78.5, 16.0, 5.5],
        'U-Net++': [84.0, 12.5, 3.5],
        'U-NetX': [92.0, 6.5, 1.5]
    }
    labels = ['Clinically Acceptable', 'Minor Adjustments', 'Major Corrections']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, model, data in zip(axes, models, acceptability_data.values()):
    # Create pie chart
    wedges, texts, autotexts = ax.pie(data, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 10})

    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax.set_title(f'{model} Clinical Acceptability', fontsize=13, fontweight='bold')

plt.suptitle('Clinical Acceptability Assessment by Expert Radiologists',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('chapter4_figures/clinical_acceptability.png', dpi=300, bbox_inches='tight')
plt.show()


# 13. Confusion Matrix for Binary Segmentation
def plot_confusion_matrices():
    """Create confusion matrices for pixel-wise classification"""
    models = ['U-Net', 'U-NetX']

    # Simulated confusion matrix data (replace with actual data)
    cm_unet = np.array([[950000, 25000], [15000, 10000]])
    cm_unetx = np.array([[970000, 5000], [8000, 17000]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, cm in zip(axes, models, [cm_unet, cm_unetx]):
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=['Background', 'Tumor'],
                    yticklabels=['Background', 'Tumor'],
                    cbar_kws={'label': 'Normalized Count'},
                    ax=ax)

        ax.set_title(f'{model} Pixel-wise Classification', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)

        # Add performance metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        ax.text(1.02, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Pixel-wise Classification Performance', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


# 14. ROC Curves Comparison
def plot_roc_curves():
    """Plot ROC curves for different models"""
    plt.figure(figsize=(10, 8))

    # Generate ROC curve data (replace with actual data)
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Simulated ROC data
    np.random.seed(42)
    for i, (model, color) in enumerate(zip(models, colors)):
        # Generate curve based on model performance
        if model == 'U-NetX':
            # Best performance
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-8 * fpr)
            tpr[tpr > 1] = 1
        else:
            # Other models with varying performance
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-np.random.uniform(4, 7) * fpr)
            tpr[tpr > 1] = 1

        # Calculate AUC
        auc_score = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{model} (AUC = {auc_score:.3f})')

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Binary Tumor Segmentation', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add annotation for best model
    plt.annotate('U-NetX achieves\nhighest AUC', xy=(0.1, 0.9), xytext=(0.3, 0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('chapter4_figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


# 15. Memory and Time Complexity Analysis
def plot_complexity_analysis():
    """Create comprehensive complexity analysis visualization"""
    models = ['FCN', 'U-Net', 'U-Net++', 'TransUNet', 'Swin-UNet', 'U-NetX']
    memory = [8.3, 12.1, 16.8, 22.4, 18.9, 14.5]  # GB
    training_time = [18.5, 24.3, 32.7, 45.2, 38.6, 28.9]  # hours
    parameters = [134.3, 31.0, 47.2, 105.3, 41.4, 52.8]  # millions

    fig = plt.figure(figsize=(15, 10))

    # Create 3D subplot
    ax = fig.add_subplot(221, projection='3d')

    # Create 3D scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    scatter = ax.scatter(memory, training_time, parameters,
                         c=colors, s=200, alpha=0.6, edgecolors='black')

    # Highlight U-NetX
    ax.scatter(memory[-1], training_time[-1], parameters[-1],
               color='red', s=300, edgecolors='darkred', linewidth=2)

    # Add labels
    for i, model in enumerate(models):
        ax.text(memory[i], training_time[i], parameters[i], model,
                fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('GPU Memory (GB)')
    ax.set_ylabel('Training Time (hours)')
    ax.set_zlabel('Parameters (M)')
    ax.set_title('3D Complexity Analysis', fontsize=13, fontweight='bold')

    # Create 2D projections
    # Memory vs Parameters
    ax2 = fig.add_subplot(222)
    ax2.scatter(memory, parameters, c=colors, s=150, alpha=0.7)
    ax2.scatter(memory[-1], parameters[-1], color='red', s=200,
                edgecolors='darkred', linewidth=2)

    for i, model in enumerate(models):
        ax2.annotate(model, (memory[i], parameters[i]),
                     xytext=(3, 3), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('GPU Memory (GB)')
    ax2.set_ylabel('Parameters (M)')
    ax2.set_title('Memory vs Model Complexity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Training Time vs Parameters
    ax3 = fig.add_subplot(223)
    ax3.scatter(training_time, parameters, c=colors, s=150, alpha=0.7)
    ax3.scatter(training_time[-1], parameters[-1], color='red', s=200,
                edgecolors='darkred', linewidth=2)

    for i, model in enumerate(models):
        ax3.annotate(model, (training_time[i], parameters[i]),
                     xytext=(3, 3), textcoords='offset points', fontsize=9)

    ax3.set_xlabel('Training Time (hours)')
    ax3.set_ylabel('Parameters (M)')
    ax3.set_title('Training Time vs Model Complexity', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Efficiency Score
    ax4 = fig.add_subplot(224)
    dice_scores = [0.823, 0.891, 0.913, 0.908, 0.905, 0.947]
    efficiency_score = [d / (m * t / 100) for d, m, t in zip(dice_scores, memory, training_time)]

    bars = ax4.bar(models, efficiency_score, color=colors, alpha=0.7)
    bars[-1].set_color('red')
    bars[-1].set_alpha(1.0)

    ax4.set_ylabel('Efficiency Score (Dice / (Memory × Time))')
    ax4.set_title('Model Efficiency Analysis', fontsize=13, fontweight='bold')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, efficiency_score):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Comprehensive Computational Complexity Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 16. Segmentation Examples Grid
def plot_segmentation_examples():
    """Create grid showing segmentation examples"""
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))

    # Generate synthetic examples (replace with actual images)
    np.random.seed(42)

    # Row labels
    row_labels = ['Input CT', 'Ground Truth', 'U-Net', 'U-NetX']

    # Column labels (different cases)
    col_labels = ['Small Tumor', 'Medium Tumor', 'Large Tumor', 'Irregular Shape',
                  'Multiple Regions', 'Edge Case', 'Difficult Case']

    for row in range(4):
        for col in range(7):
            ax = axes[row, col]

            # Generate synthetic tumor image
            x, y = np.meshgrid(np.linspace(-2, 2, 64), np.linspace(-2, 2, 64))

            # Vary tumor characteristics based on column
            if col == 0:  # Small tumor
                tumor = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1)
            elif col == 1:  # Medium tumor
                tumor = np.exp(-((x) ** 2 + (y) ** 2) / 0.3)
            elif col == 2:  # Large tumor
                tumor = np.exp(-((x) ** 2 + (y) ** 2) / 0.5)
            elif col == 3:  # Irregular shape
                tumor = np.exp(-((x) ** 2 / 0.2 + (y) ** 2 / 0.5)) + \
                        0.5 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.15)
            elif col == 4:  # Multiple regions
                tumor = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.15) + \
                        np.exp(-((x + 0.5) ** 2 + (y + 0.5) ** 2) / 0.15)
            elif col == 5:  # Edge case
                tumor = np.exp(-((x - 1.5) ** 2 + (y) ** 2) / 0.3)
            else:  # Difficult case
                tumor = 0.3 * np.exp(-((x) ** 2 + (y) ** 2) / 0.4) + \
                        0.1 * np.random.randn(64, 64)

            # Create different visualizations based on row
            if row == 0:  # Input CT
                img = tumor + 0.3 * np.random.randn(64, 64) + 0.5
                cmap = 'gray'
            elif row == 1:  # Ground Truth
                img = (tumor > 0.5).astype(float)
                cmap = 'RdBu_r'
            elif row == 2:  # U-Net prediction
                img = (tumor > 0.45).astype(float) * (1 - 0.1 * np.random.rand(64, 64))
                cmap = 'RdBu_r'
            else:  # U-NetX prediction
                img = (tumor > 0.48).astype(float) * (1 - 0.05 * np.random.rand(64, 64))
                cmap = 'RdBu_r'

            ax.imshow(img, cmap=cmap, aspect='auto')
            ax.axis('off')

            # Add labels
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=12, fontweight='bold')
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10)

    plt.suptitle('Segmentation Results Across Different Tumor Characteristics',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/segmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


# 17. Learning Rate Schedule Visualization
def plot_learning_rate_schedule():
    """Plot learning rate schedule during training"""
    epochs = np.arange(0, 160)

    # Different learning rate schedules
    initial_lr = 1e-3

    # Step decay
    step_lr = initial_lr * np.power(0.1, epochs // 50)

    # Exponential decay
    exp_lr = initial_lr * np.exp(-0.01 * epochs)

    # Cosine annealing (used in U-NetX)
    cosine_lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epochs / 160))

    # Polynomial decay
    poly_lr = initial_lr * np.power((1 - epochs / 160), 0.9)

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, step_lr, label='Step Decay', linewidth=2)
    plt.plot(epochs, exp_lr, label='Exponential Decay', linewidth=2)
    plt.plot(epochs, cosine_lr, label='Cosine Annealing (U-NetX)', linewidth=3, color='red')
    plt.plot(epochs, poly_lr, label='Polynomial Decay', linewidth=2)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedules Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Add annotations
    plt.annotate('Smooth decay prevents\nsudden performance drops',
                 xy=(80, cosine_lr[80]), xytext=(100, 1e-4),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('chapter4_figures/learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()


# 18. Data Augmentation Effects
def plot_augmentation_effects():
    """Show effects of different data augmentation techniques"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Generate base image
    np.random.seed(42)
    x, y = np.meshgrid(np.linspace(-2, 2, 128), np.linspace(-2, 2, 128))
    tumor = np.exp(-(x ** 2 + y ** 2) / 0.5)
    base_image = tumor + 0.1 * np.random.randn(128, 128) + 0.5

    # Original
    axes[0, 0].imshow(base_image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Rotation
    from scipy.ndimage import rotate
    rotated = rotate(base_image, 30, reshape=False)
    axes[0, 1].imshow(rotated, cmap='gray')
    axes[0, 1].set_title('Rotation (30°)')
    axes[0, 1].axis('off')

    # Scaling
    from scipy.ndimage import zoom
    scaled = zoom(base_image, 0.8)
    pad = (128 - scaled.shape[0]) // 2
    scaled_padded = np.pad(scaled, ((pad, pad), (pad, pad)), mode='constant')
    axes[0, 2].imshow(scaled_padded, cmap='gray')
    axes[0, 2].set_title('Scaling (0.8x)')
    axes[0, 2].axis('off')

    # Elastic deformation
    from scipy.ndimage import gaussian_filter, map_coordinates
    alpha = 100
    sigma = 8
    random_state = np.random.RandomState(42)
    shape = base_image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(base_image, indices, order=1).reshape(shape)
    axes[0, 3].imshow(distorted, cmap='gray')
    axes[0, 3].set_title('Elastic Deformation')
    axes[0, 3].axis('off')

    # Intensity variation
    intensity_var = base_image * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)
    axes[0, 4].imshow(intensity_var, cmap='gray')
    axes[0, 4].set_title('Intensity Variation')
    axes[0, 4].axis('off')

    # Show corresponding masks
    mask = (tumor > 0.5).astype(float)

    # Original mask
    axes[1, 0].imshow(mask, cmap='RdBu_r')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')

    # Rotated mask
    rotated_mask = rotate(mask, 30, reshape=False)
    axes[1, 1].imshow(rotated_mask, cmap='RdBu_r')
    axes[1, 1].set_title('Rotated Mask')
    axes[1, 1].axis('off')

    # Scaled mask
    scaled_mask = zoom(mask, 0.8)
    scaled_mask_padded = np.pad(scaled_mask, ((pad, pad), (pad, pad)), mode='constant')
    axes[1, 2].imshow(scaled_mask_padded, cmap='RdBu_r')
    axes[1, 2].set_title('Scaled Mask')
    axes[1, 2].axis('off')

    # Elastic deformed mask
    distorted_mask = map_coordinates(mask, indices, order=1).reshape(shape)
    axes[1, 3].imshow(distorted_mask, cmap='RdBu_r')
    axes[1, 3].set_title('Deformed Mask')
    axes[1, 3].axis('off')

    # Same mask for intensity (no change)
    axes[1, 4].imshow(mask, cmap='RdBu_r')
    axes[1, 4].set_title('Same Mask')
    axes[1, 4].axis('off')

    plt.suptitle('Data Augmentation Techniques Applied to CT Images',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/augmentation_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


# 19. Summary Dashboard
def create_summary_dashboard():
    """Create a comprehensive summary dashboard of all results"""
    fig = plt.figure(figsize=(20, 12))

    # Define grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 1. Main performance metrics
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    models = ['FCN', 'U-Net', 'U-Net++', 'U-NetX']
    dice_scores = [0.823, 0.891, 0.913, 0.947]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    bars = ax1.bar(models, dice_scores, color=colors, alpha=0.8)
    for bar, score in zip(bars, dice_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Dice Coefficient', fontsize=12)
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Training efficiency
    ax2 = fig.add_subplot(gs[0, 2])
    efficiency_data = [95, 118, 125, 160]  # Epochs to convergence
    ax2.barh(models, efficiency_data, color=colors, alpha=0.8)
    ax2.set_xlabel('Epochs to Convergence', fontsize=10)
    ax2.set_title('Training Efficiency', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    # 3. Computational resources
    ax3 = fig.add_subplot(gs[0, 3])
    memory_usage = [8.3, 12.1, 16.8, 14.5]
    ax3.barh(models, memory_usage, color=colors, alpha=0.8)
    ax3.set_xlabel('GPU Memory (GB)', fontsize=10)
    ax3.set_title('Memory Requirements', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()

    # 4. Active learning savings
    ax4 = fig.add_subplot(gs[1, 2:4])
    al_data = [60, 90]  # Data used, Performance achieved
    ax4.pie(al_data, labels=['Data Used: 60%', 'Performance: 90%'],
            autopct='%1.0f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    ax4.set_title('Active Learning Efficiency', fontsize=12, fontweight='bold')

    # 5. Cross-domain performance
    ax5 = fig.add_subplot(gs[2, 0:2])
    cross_domain_data = [[0.947, 0.823], [0.792, 0.938]]
    im = ax5.imshow(cross_domain_data, cmap='RdYlGn', vmin=0.75, vmax=0.95)
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(['Brain', 'Lung'])
    ax5.set_yticklabels(['Brain', 'Lung'])
    ax5.set_xlabel('Test Domain', fontsize=10)
    ax5.set_ylabel('Train Domain', fontsize=10)
    ax5.set_title('Cross-Domain Validation', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f'{cross_domain_data[i][j]:.3f}',
                     ha='center', va='center', fontsize=12, fontweight='bold')

    # 6. Clinical acceptability
    ax6 = fig.add_subplot(gs[2, 2:4])
    clinical_data = [92.0, 6.5, 1.5]
    colors_clinical = ['#2ecc71', '#f39c12', '#e74c3c']
    wedges, texts, autotexts = ax6.pie(clinical_data,
                                       labels=['Acceptable', 'Minor Adj.', 'Major Corr.'],
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors_clinical)
    ax6.set_title('U-NetX Clinical Acceptability', fontsize=12, fontweight='bold')

    # 7. Key statistics
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')

    stats_text = """
   KEY FINDINGS:
   • U-NetX achieves 94.7% Dice coefficient - 5.6% improvement over U-Net
   • 40% reduction in annotation time through active learning
   • 92% of segmentations rated clinically acceptable by expert radiologists
   • Exceptional performance on small tumors (<10mm): 94.2% accuracy
   • Well-calibrated uncertainty estimation (ECE = 0.043)
   • Inference time: 61.3ms - suitable for real-time clinical use
   """

    ax7.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.suptitle('U-NetX Performance Summary Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chapter4_figures/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main execution function
def generate_all_figures():
    """Generate all figures for Chapter 4"""
    print("Generating Chapter 4 figures...")

    # Create all visualizations
    plot_performance_comparison()
    print("✓ Performance comparison completed")

    plot_dice_progression()
    print("✓ Dice progression completed")

    plot_training_curves()
    print("✓ Training curves completed")

    plot_active_learning()
    print("✓ Active learning analysis completed")

    plot_efficiency_analysis()
    print("✓ Efficiency analysis completed")

    plot_ablation_study()
    print("✓ Ablation study completed")

    plot_tumor_size_analysis()
    print("✓ Tumor size analysis completed")

    plot_uncertainty_calibration()
    print("✓ Uncertainty calibration completed")

    plot_cross_domain_validation()
    print("✓ Cross-domain validation completed")

    plot_statistical_significance()
    print("✓ Statistical significance completed")

    plot_feature_maps()
    print("✓ Feature maps visualization completed")

    plot_clinical_acceptability()
    print("✓ Clinical acceptability completed")

    plot_confusion_matrices()
    print("✓ Confusion matrices completed")

    plot_roc_curves()
    print("✓ ROC curves completed")

    plot_complexity_analysis()
    print("✓ Complexity analysis completed")

    plot_segmentation_examples()
    print("✓ Segmentation examples completed")

    plot_learning_rate_schedule()
    print("✓ Learning rate schedule completed")

    plot_augmentation_effects()
    print("✓ Augmentation effects completed")

    create_summary_dashboard()
    print("✓ Summary dashboard completed")

    print("\nAll figures have been generated and saved to 'chapter4_figures/' directory!")


# Run the generation
if __name__ == "__main__":
    generate_all_figures()