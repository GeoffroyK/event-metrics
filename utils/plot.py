from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_3d_spatial_histogram(train_stats: Dict, test_stats: Dict, dataset_name: str):
    """Plot 3D surface showing event distribution over spatial coordinates."""

    if 'spatial_histograms' not in train_stats or 'spatial_histograms' not in test_stats:
        print("Spatial histogram data not available. Make sure to collect spatial statistics.")
        return

    fig = plt.figure(figsize=(16, 6))

    # Train data 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    train_hist = np.mean(train_stats['spatial_histograms'], axis=0)  # Average across samples

    height, width = train_hist.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    surf1 = ax1.plot_surface(X, Y, train_hist, cmap='viridis', alpha=0.8)
    ax1.set_title(f'{dataset_name} Train - Average Event Density')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_zlabel('Average Event Count')

    # Test data 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    test_hist = np.mean(test_stats['spatial_histograms'], axis=0)

    # Use same z-scale for fair comparison
    z_max = max(train_hist.max(), test_hist.max())
    ax1.set_zlim(0, z_max)
    ax2.set_zlim(0, z_max)

    surf2 = ax2.plot_surface(X, Y, test_hist, cmap='viridis', alpha=0.8)
    ax2.set_title(f'{dataset_name} Test - Average Event Density')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_zlabel('Average Event Count')

    plt.tight_layout()
    plt.show()


def plot_flux_statistics(train_stats: Dict, test_stats: Dict, dataset_name: str):
    """Plot comprehensive flux statistics comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} Event Flux Statistics', fontsize=16, fontweight='bold')

    # 1. Event Density Distribution
    ax1 = axes[0, 0]
    ax1.hist(train_stats['event_densities'], bins=50, alpha=0.7, label='Train', density=True)
    ax1.hist(test_stats['event_densities'], bins=50, alpha=0.7, label='Test', density=True)
    ax1.set_xlabel('Event Density (events/second)')
    ax1.set_ylabel('Density')
    ax1.set_title('Event Density Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Number of Events Distribution
    ax2 = axes[0, 1]
    ax2.hist(train_stats['event_counts'], bins=50, alpha=0.7, label='Train', density=True)
    ax2.hist(test_stats['event_counts'], bins=50, alpha=0.7, label='Test', density=True)
    ax2.set_xlabel('Number of Events per Window')
    ax2.set_ylabel('Density')
    ax2.set_title('Event Count Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Polarity Ratio Distribution
    ax3 = axes[1, 0]
    # Filter out infinite values for plotting
    train_pol_filtered = train_stats['polarity_ratios'][np.isfinite(train_stats['polarity_ratios'])]
    test_pol_filtered = test_stats['polarity_ratios'][np.isfinite(test_stats['polarity_ratios'])]

    ax3.hist(train_pol_filtered, bins=15, alpha=0.7, label='Train', density=True)
    ax3.hist(test_pol_filtered, bins=15, alpha=0.7, label='Test', density=True)
    ax3.set_xlabel('Polarity Ratio (Positive/Negative)')
    ax3.set_ylabel('Density')
    ax3.set_title('Polarity Ratio Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Non-zero Pixel Percentage Distribution
    ax4 = axes[1, 1]
    ax4.hist(train_stats['nonzero_pixel_percentages'], bins=30, alpha=0.7, label='Train', density=True)
    ax4.hist(test_stats['nonzero_pixel_percentages'], bins=30, alpha=0.7, label='Test', density=True)
    ax4.set_xlabel('Non-zero Pixel Percentage')
    ax4.set_ylabel('Density')
    ax4.set_title('Non-zero Pixel Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plot_3d_spatial_histogram(train_stats, test_stats, dataset_name)

    # Print summary statistics
    print(f"\n{dataset_name} Flux Statistics Summary:")
    print("-" * 50)
    print(
        f"Event Density (events/s) - Train: {np.mean(train_stats['event_densities']):.2f} ± {np.std(train_stats['event_densities']):.2f}")
    print(
        f"Event Density (events/s) - Test:  {np.mean(test_stats['event_densities']):.2f} ± {np.std(test_stats['event_densities']):.2f}")
    print(
        f"Avg Events/Window - Train: {np.mean(train_stats['event_counts']):.2f} ± {np.std(train_stats['event_counts']):.2f}")
    print(
        f"Avg Events/Window - Test:  {np.mean(test_stats['event_counts']):.2f} ± {np.std(test_stats['event_counts']):.2f}")
    print(
        f"Avg Non-zero Pixels (%) - Train: {np.mean(train_stats['nonzero_pixel_percentages']):.2f} ± {np.std(train_stats['nonzero_pixel_percentages']):.2f}")
    print(
        f"Avg Non-zero Pixels (%) - Test:  {np.mean(test_stats['nonzero_pixel_percentages']):.2f} ± {np.std(test_stats['nonzero_pixel_percentages']):.2f}")