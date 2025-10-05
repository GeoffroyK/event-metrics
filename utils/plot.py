from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.collect import calculate_histogram_kl_divergences


def plot_3d_spatial_histogram(stats_list: List[Tuple[str, Dict]], dataset_name: str):
    """Plot 3D surface showing event distribution over spatial coordinates."""

    # Filter stats that have spatial histogram data
    valid_stats = [(name, stats) for name, stats in stats_list
                   if 'spatial_histograms' in stats]

    if not valid_stats:
        print("Spatial histogram data not available. Make sure to collect spatial statistics.")
        return

    n_datasets = len(valid_stats)

    # Calculate grid dimensions for better layout
    n_cols = min(2, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{dataset_name} {name} - Average Event Density'
                        for name, _ in valid_stats],
        specs=[[{'type': 'surface'}] * n_cols for _ in range(n_rows)]
    )

    # Calculate global z_max for consistent scaling
    z_max = max(np.mean(stats['spatial_histograms'], axis=0).max()
                for _, stats in valid_stats)

    for idx, (name, stats) in enumerate(valid_stats):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        hist = np.mean(stats['spatial_histograms'], axis=0)
        height, width = hist.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        fig.add_trace(
            go.Surface(x=X, y=Y, z=hist, colorscale='Viridis',
                       showscale=(idx == n_datasets - 1)),
            row=row, col=col
        )

    fig.update_scenes(
        xaxis_title='X coordinate',
        yaxis_title='Y coordinate',
        zaxis_title='Average Event Count',
        zaxis_range=[0, z_max]
    )

    fig.update_layout(height=600 * n_rows, width=800 * n_cols, showlegend=False)
    fig.show()


def plot_flux_statistics(stats_list: List[Tuple[str, Dict]], dataset_name: str):
    """Plot comprehensive flux statistics comparison.

    Args:
        stats_list: List of tuples (name, stats_dict) for each dataset split
        dataset_name: Name of the overall dataset
    """

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Event Density Distribution', 'Event Count Distribution',
                        'Polarity Ratio Distribution', 'Non-zero Pixel Distribution')
    )

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for idx, (split_name, stats) in enumerate(stats_list):
        color = colors[idx % len(colors)]
        show_legend = True  # Only show legend for first two datasets to avoid clutter

        # 1. Event Density Distribution
        fig.add_trace(
            go.Histogram(x=stats['event_densities'], name=split_name, opacity=0.8,
                         histnorm='probability density', nbinsx=100,
                         marker_color=color, showlegend=show_legend,
                         legendgroup=split_name),
            row=1, col=1
        )

        # 2. Number of Events Distribution
        fig.add_trace(
            go.Histogram(x=stats['event_counts'], name=split_name, opacity=0.8,
                         histnorm='probability density', nbinsx=100,
                         marker_color=color, showlegend=False,
                         legendgroup=split_name),
            row=1, col=2
        )

        # 3. Polarity Ratio Distribution
        pol_filtered = stats['polarity_ratios'][np.isfinite(stats['polarity_ratios'])]
        fig.add_trace(
            go.Histogram(x=pol_filtered, name=split_name, opacity=0.8,
                         histnorm='probability density', nbinsx=100,
                         marker_color=color, showlegend=False,
                         legendgroup=split_name),
            row=2, col=1
        )

        # 4. Non-zero Pixel Percentage Distribution
        fig.add_trace(
            go.Histogram(x=stats['nonzero_pixel_percentages'], name=split_name, opacity=0.8,
                         histnorm='probability density', nbinsx=100,
                         marker_color=color, showlegend=False,
                         legendgroup=split_name),
            row=2, col=2
        )

    # Rest of the function remains the same
    fig.update_xaxes(title_text='Event Density (events/second)', row=1, col=1)
    fig.update_xaxes(title_text='Number of Events per Window', row=1, col=2)
    fig.update_xaxes(title_text='Polarity Ratio (Positive/Negative)', row=2, col=1)
    fig.update_xaxes(title_text='Non-zero Pixel Percentage', row=2, col=2)

    fig.update_yaxes(title_text='Density', row=1, col=1)
    fig.update_yaxes(title_text='Density', row=1, col=2)
    fig.update_yaxes(title_text='Density', row=2, col=1)
    fig.update_yaxes(title_text='Density', row=2, col=2)

    fig.update_layout(
        title_text=f'{dataset_name} Event Flux Statistics',
        height=1200,
        width=1500,
        barmode='overlay',
        showlegend=True
    )

    fig.show()

    plot_3d_spatial_histogram(stats_list, dataset_name)

    # Print summary statistics
    print(f"\n{dataset_name} Flux Statistics Summary:")
    print("-" * 50)
    for split_name, stats in stats_list:
        print(f"\n{split_name}:")
        print(
            f"  Event Density (events/s): {np.mean(stats['event_densities']):.2f} ± {np.std(stats['event_densities']):.2f}")
        print(f"  Avg Events/Window: {np.mean(stats['event_counts']):.2f} ± {np.std(stats['event_counts']):.2f}")
        print(
            f"  Avg Non-zero Pixels (%): {np.mean(stats['nonzero_pixel_percentages']):.2f} ± {np.std(stats['nonzero_pixel_percentages']):.2f}")

    # Calculate and display KL divergences
    print("\n" + "=" * 70)
    print("KL DIVERGENCE ANALYSIS (comparing to baseline)")
    print("=" * 70)

    kl_results = calculate_histogram_kl_divergences(stats_list, baseline_name="baseline")

    for name, kl_metrics in kl_results.items():
        print(f"\n{name}:")
        if kl_metrics.get('spatial_histogram_kl', float('nan')) == 0.0:
            print("  [BASELINE - all KL divergences = 0.0]")
        else:
            if 'spatial_histogram_kl' in kl_metrics:
                print(f"  Spatial Histogram KL:     {kl_metrics['spatial_histogram_kl']:.6f} ± {kl_metrics.get('spatial_histogram_kl_std', 0):.6f}")
            if 'event_density_kl' in kl_metrics:
                print(f"  Event Density KL:         {kl_metrics['event_density_kl']:.6f}")
            if 'event_count_kl' in kl_metrics:
                print(f"  Event Count KL:           {kl_metrics['event_count_kl']:.6f}")
            if 'polarity_ratio_kl' in kl_metrics:
                print(f"  Polarity Ratio KL:        {kl_metrics['polarity_ratio_kl']:.6f}")
            if 'nonzero_pixel_kl' in kl_metrics:
                print(f"  Non-zero Pixel KL:        {kl_metrics['nonzero_pixel_kl']:.6f}")

    # Create bar plot for KL divergences
    plot_kl_divergences(kl_results, dataset_name)


def plot_kl_divergences(kl_results: Dict, dataset_name: str):
    """Plot bar chart comparing KL divergences across different augmentations.

    Args:
        kl_results: Dictionary with KL divergence results for each dataset
        dataset_name: Name of the dataset
    """
    # Prepare data for plotting
    dataset_names = []
    metrics = ['spatial_histogram_kl', 'event_density_kl', 'event_count_kl',
               'polarity_ratio_kl', 'nonzero_pixel_kl']
    metric_labels = ['Spatial Histogram', 'Event Density', 'Event Count',
                     'Polarity Ratio', 'Non-zero Pixels']

    data_by_metric = {metric: [] for metric in metrics}

    for name, kl_metrics in kl_results.items():
        if name != "baseline":  # Skip baseline since it's always 0
            dataset_names.append(name)
            for metric in metrics:
                value = kl_metrics.get(metric, float('nan'))
                data_by_metric[metric].append(value)

    # Create grouped bar chart
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        fig.add_trace(go.Bar(
            name=label,
            x=dataset_names,
            y=data_by_metric[metric],
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title=f'{dataset_name} - KL Divergence from Baseline',
        xaxis_title='Augmentation Method',
        yaxis_title='KL Divergence',
        barmode='group',
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    fig.show()
