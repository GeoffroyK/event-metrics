from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

