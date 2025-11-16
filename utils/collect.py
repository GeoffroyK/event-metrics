import numpy as np
from typing import Dict, Callable, List, Tuple
from scipy.stats import entropy
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def _build_spatial_histogram_fast(x_coords, y_coords, height, width):
    """Fast spatial histogram using numba."""
    hist = np.zeros((height, width), dtype=np.int32)
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if 0 <= x < width and 0 <= y < height:
            hist[y, x] += 1
    return hist


def calculate_kl_divergence(p_hist: np.ndarray, q_hist: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate KL divergence between two probability distributions."""
    # Flatten and normalize in one go
    p_flat = p_hist.ravel()
    q_flat = q_hist.ravel()

    p_sum = p_flat.sum()
    q_sum = q_flat.sum()

    # Avoid division if sum is zero
    if p_sum == 0 or q_sum == 0:
        return float('nan')

    # Normalize and add epsilon
    p_prob = (p_flat / p_sum) + epsilon
    q_prob = (q_flat / q_sum) + epsilon

    # Re-normalize
    p_prob /= p_prob.sum()
    q_prob /= q_prob.sum()

    return entropy(p_prob, q_prob)


def calculate_histogram_kl_divergences(stats_list: List[Tuple[str, Dict]], baseline_name: str = "baseline") -> Dict:
    """Calculate KL divergence for each histogram metric against baseline."""
    # Find baseline stats
    baseline_stats = next((stats for name, stats in stats_list if name == baseline_name), None)

    if baseline_stats is None:
        print(f"Warning: Baseline '{baseline_name}' not found. Using first dataset as baseline.")
        baseline_stats = stats_list[0][1]
        baseline_name = stats_list[0][0]

    kl_results = {}

    # Pre-compute baseline histograms to avoid repeated computation
    baseline_histograms = {}

    for metric in ['event_densities', 'event_counts', 'polarity_ratios', 'nonzero_pixel_percentages', 'event_durations']:
        if metric in baseline_stats:
            data = baseline_stats[metric]
            if metric == 'polarity_ratios':
                data = data[np.isfinite(data)]
            if len(data) > 0:
                bins = np.histogram_bin_edges(data, bins=100)
                hist, _ = np.histogram(data, bins=bins)
                baseline_histograms[metric] = (hist, bins)

    for name, stats in stats_list:
        if name == baseline_name:
            kl_results[name] = {
                'spatial_histogram_kl': 0.0,
                'event_density_kl': 0.0,
                'event_count_kl': 0.0,
                'polarity_ratio_kl': 0.0,
                'nonzero_pixel_kl': 0.0,
                'event_duration_kl': 0.0
            }
            continue

        result = {}

        # 1. Spatial histogram KL divergence
        if 'spatial_histograms' in stats and 'spatial_histograms' in baseline_stats:
            n_samples = min(len(stats['spatial_histograms']), len(baseline_stats['spatial_histograms']))

            # Vectorized KL computation
            kl_values = np.array([
                calculate_kl_divergence(baseline_stats['spatial_histograms'][i], stats['spatial_histograms'][i])
                for i in range(n_samples)
            ])

            kl_values = kl_values[np.isfinite(kl_values)]
            result['spatial_histogram_kl'] = np.mean(kl_values) if len(kl_values) > 0 else float('nan')
            result['spatial_histogram_kl_std'] = np.std(kl_values) if len(kl_values) > 0 else float('nan')

        # 2-6. Use pre-computed baseline histograms
        for metric, key in [
            ('event_densities', 'event_density_kl'),
            ('event_counts', 'event_count_kl'),
            ('polarity_ratios', 'polarity_ratio_kl'),
            ('nonzero_pixel_percentages', 'nonzero_pixel_kl'),
            ('event_durations', 'event_duration_kl')
        ]:
            if metric in stats and metric in baseline_histograms:
                data = stats[metric]
                if metric == 'polarity_ratios':
                    data = data[np.isfinite(data)]

                if len(data) > 0:
                    baseline_hist, bins = baseline_histograms[metric]
                    current_hist, _ = np.histogram(data, bins=bins)
                    result[key] = calculate_kl_divergence(baseline_hist, current_hist)

        kl_results[name] = result

    return kl_results


def collect_flux_statistics(dataset, n_samples: int, dataset_name: str, representation: Callable) -> Dict:
    """Collect event flux statistics using a tonic representation function."""
    print(f"Analyzing event flux for {n_samples} {dataset_name} samples...")

    # Pre-allocate lists with approximate capacity
    stats_dict = {
        'event_densities': [],
        'event_counts': [],
        'polarity_ratios': [],
        'nonzero_pixel_percentages': [],
        'spatial_histograms': [],
        'event_durations': []
    }

    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, size=min(n_samples, dataset_size), replace=False)

    for i in tqdm(indices):
        events, _ = dataset[i]

        if len(events) == 0:
            continue

        # Apply representation
        frames = representation(events)

        # Calculate event density
        time_span = events['t'][-1] - events['t'][0]
        if time_span > 0:
            stats_dict['event_densities'].append(len(events) / (time_span / 1e6))
            # Store duration in microseconds
            stats_dict['event_durations'].append(time_span)

        # Process frames in batch where possible
        if len(frames.shape) == 4:  # (T, C, H, W)
            # Vectorized frame statistics
            event_counts = frames.sum(axis=(1, 2, 3))
            stats_dict['event_counts'].extend(event_counts)

            if frames.shape[1] == 2:  # Polarity channels
                pos_events = frames[:, 0].sum(axis=(1, 2)).astype(np.float64)
                neg_events = frames[:, 1].sum(axis=(1, 2)).astype(np.float64)
                # Create float array for division output
                ratios = np.full(pos_events.shape, np.inf, dtype=np.float64)
                np.divide(pos_events, neg_events, where=neg_events > 0, out=ratios)
                stats_dict['polarity_ratios'].extend(ratios)

            # Non-zero pixel percentages
            combined = frames.sum(axis=1)  # Sum over channels
            total_pixels = combined.shape[1] * combined.shape[2]
            nonzero = (combined != 0).sum(axis=(1, 2))
            percentages = (nonzero / total_pixels) * 100
            stats_dict['nonzero_pixel_percentages'].extend(percentages)

        # Build spatial histogram using optimized function
        height, width = frames.shape[-2], frames.shape[-1]
        spatial_hist = _build_spatial_histogram_fast(events['x'], events['y'], height, width)
        stats_dict['spatial_histograms'].append(spatial_hist)

    # Convert to numpy arrays
    return {k: np.array(v) for k, v in stats_dict.items()}
