import numpy as np
from typing import Dict, Callable, List, Tuple
from scipy.stats import entropy


def calculate_kl_divergence(p_hist: np.ndarray, q_hist: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate KL divergence between two probability distributions.

    Args:
        p_hist: Reference histogram (e.g., baseline)
        q_hist: Comparison histogram (e.g., augmented)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    # Flatten histograms
    p_flat = p_hist.flatten()
    q_flat = q_hist.flatten()

    # Normalize to probability distributions
    p_prob = p_flat / (np.sum(p_flat) + epsilon)
    q_prob = q_flat / (np.sum(q_flat) + epsilon)

    # Add epsilon to avoid log(0)
    p_prob = p_prob + epsilon
    q_prob = q_prob + epsilon

    # Re-normalize after adding epsilon
    p_prob = p_prob / np.sum(p_prob)
    q_prob = q_prob / np.sum(q_prob)

    # Calculate KL divergence
    kl_div = entropy(p_prob, q_prob)

    return kl_div


def calculate_histogram_kl_divergences(stats_list: List[Tuple[str, Dict]], baseline_name: str = "baseline") -> Dict:
    """Calculate KL divergence for each histogram metric against baseline.

    Args:
        stats_list: List of (name, stats_dict) tuples
        baseline_name: Name of the baseline dataset to compare against

    Returns:
        Dictionary with KL divergence results for each metric
    """
    # Find baseline stats
    baseline_stats = None
    for name, stats in stats_list:
        if name == baseline_name:
            baseline_stats = stats
            break

    if baseline_stats is None:
        print(f"Warning: Baseline '{baseline_name}' not found. Using first dataset as baseline.")
        baseline_stats = stats_list[0][1]

    kl_results = {}

    for name, stats in stats_list:
        if name == baseline_name:
            # Baseline has 0 divergence from itself
            kl_results[name] = {
                'spatial_histogram_kl': 0.0,
                'event_density_kl': 0.0,
                'event_count_kl': 0.0,
                'polarity_ratio_kl': 0.0,
                'nonzero_pixel_kl': 0.0
            }
            continue

        result = {}

        # 1. Spatial histogram KL divergence (average across samples)
        if 'spatial_histograms' in stats and 'spatial_histograms' in baseline_stats:
            spatial_kl_values = []
            n_samples = min(len(stats['spatial_histograms']), len(baseline_stats['spatial_histograms']))

            for i in range(n_samples):
                kl = calculate_kl_divergence(baseline_stats['spatial_histograms'][i],
                                            stats['spatial_histograms'][i])
                if np.isfinite(kl):
                    spatial_kl_values.append(kl)

            result['spatial_histogram_kl'] = np.mean(spatial_kl_values) if spatial_kl_values else float('nan')
            result['spatial_histogram_kl_std'] = np.std(spatial_kl_values) if spatial_kl_values else float('nan')

        # 2. Event density distribution KL divergence
        if 'event_densities' in stats and 'event_densities' in baseline_stats:
            # Create histograms from the data
            bins = np.linspace(
                min(baseline_stats['event_densities'].min(), stats['event_densities'].min()),
                max(baseline_stats['event_densities'].max(), stats['event_densities'].max()),
                100
            )
            baseline_hist, _ = np.histogram(baseline_stats['event_densities'], bins=bins)
            current_hist, _ = np.histogram(stats['event_densities'], bins=bins)
            result['event_density_kl'] = calculate_kl_divergence(baseline_hist, current_hist)

        # 3. Event count distribution KL divergence
        if 'event_counts' in stats and 'event_counts' in baseline_stats:
            bins = np.linspace(
                min(baseline_stats['event_counts'].min(), stats['event_counts'].min()),
                max(baseline_stats['event_counts'].max(), stats['event_counts'].max()),
                100
            )
            baseline_hist, _ = np.histogram(baseline_stats['event_counts'], bins=bins)
            current_hist, _ = np.histogram(stats['event_counts'], bins=bins)
            result['event_count_kl'] = calculate_kl_divergence(baseline_hist, current_hist)

        # 4. Polarity ratio distribution KL divergence
        if 'polarity_ratios' in stats and 'polarity_ratios' in baseline_stats:
            baseline_pol = baseline_stats['polarity_ratios'][np.isfinite(baseline_stats['polarity_ratios'])]
            current_pol = stats['polarity_ratios'][np.isfinite(stats['polarity_ratios'])]

            if len(baseline_pol) > 0 and len(current_pol) > 0:
                bins = np.linspace(
                    min(baseline_pol.min(), current_pol.min()),
                    max(baseline_pol.max(), current_pol.max()),
                    100
                )
                baseline_hist, _ = np.histogram(baseline_pol, bins=bins)
                current_hist, _ = np.histogram(current_pol, bins=bins)
                result['polarity_ratio_kl'] = calculate_kl_divergence(baseline_hist, current_hist)

        # 5. Non-zero pixel percentage distribution KL divergence
        if 'nonzero_pixel_percentages' in stats and 'nonzero_pixel_percentages' in baseline_stats:
            bins = np.linspace(
                min(baseline_stats['nonzero_pixel_percentages'].min(),
                    stats['nonzero_pixel_percentages'].min()),
                max(baseline_stats['nonzero_pixel_percentages'].max(),
                    stats['nonzero_pixel_percentages'].max()),
                100
            )
            baseline_hist, _ = np.histogram(baseline_stats['nonzero_pixel_percentages'], bins=bins)
            current_hist, _ = np.histogram(stats['nonzero_pixel_percentages'], bins=bins)
            result['nonzero_pixel_kl'] = calculate_kl_divergence(baseline_hist, current_hist)

        kl_results[name] = result

    return kl_results


def collect_flux_statistics(dataset, n_samples: int, dataset_name: str, representation: Callable) -> Dict:
    """Collect event flux statistics using a tonic representation function."""

    print(f"Analyzing event flux for {n_samples} {dataset_name} samples using {representation.__class__.__name__}...")

    event_densities = []
    event_counts = []
    polarity_ratios = []
    nonzero_pixel_percentages = []
    spatial_histograms = []

    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, size=min(n_samples, dataset_size), replace=False)

    for i in indices:
        events, _ = dataset[i]

        if len(events) > 0:
            # Apply tonic representation to get frames
            frames = representation(events)

            # Calculate event density (events per second)
            start_time = events['t'][0]
            end_time = events['t'][-1]
            total_duration = end_time - start_time

            if total_duration > 0:
                density = len(events) / (total_duration / 1e6)  # events/second
                event_densities.append(density)

            # Analyze each frame
            if len(frames.shape) == 4:  # (T, C, H, W)
                for frame_idx in range(frames.shape[0]):
                    frame = frames[frame_idx]

                    # Event count (sum of all values in frame)
                    event_count = np.sum(frame)
                    event_counts.append(event_count)

                    # Polarity ratio (assuming 2 channels for pos/neg)
                    if frame.shape[0] == 2:
                        pos_events = np.sum(frame[0])
                        neg_events = np.sum(frame[1])
                        ratio = pos_events / neg_events if neg_events > 0 else float('inf')
                        polarity_ratios.append(ratio)

                    # Non-zero pixel percentage
                    combined_frame = np.sum(frame, axis=0)
                    total_pixels = combined_frame.size
                    nonzero_pixels = np.count_nonzero(combined_frame)
                    nonzero_percentage = (nonzero_pixels / total_pixels) * 100
                    nonzero_pixel_percentages.append(nonzero_percentage)

            # Create spatial histogram for the entire sample
            height, width = frames.shape[-2], frames.shape[-1]
            spatial_hist = np.zeros((height, width))

            for event in events:
                x, y = event['x'], event['y']
                if 0 <= x < width and 0 <= y < height:
                    spatial_hist[y, x] += 1

            spatial_histograms.append(spatial_hist)

    return {
        'event_densities': np.array(event_densities),
        'event_counts': np.array(event_counts),
        'polarity_ratios': np.array(polarity_ratios),
        'nonzero_pixel_percentages': np.array(nonzero_pixel_percentages),
        'spatial_histograms': np.array(spatial_histograms)
    }
