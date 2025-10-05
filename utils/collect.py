import numpy as np
from typing import Dict, Callable


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
