import numpy as np
from typing import Dict


def collect_flux_statistics(dataset, n_samples: int, dataset_name: str, time_window_ms: float = 33.0) -> Dict:
    """Collect event flux statistics over time windows."""

    print(f"Analyzing event flux for {n_samples} {dataset_name} samples (window: {time_window_ms}ms)...")

    event_densities = []
    event_counts = []
    polarity_ratios = []
    nonzero_pixel_percentages = []
    spatial_histograms = []  # Move this outside the loop

    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, size=min(n_samples, dataset_size), replace=False)

    for i in indices:
        events, _ = dataset[i]

        # Convert time window to microseconds (tonic uses microseconds)
        time_window_us = time_window_ms * 1000

        # Get time range
        if len(events) > 0:
            start_time = events['t'][0]
            end_time = events['t'][-1]
            total_duration = end_time - start_time

            if total_duration > 0:
                # Calculate event density (events per second)
                density = len(events) / (total_duration / 1e6)  # events/second
                event_densities.append(density)

                # Split into time windows for analysis
                n_windows = max(1, int(total_duration / time_window_us))

                window_event_counts = []
                window_polarity_ratios = []
                window_nonzero_pixels = []

                for w in range(n_windows):
                    window_start = start_time + w * time_window_us
                    window_end = window_start + time_window_us

                    # Events in this window
                    mask = (events['t'] >= window_start) & (events['t'] < window_end)
                    window_events = events[mask]

                    if len(window_events) > 0:
                        # Event count
                        window_event_counts.append(len(window_events))

                        # Polarity ratio
                        pos_events = np.sum(window_events['p'] == 1)
                        neg_events = np.sum(window_events['p'] == 0)
                        ratio = pos_events / neg_events if neg_events > 0 else float('inf')
                        window_polarity_ratios.append(ratio)

                        # Create frame representation for non-zero pixels
                        height, width = 34, 34  # NMNIST dimensions
                        frame = np.zeros((2, height, width))

                        # Accumulate events in frame
                        for event in window_events:
                            x, y, p = event['x'], event['y'], event['p']
                            if 0 <= x < width and 0 <= y < height:
                                frame[p, y, x] += 1

                        # Calculate non-zero pixel percentage
                        combined_frame = frame[0] + frame[1]
                        total_pixels = combined_frame.size
                        nonzero_pixels = np.count_nonzero(combined_frame)
                        nonzero_percentage = (nonzero_pixels / total_pixels) * 100
                        window_nonzero_pixels.append(nonzero_percentage)

                # Store window statistics
                event_counts.extend(window_event_counts)
                polarity_ratios.extend(window_polarity_ratios)
                nonzero_pixel_percentages.extend(window_nonzero_pixels)

            # Create spatial histogram for the entire sample
            height, width = 34, 34  # NMNIST dimensions
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
