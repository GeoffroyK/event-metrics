from typing import List

import tonic

from utils.collect import collect_flux_statistics
from utils.plot import plot_flux_statistics


def analyse_datasets(datasets: List, n_samples: int = 1000, time_window_ms: float = 33.3):
    train_stats = collect_flux_statistics(datasets[0], n_samples, "train", time_window_ms)
    test_stats = collect_flux_statistics(datasets[1], n_samples, "test", time_window_ms)

    plot_flux_statistics(train_stats, test_stats, "NMNIST")


if __name__ == '__main__':
    mnist_train_dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
    mnist_test_dataset = tonic.datasets.NMNIST(save_to='./data', train=False)

    analyse_datasets([mnist_train_dataset, mnist_test_dataset], n_samples=1000, time_window_ms=33.3)
