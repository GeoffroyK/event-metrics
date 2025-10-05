from typing import Dict

import tonic
from tonic.transforms import *

from utils.collect import collect_flux_statistics
from utils.plot import plot_flux_statistics


def analyse_datasets(datasets: Dict[str, tonic.Dataset], n_samples: int, representation: Callable):
    stats_list = [
        (name, collect_flux_statistics(dataset, n_samples, name, representation)) for name, dataset in datasets.items()
    ]
    plot_flux_statistics(stats_list, 'NMNIST')


if __name__ == '__main__':
    augmentations = {
        "baseline": None,
        "flip_lr": Compose([RandomFlipLR(sensor_size=tonic.datasets.NMNIST.sensor_size, p=0.5)]),
        "time_jitter": Compose([TimeJitter(std=1.0, clip_negative=True)]),
        "spatial_jitter": Compose([
            SpatialJitter(sensor_size=tonic.datasets.NMNIST.sensor_size, var_x=2, var_y=2, clip_outliers=True)
        ]),
        "event_drop": Compose([DropEvent(p=1 / 3)]),
        "uniform_noise": Compose([UniformNoise(sensor_size=tonic.datasets.NMNIST.sensor_size, n=1000)]),
    }

    representation = ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=33000)

    datasets = {
        name: tonic.datasets.NMNIST(save_to='./data', train=True, transform=augmentation) for name, augmentation in augmentations.items()
    }

    analyse_datasets(datasets, n_samples=1000, representation=representation)
