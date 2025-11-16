from typing import Dict

import tonic
from scipy.ndimage import zoom
from tonic.transforms import *

from utils.collect import collect_flux_statistics
from utils.plot import plot_flux_statistics


def analyse_datasets(datasets: Dict[str, tonic.Dataset], n_samples: int, representation: Callable):
    stats_list = [
        (name, collect_flux_statistics(dataset, n_samples, name, representation)) for name, dataset in datasets.items()
    ]
    plot_flux_statistics(stats_list, 'CIFAR10DVS')


if __name__ == '__main__':
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size


    class NumpyResize:
        def __init__(self, size, interpolation='bilinear'):
            self.size = size
            # Map interpolation names to scipy order
            self.order = {
                'nearest': 0,
                'bilinear': 1,
                'bicubic': 3,
            }.get(interpolation, 1)

        def __call__(self, frames):
            if frames.ndim == 3:
                T, H, W = frames.shape
                scale_factors = (1, self.size[0] / H, self.size[1] / W)
            else:  # ndim == 4
                T, C, H, W = frames.shape
                scale_factors = (1, 1, self.size[0] / H, self.size[1] / W)

            return zoom(frames, scale_factors, order=self.order)


    augmentations = {
        "baseline": None,
        "flip_lr": Compose([RandomFlipLR(sensor_size=sensor_size, p=0.5)]),
        "time_jitter": Compose([TimeJitter(std=1.0, clip_negative=True)]),
        "spatial_jitter": Compose([
            SpatialJitter(sensor_size=sensor_size, var_x=2, var_y=2, clip_outliers=True)
        ]),
        "event_drop": Compose([DropEvent(p=1 / 3)]),
        "uniform_noise": Compose([UniformNoise(sensor_size=sensor_size, n=150_000)]),
    }

    representation = tonic.transforms.Compose([
        ToFrame(sensor_size=sensor_size, time_window=33000),
        NumpyResize((48, 48))
    ])


    datasets = {
        name: tonic.datasets.CIFAR10DVS(save_to='./data', transform=augmentation) for name, augmentation in augmentations.items()
    }

    analyse_datasets(datasets, n_samples=1000, representation=representation)
