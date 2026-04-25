import os
from typing import List

import tqdm
import torch

from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from feature_3dgs.extractor import FeatureCameraDataset, TrainableFeatureCameraDataset

from .extractor import AbstractSequenceFeatureExtractor


class SequenceFeatureCameraDataset:
    """A sequence of per-timestep :class:`FeatureCameraDataset`.

    Takes multiple :class:`CameraDataset` objects (one per timestep) plus a
    shared :class:`AbstractSequenceFeatureExtractor`, and wraps each of them
    as a :class:`FeatureCameraDataset` (or :class:`TrainableFeatureCameraDataset`
    when the input is a :class:`TrainableCameraDataset`).  Indexing the
    sequence with ``seq[t]`` returns the ``t``-th timestep's feature dataset.
    """

    def __init__(self, cameras_list: List[CameraDataset], extractor: AbstractSequenceFeatureExtractor, cache_device=None):
        self.extractor = extractor
        self.cache_device = cache_device
        self.datasets: List[FeatureCameraDataset] = []
        for cameras in cameras_list:
            if isinstance(cameras, TrainableCameraDataset):
                cls = TrainableFeatureCameraDataset
            elif isinstance(cameras, CameraDataset):
                cls = FeatureCameraDataset
            else:
                raise ValueError(f"Invalid camera dataset type: {type(cameras)}")
            self.datasets.append(cls(cameras=cameras, extractor=extractor, cache_device=cache_device))

    def to(self, device) -> 'SequenceFeatureCameraDataset':
        for dataset in self.datasets:
            dataset.to(device)
        return self

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx) -> FeatureCameraDataset:
        return self.datasets[idx]

    @property
    def embed_dim(self) -> int:
        return self.datasets[0].embed_dim

    def preload_cache(self):
        sequences = ((camera.ground_truth_image for camera in dataset.cameras) for dataset in self.datasets)
        for dataset, feature_maps in zip(self.datasets, self.extractor.extract_sequence_all(sequences)):
            dataset.feature_map_cache = []
            for feature_map in tqdm.tqdm(feature_maps, total=len(dataset.cameras), desc=f"Preloading feature maps"):
                if dataset.cache_device is not None:
                    feature_map = feature_map.to(dataset.cache_device)
                dataset.feature_map_cache.append(feature_map)
                torch.cuda.empty_cache()
            del dataset.extractor
        del self.extractor
