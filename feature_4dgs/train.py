import os
import random
import shutil
from typing import List, Tuple

import torch
from tqdm import tqdm

from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.train import save_cfg_args
from gaussian_splatting.utils import psnr
from feature_3dgs import SemanticGaussianModel
from feature_3dgs.prepare import prepare_trainer, modes

from .extractor import SequenceFeatureCameraDataset
from .prepare import prepare_datasets_and_decoder, prepare_gaussians_sequence


def prepare_training(
        name: str, sh_degree: int, mode: str, sources: List[str], embed_dim: int,
        device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_plys: List[str] = None, load_cameras: List[str] = None,
        load_mask=True, load_depth=True, load_semantic: bool = True,
        preload_cache: bool = True, configs={}, extractor_configs={},
) -> Tuple[SequenceFeatureCameraDataset, List[SemanticGaussianModel], List[AbstractTrainer]]:
    """Prepare a sequence dataset, per-timestep gaussians and trainers.

    All timesteps share a single decoder (owned by the extractor/decoder
    factory) that is attached to every :class:`SemanticGaussianModel`.
    """
    datasets, decoder = prepare_datasets_and_decoder(
        name=name, sources=sources, embed_dim=embed_dim, device=device, dataset_cache_device=dataset_cache_device,
        trainable_camera=trainable_camera, load_cameras=load_cameras,
        load_mask=load_mask, load_depth=load_depth, preload_cache=preload_cache, configs=extractor_configs,
    )
    gaussians_list = prepare_gaussians_sequence(
        decoder=decoder, sh_degree=sh_degree, sources=sources, datasets=datasets, device=device,
        trainable_camera=trainable_camera, load_plys=load_plys, load_semantic=load_semantic,
    )
    trainers = [prepare_trainer(
        gaussians=gaussians, dataset=dataset, mode=mode,
        trainable_camera=trainable_camera, configs=configs,
    ) for gaussians, dataset in zip(gaussians_list, datasets)]
    return datasets, gaussians_list, trainers
