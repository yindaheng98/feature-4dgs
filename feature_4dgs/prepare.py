import os
import tempfile
from typing import List, Tuple

import torch
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.prepare import prepare_gaussians
from feature_3dgs import SemanticGaussianModel
from .registry import build_extractor_decoder

from .extractor import SequenceFeatureCameraDataset


def prepare_datasets(sources: List[str], device: str, trainable_camera: bool = False, load_cameras: List[str] = None, load_mask=True, load_depth=True) -> List[CameraDataset]:
    """Load one :class:`CameraDataset` per timestep by calling
    :func:`gaussian_splatting.prepare.prepare_dataset` for each source.
    """
    load_cameras = load_cameras if load_cameras is not None else [None] * len(sources)
    assert len(load_cameras) == len(sources), "len(load_cameras) must equal len(sources)"
    return [
        prepare_dataset(
            source=source, device=device,
            trainable_camera=trainable_camera, load_camera=load_camera,
            load_mask=load_mask, load_depth=load_depth,
        )
        for source, load_camera in zip(sources, load_cameras)
    ]


def prepare_datasets_and_decoder(
        name: str, sources: List[str], embed_dim: int, device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_cameras: List[str] = None, load_mask=True, load_depth=True,
        preload_cache: bool = True, configs={},
) -> Tuple[SequenceFeatureCameraDataset, AbstractTrainableDecoder]:
    """Prepare a :class:`SequenceFeatureCameraDataset` and a shared decoder.

    All timesteps share a single extractor/decoder pair.  The
    :class:`AbstractSequenceFeatureExtractor` returned by
    :func:`build_extractor_decoder` is used directly.
    """
    cameras_list = prepare_datasets(
        sources=sources, device=device,
        trainable_camera=trainable_camera, load_cameras=load_cameras,
        load_mask=load_mask, load_depth=load_depth,
    )
    extractor, decoder = build_extractor_decoder(
        name=name, embed_dim=embed_dim, **configs
    )
    datasets = SequenceFeatureCameraDataset(cameras_list=cameras_list, extractor=extractor, cache_device=dataset_cache_device).to(device)
    if preload_cache:
        datasets.preload_cache()
    return datasets, decoder


def prepare_gaussians_sequence(
        decoder: AbstractTrainableDecoder, sh_degree: int,
        sources: List[str], datasets: SequenceFeatureCameraDataset, device: str,
        trainable_camera: bool = False, load_plys: List[str] = None, load_semantic: bool = True,
) -> List[SemanticGaussianModel]:
    """Prepare one :class:`SemanticGaussianModel` per timestep, ensuring a
    consistent decoder initialisation across all frames.

    After the first frame is prepared its decoder state is saved to a temporary
    file.  Every subsequent frame that would go through ``init_semantic``
    (i.e. ``not load_ply or not load_semantic``) receives that file via the
    ``load_decoder`` parameter so all frames start from the same weights.
    """
    load_plys = load_plys if load_plys is not None else [None] * len(sources)
    assert len(load_plys) == len(sources), "len(load_plys) must equal len(sources)"

    first_gaussians = prepare_gaussians(
        decoder=decoder, sh_degree=sh_degree, source=sources[0], dataset=datasets[0],
        device=device, trainable_camera=trainable_camera,
        load_ply=load_plys[0], load_semantic=load_semantic,
    )
    gaussians_list: List[SemanticGaussianModel] = [first_gaussians]

    with tempfile.TemporaryDirectory(prefix="feature4dgs_decoder_") as tmp_decoder_dir:
        tmp_decoder_path = os.path.join(tmp_decoder_dir, "init")
        torch.save(decoder.state_dict(), tmp_decoder_path + ".decoder.pt")

        for source, dataset, load_ply in zip(sources[1:], datasets[1:], load_plys[1:]):
            load_decoder = tmp_decoder_path if (not load_ply or not load_semantic) else None
            gaussians = prepare_gaussians(
                decoder=decoder, sh_degree=sh_degree, source=source, dataset=dataset, device=device,
                trainable_camera=trainable_camera, load_ply=load_ply, load_semantic=load_semantic,
                load_decoder=load_decoder,
            )
            gaussians_list.append(gaussians)

    return gaussians_list
