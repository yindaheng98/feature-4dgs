from typing import List, Tuple

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset
from feature_3dgs.decoder import AbstractTrainableDecoder
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
