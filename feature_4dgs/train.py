from typing import List, Tuple


from gaussian_splatting.trainer import AbstractTrainer
from feature_3dgs import SemanticGaussianModel
from feature_3dgs.prepare import prepare_gaussians, prepare_trainer

from .extractor import SequenceFeatureCameraDataset
from .prepare import prepare_datasets_and_decoder


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
        name=name, sources=sources, embed_dim=embed_dim,
        device=device, dataset_cache_device=dataset_cache_device,
        trainable_camera=trainable_camera, load_cameras=load_cameras,
        load_mask=load_mask, load_depth=load_depth,
        preload_cache=preload_cache, configs=extractor_configs,
    )
    load_plys = load_plys if load_plys is not None else [None] * len(sources)
    assert len(load_plys) == len(sources), "len(load_plys) must equal len(sources)"

    gaussians_list = []
    trainers = []
    for source, dataset, load_ply in zip(sources, datasets, load_plys):
        gaussians = prepare_gaussians(
            decoder=decoder, sh_degree=sh_degree, source=source, dataset=dataset,
            device=device, trainable_camera=trainable_camera,
            load_ply=load_ply, load_semantic=load_semantic,
        )
        trainer = prepare_trainer(
            gaussians=gaussians, dataset=dataset, mode=mode,
            trainable_camera=trainable_camera, configs=configs,
        )
        gaussians_list.append(gaussians)
        trainers.append(trainer)
    return datasets, gaussians_list, trainers
