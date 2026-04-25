from typing import List

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset


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
