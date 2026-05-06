from collections.abc import Iterable, Iterator

import torch

from gaussian_splatting.dataset import CameraDataset
from feature_3dgs.extractor import AbstractFeatureExtractor

from .extractor import AbstractSequenceFeatureExtractor
from .dataset import SequenceFeatureCameraDataset


class InherentSequenceFeatureExtractor(AbstractSequenceFeatureExtractor):
    """Lift any :class:`AbstractFeatureExtractor` into the
    :class:`AbstractSequenceFeatureExtractor` interface by simply forwarding
    ``__call__``, ``to`` and ``extract_all`` to the wrapped extractor.

    The default :meth:`extract_sequence_all` from the base class then calls
    ``extract_all`` once per timestep.
    """

    def __init__(self, extractor: AbstractFeatureExtractor):
        self.extractor = extractor

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.extractor(image)

    def to(self, device) -> 'InherentSequenceFeatureExtractor':
        self.extractor = self.extractor.to(device)
        return self

    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        return self.extractor.extract_all(images)


class InherentSequenceFeatureCameraDataset(SequenceFeatureCameraDataset):
    """Convenience subclass of :class:`SequenceFeatureCameraDataset` that
    accepts a plain :class:`AbstractFeatureExtractor` and wraps it with a
    :class:`InherentSequenceFeatureExtractor` before constructing the sequence.
    """

    def __init__(self, cameras_list: Iterable[CameraDataset], extractor: AbstractFeatureExtractor, cache_device=None):
        super().__init__(cameras_list=cameras_list, extractor=InherentSequenceFeatureExtractor(extractor), cache_device=cache_device)
