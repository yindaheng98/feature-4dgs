from collections.abc import Iterable, Iterator

import torch

from feature_3dgs.extractor import AbstractFeatureExtractor


class AbstractSequenceFeatureExtractor(AbstractFeatureExtractor):
    """A feature extractor that can also process a sequence of image streams.

    A "sequence" here is an iterable of per-timestep iterables of images.
    The default :meth:`extract_sequence_all` simply dispatches to
    :meth:`extract_all` for each timestep; subclasses may override it to
    perform cross-timestep batching, caching or scheduling.
    """

    def extract_sequence_all(self, sequences: Iterable[Iterable[torch.Tensor]]) -> Iterator[Iterator[torch.Tensor]]:
        for sequence in sequences:
            yield self.extract_all(sequence)
