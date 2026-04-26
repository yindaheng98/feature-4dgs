from collections.abc import Iterable, Iterator
from itertools import islice

import torch

from feature_3dgs.vggt.extractor import VGGTExtractor
from feature_3dgs.vggt.track import VGGTrackExtractor
from feature_4dgs.extractor import InherentSequenceFeatureExtractor


class VGGTSequenceExtractor(InherentSequenceFeatureExtractor):
    """VGGT-aware sequence extractor.

    Wraps a :class:`VGGTExtractor` via :class:`InherentSequenceFeatureExtractor`
    and overrides :meth:`extract_sequence_all` to flatten all per-timestep
    images into a single call to :meth:`extract_all`, allowing VGGT's
    multi-view aggregator to see every frame at once.
    """

    def __init__(self, extractor: VGGTExtractor):
        assert isinstance(extractor, VGGTExtractor), "VGGTSequenceExtractor requires a VGGTExtractor instance."
        super().__init__(extractor)

    def extract_sequence_all(self, sequences: Iterable[Iterable[torch.Tensor]]) -> Iterator[Iterator[torch.Tensor]]:
        flatten: list[torch.Tensor] = []
        counts: list[int] = []
        for sequence in sequences:
            sequence = list(sequence)
            counts.append(len(sequence))
            flatten.extend(sequence)
        results = self.extract_all(flatten)
        for count in counts:
            yield islice(results, count)


class VGGTrackSequenceExtractor(VGGTSequenceExtractor):
    """VGGTrack-aware sequence extractor.

    Identical to :class:`VGGTSequenceExtractor` but asserts the wrapped
    extractor is a :class:`VGGTrackExtractor`.
    """

    def __init__(self, extractor: VGGTrackExtractor):
        assert isinstance(extractor, VGGTrackExtractor), "VGGTrackSequenceExtractor requires a VGGTrackExtractor instance."
        super().__init__(extractor)
