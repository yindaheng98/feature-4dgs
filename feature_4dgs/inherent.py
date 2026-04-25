"""Re-register every extractor-decoder from ``feature_3dgs`` into the
``feature_4dgs`` registry by wrapping each extractor with
:class:`ElementWiseSequenceFeatureExtractor`.

Names are suffixed with ``-elementwise`` to distinguish them from potential
native sequence implementations.
"""

from typing import Tuple

import feature_3dgs  # noqa: F401  — triggers feature_3dgs's own registrations
from feature_3dgs.registry import ExtractorDecoderFactory, REGISTRY
from feature_3dgs.decoder import AbstractTrainableDecoder

from .extractor import AbstractSequenceFeatureExtractor, ElementWiseSequenceFeatureExtractor
from .registry import register_extractor_decoder


class ElementWiseSequenceExtractorDecoderFactory:
    """Wraps a ``feature_3dgs`` :class:`ExtractorDecoderFactory` so that
    calling it returns an :class:`ElementWiseSequenceFeatureExtractor`
    paired with the original decoder.
    """

    def __init__(self, factory: ExtractorDecoderFactory):
        self.factory = factory

    def __call__(self, embed_dim: int, *args, **kwargs) -> Tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]:
        extractor, decoder = self.factory(embed_dim, *args, **kwargs)
        return ElementWiseSequenceFeatureExtractor(extractor), decoder


for name, factory in REGISTRY.items():
    register_extractor_decoder(f"{name}-elementwise", ElementWiseSequenceExtractorDecoderFactory(factory))
