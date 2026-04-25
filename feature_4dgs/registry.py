from typing import Tuple, Protocol

from feature_3dgs.decoder import AbstractTrainableDecoder
from .extractor import AbstractSequenceFeatureExtractor


class SequenceExtractorDecoderFactory(Protocol):
    def __call__(self, embed_dim: int, *args: object, **kwargs: object) -> tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]: ...


REGISTRY: dict[str, SequenceExtractorDecoderFactory] = {}


def register_extractor_decoder(name: str, factory: SequenceExtractorDecoderFactory) -> None:
    """Register an (AbstractSequenceFeatureExtractor, AbstractTrainableDecoder) factory under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Sequence extractor-decoder combination '{name}' is already registered.")
    REGISTRY[name] = factory


def get_available_extractor_decoders() -> list[str]:
    """Return the names of all registered sequence extractor-decoder combinations."""
    return list(REGISTRY.keys())


def build_extractor_decoder(name: str, embed_dim: int, **configs) -> Tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]:
    """Build an (AbstractSequenceFeatureExtractor, AbstractTrainableDecoder) pair by name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Sequence extractor-decoder combination '{name}' not found. "
            f"Available: {get_available_extractor_decoders()}"
        )
    return REGISTRY[name](embed_dim, **configs)
