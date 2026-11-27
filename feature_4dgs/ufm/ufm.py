from typing import Tuple

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.uniception import VARIANTS, UniCeptionFeatureExtractor, UniCeptionLinearAvgDecoder
from feature_4dgs.extractor import AbstractSequenceFeatureExtractor
from feature_4dgs.registry import register_extractor_decoder

from .extractor import UFMSequenceExtractor


def load_uniflowmatch(model: str = "ufm-base"):
    """Load a pretrained UniFlowMatch checkpoint.

    ``model`` uses the same names as ``feature_3dgs.uniception.VARIANTS``.
    """
    if model not in VARIANTS:
        raise ValueError(f"Please choose from {list(VARIANTS)}")
    if "base" in model:
        loaded = UniFlowMatchConfidence.from_pretrained(VARIANTS[model])
    elif "refine" in model:
        loaded = UniFlowMatchClassificationRefinement.from_pretrained(VARIANTS[model])
    else:
        raise ValueError(f"Please choose from {list(VARIANTS)}")
    loaded.eval()
    return loaded


def build_factory(variant: str):
    def factory(encoded_dim: int, **configs) -> Tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]:
        del configs
        encoder = UniCeptionFeatureExtractor(variant)
        extractor = UFMSequenceExtractor(encoder, load_uniflowmatch(variant).info_sharing)
        decoder = UniCeptionLinearAvgDecoder(
            in_channels=encoded_dim,
            out_channels=extractor.feature_dim,
            patch_size=extractor.patch_size,
            resolutions=extractor.resolutions,
        )
        return extractor, decoder

    return factory


for variant in VARIANTS:
    register_extractor_decoder(variant, build_factory(variant))
