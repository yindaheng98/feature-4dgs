from typing import Tuple

from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.vggt.vggt import build_factory as build_factory_inherent
from feature_3dgs.vggt.vggt import build_track_factory as build_track_factory_inherent

from feature_4dgs.extractor import AbstractSequenceFeatureExtractor
from feature_4dgs.registry import register_extractor_decoder

from .extractor import VGGTSequenceExtractor, VGGTrackSequenceExtractor

MODEL_VGGT = "vggt"
MODEL_VGGTRACK = "vggtrack"


def build_factory():
    factory_inherent = build_factory_inherent()

    def factory(embed_dim: int, **configs,) -> Tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]:
        extractor, decoder = factory_inherent(embed_dim, **configs)
        return VGGTSequenceExtractor(extractor), decoder

    return factory


def build_track_factory():
    factory_inherent = build_track_factory_inherent()

    def factory(embed_dim: int, **configs,) -> Tuple[AbstractSequenceFeatureExtractor, AbstractTrainableDecoder]:
        extractor, decoder = factory_inherent(embed_dim, **configs)
        return VGGTrackSequenceExtractor(extractor), decoder

    return factory


register_extractor_decoder(MODEL_VGGT, build_factory())
register_extractor_decoder(MODEL_VGGTRACK, build_track_factory())
