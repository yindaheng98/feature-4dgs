from .extractor import AbstractSequenceFeatureExtractor, SequenceFeatureCameraDataset
from .extractor import InherentSequenceFeatureExtractor, InherentSequenceFeatureCameraDataset
from .registry import register_extractor_decoder, get_available_extractor_decoders, build_extractor_decoder
from . import inherent  # noqa: F401  — registers feature_3dgs extractors as inherent
from . import vggt  # noqa: F401  — registers feature_3dgs extractors as inherent
