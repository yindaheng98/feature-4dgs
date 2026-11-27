from collections.abc import Iterable, Iterator

import contextlib

import torch
import torch.nn.functional as F
from uniception.models.encoders import ViTEncoderInput
from uniception.models.info_sharing.base import MultiViewTransformerInput

from feature_3dgs.uniception import UniCeptionExtractor
from feature_4dgs.extractor import AbstractSequenceFeatureExtractor


class UFMSequenceExtractor(AbstractSequenceFeatureExtractor):
    """UFM info-sharing features for exactly two frames.

    ``UniCeptionExtractor`` is the single-image encoder. Each view of frame 0
    is paired with the same-index view of frame 1 and passed through
    ``info_sharing``. Both frames must contain the same number of views.

    Each view's map is the four tensors one info-sharing pass feeds to the DPT
    head, concatenated on the channel axis: the encoder feature, the two
    intermediate maps, and the final map.
    """

    require_preload = True

    def __init__(self, encoder: UniCeptionExtractor, info_sharing: torch.nn.Module):
        if encoder.feature_dim != int(info_sharing.input_embed_dim):
            raise ValueError(
                f"Encoder feature dim {encoder.feature_dim} does not match "
                f"info sharing input dim {info_sharing.input_embed_dim}"
            )
        self.encoder = encoder
        self.info_sharing = info_sharing
        self.info_sharing.eval()
        self.patch_size = encoder.patch_size
        self.resolutions = encoder.resolutions
        self._feature_dim = int(encoder.feature_dim) + 3 * int(info_sharing.dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("UFM info sharing requires exactly two frames. Use extract_sequence_all().")

    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        raise NotImplementedError("UFM info sharing requires exactly two frames. Use extract_sequence_all().")

    @torch.no_grad()
    def encode_image_pairs(self, image0: torch.Tensor, image1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both images in one batch, then one info-sharing pass.

        Same order as ``UniFlowMatch.forward``: ``_encode_image_pairs`` concatenates
        the pair, and the DPT list is the encoder map, two intermediate maps, and
        the final map. Channels are concatenated and the batch dimension is removed.
        """
        prepared0 = self.encoder.prepare(image0)
        prepared1 = self.encoder.prepare(image1)
        if prepared1.shape[-2:] != prepared0.shape[-2:]:
            prepared1 = F.interpolate(prepared1, size=prepared0.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        prepared = torch.cat((prepared0, prepared1), dim=0)
        if prepared.is_cuda:
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context = contextlib.nullcontext()
        with context:
            layers = self.encoder.encoder(ViTEncoderInput(
                image=prepared,
                data_norm_type=self.encoder.encoder.data_norm_type,
            ))
            feat0, feat1 = layers[-1].features.chunk(2, dim=0)
            final, intermediate = self.info_sharing(MultiViewTransformerInput(features=[feat0, feat1]))

        def pack(view: int, encoded: torch.Tensor) -> torch.Tensor:
            maps = (
                encoded,
                intermediate[0].features[view],
                intermediate[1].features[view],
                final.features[view],
            )
            return torch.cat([item[0].float() for item in maps], dim=0).contiguous()

        return pack(0, feat0), pack(1, feat1)

    @torch.no_grad()
    def extract_sequence_all(self, sequences: Iterable[Iterable[torch.Tensor]]) -> Iterator[Iterator[torch.Tensor]]:
        frames = [list(sequence) for sequence in sequences]
        if len(frames) != 2:
            raise ValueError(f"UFM info sharing requires exactly two frames, got {len(frames)}")
        frame0, frame1 = frames
        if len(frame0) != len(frame1):
            raise ValueError(
                f"UFM info sharing requires the same number of views in both frames, got {len(frame0)} and {len(frame1)}"
            )
        features0 = []
        features1 = []
        for image0, image1 in zip(frame0, frame1):
            shared0, shared1 = self.encode_image_pairs(image0, image1)
            features0.append(shared0)
            features1.append(shared1)
        yield iter(features0)
        yield iter(features1)

    def to(self, device) -> "UFMSequenceExtractor":
        self.encoder = self.encoder.to(device)
        self.info_sharing = self.info_sharing.to(device)
        self.info_sharing.eval()
        return self
