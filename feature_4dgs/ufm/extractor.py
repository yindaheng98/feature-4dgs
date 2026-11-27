from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from uniception.models.info_sharing.base import MultiViewTransformerInput

from feature_3dgs.uniception import UniCeptionExtractor
from feature_4dgs.extractor import AbstractSequenceFeatureExtractor


class UFMSequenceExtractor(AbstractSequenceFeatureExtractor):
    """UFM info-sharing features for exactly two frames.

    ``UniCeptionExtractor`` is the single-image encoder. Each view of frame 0
    is paired with the same-index view of frame 1 and passed through
    ``info_sharing``. Both frames must contain the same number of views.

    The yielded map is the final info-sharing feature of that view,
    ``(dim, H / patch, W / patch)``. ``dim`` is 768 for the ViT-L UFM
    checkpoints, 1024 for DINOv2-L init, and 1536 for DINOv2-G init.
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
        self._feature_dim = int(info_sharing.dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("UFM info sharing requires exactly two frames. Use extract_sequence_all().")

    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        raise NotImplementedError("UFM info sharing requires exactly two frames. Use extract_sequence_all().")

    @torch.no_grad()
    def encode_image_pairs(self, image0: torch.Tensor, image1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two images and pass the features through info sharing.

        ``uniflowmatch/models/ufm.py`` ``UniFlowMatch.forward``: ``_encode_image_pairs``
        then ``info_sharing`` on the last encoder maps. Returns the two views of
        ``final_info_sharing_multi_view_feat``.
        """
        # UniCeptionExtractor is feat_list[-1] from _encode_image_pairs, with the batch dim removed.
        feat0 = self.encoder(image0).unsqueeze(0)
        feat1 = self.encoder(image1).unsqueeze(0)
        # _encode_image_pairs raises "Unequal Image sizes are not supported now".
        # Resize view 2 onto view 1 so the pair still runs.
        if feat1.shape[-2:] != feat0.shape[-2:]:
            feat1 = F.interpolate(feat1, size=feat0.shape[-2:], mode="bilinear", align_corners=False)
        # feature_3dgs/vggt/extractor.py VGGTExtractor.run_aggregator. Autocast follows the feature device.
        device = feat0.device
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
        with torch.amp.autocast(device.type, dtype=dtype):
            # UniFlowMatch.forward: Pass the features through the info_sharing.
            final, _intermediate = self.info_sharing(MultiViewTransformerInput(features=[feat0, feat1]))
        # info_sharing_outputs["1"][-1] and ["2"][-1].
        return (
            final.features[0][0].float().contiguous(),
            final.features[1][0].float().contiguous(),
        )

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
