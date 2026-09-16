import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import viridis
from tqdm import tqdm

from gaussian_splatting import Camera
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.segmentation2d import get_feature

from .prepare import prepare_datasets_and_decoder
from .extractor import SequenceFeatureCameraDataset

DPI = 150


def save_heatmaps(
    datasets: SequenceFeatureCameraDataset, query: torch.Tensor, decoder: AbstractTrainableDecoder,
    destination: str, time: int, view: int, width_idx: int, height_idx: int, upsample: str,
):
    os.makedirs(destination, exist_ok=True)
    pbar = tqdm(total=sum(len(dataset) for dataset in datasets), desc="Saving heatmaps", dynamic_ncols=True)
    for t, dataset in enumerate(datasets):
        for i in range(len(dataset)):
            camera: Camera = dataset[i]
            image = camera.ground_truth_image.clamp(0, 1)
            H, W = image.shape[1], image.shape[2]
            fmap = camera.custom_data["feature_map"]
            if upsample == "feature":
                encoded = decoder.encode_feature_map(fmap, camera)
                sim = decoder.similarity_encoded_features(query.to(encoded.device), encoded.permute(1, 2, 0))
            else:
                sim = decoder.similarity(query.to(fmap.device), fmap.permute(1, 2, 0))
                sim = F.interpolate(sim[None, None], size=(H, W), mode="bilinear", align_corners=True)[0, 0]
            Hf, Wf = sim.shape
            pk = sim.reshape(-1).argmax().item()
            pi, pj = divmod(pk, Wf)
            is_query = t == time and i == view
            img_xy = (width_idx, height_idx) if is_query else (
                pj * (W - 1) / max(Wf - 1, 1),
                pi * (H - 1) / max(Hf - 1, 1),
            )
            feat_xy = (
                width_idx * (Wf - 1) / max(W - 1, 1),
                height_idx * (Hf - 1) / max(H - 1, 1),
            ) if is_query else (pj, pi)
            rgb = image.detach().permute(1, 2, 0).cpu().numpy()
            sim = (sim - sim.min()) / (sim.max() - sim.min()).clamp_min(1e-8)
            heatmap = viridis(sim.detach().cpu().numpy())[..., :3]
            for img, name, xy in (
                (rgb, f"t{t}_v{i}.png", img_xy),
                (heatmap, f"t{t}_v{i}_heatmap.png", feat_xy),
            ):
                h, w = img.shape[:2]
                fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(np.clip(img, 0, 1))
                if is_query:
                    ax.plot(*xy, "r+", markersize=16, markeredgewidth=2)
                else:
                    ax.plot(*xy, "c*", markersize=12)
                ax.set_axis_off()
                fig.savefig(os.path.join(destination, name), dpi=DPI, pad_inches=0)
                plt.close(fig)
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--help", action="help")
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--encoded_dim", required=True, type=int)
    parser.add_argument("-s", "--sources", required=True, nargs="+", type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("--no_preload_dataset_cache", action="store_true")
    parser.add_argument("-e", "--option_extractor", default=[], action="append", type=str)
    parser.add_argument("-t", "--time", required=True, type=int, help="Query timestep index into --sources (0-based).")
    parser.add_argument("-v", "--view", required=True, type=int, help="Query camera index within that timestep (0-based).")
    parser.add_argument("-w", "--width_idx", required=True, type=int)
    parser.add_argument("-h", "--height_idx", required=True, type=int)
    parser.add_argument(
        "--upsample", choices=["feature", "similarity"], default="similarity",
        help="feature: encode_feature_map then similarity; similarity: similarity then bilinear resize to image size.",
    )
    args = parser.parse_args()

    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    datasets, decoder = prepare_datasets_and_decoder(
        name=args.name, sources=args.sources, encoded_dim=args.encoded_dim,
        device=args.device, dataset_cache_device=args.dataset_cache_device,
        trainable_camera=False, load_mask=False, load_depth=False,
        preload_cache=not args.no_preload_dataset_cache, configs=extractor_configs,
    )
    decoder.init_semantic(datasets[0])
    decoder.to(args.device).eval()
    with torch.no_grad():
        query = get_feature(datasets[args.time], args.view, args.width_idx, args.height_idx)
        save_heatmaps(
            datasets, query, decoder, args.destination,
            args.time, args.view, args.width_idx, args.height_idx, args.upsample,
        )
