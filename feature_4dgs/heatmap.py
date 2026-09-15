import os
from typing import Sequence

import torch
import torch.nn.functional as F
from matplotlib.cm import viridis
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_splatting import Camera
from feature_3dgs.segmentation2d import get_feature

from .prepare import prepare_datasets_and_decoder


def save_heatmaps(datasets, sources: Sequence[str], query: torch.Tensor, destination: str):
    pbar = tqdm(total=sum(len(dataset) for dataset in datasets), desc="Saving heatmaps", dynamic_ncols=True)
    for t, dataset in enumerate(datasets):
        frame = os.path.basename(os.path.normpath(sources[t]))
        for i in range(len(dataset)):
            camera: Camera = dataset[i]
            view_dir = os.path.join(destination, str(i))
            os.makedirs(view_dir, exist_ok=True)
            image = camera.ground_truth_image.clamp(0, 1)
            fmap = camera.custom_data["feature_map"]
            sim = F.cosine_similarity(query.to(fmap.device).reshape(-1, 1, 1), fmap, dim=0)
            if sim.shape != image.shape[1:]:
                sim = F.interpolate(sim[None, None], size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            heatmap = torch.from_numpy(viridis(((sim.clamp(-1, 1) + 1) / 2).detach().cpu().numpy())[..., :3]).permute(2, 0, 1).float()
            save_image(image, os.path.join(view_dir, f"{frame}.png"))
            save_image(heatmap, os.path.join(view_dir, f"{frame}_heatmap.png"))
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
    args = parser.parse_args()

    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    datasets, decoder = prepare_datasets_and_decoder(
        name=args.name, sources=args.sources, encoded_dim=args.encoded_dim,
        device=args.device, dataset_cache_device=args.dataset_cache_device,
        trainable_camera=False, load_mask=False, load_depth=False,
        preload_cache=not args.no_preload_dataset_cache, configs=extractor_configs,
    )
    del decoder
    with torch.no_grad():
        query = get_feature(datasets[args.time], args.view, args.width_idx, args.height_idx)
        save_heatmaps(datasets, args.sources, query, args.destination)
