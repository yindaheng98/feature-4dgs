from typing import Sequence

import torch

from .prepare import prepare_datasets_and_decoder


def save_heatmaps(datasets, sources: Sequence[str], time: int, view: int, width_idx: int, height_idx: int, destination: str):
    pass


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
        save_heatmaps(datasets, args.sources, args.time, args.view, args.width_idx, args.height_idx, args.destination)
