import os
import random
import shutil
from typing import List, Tuple

import torch
from tqdm import tqdm

from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.train import save_cfg_args
from gaussian_splatting.utils import psnr
from feature_3dgs import SemanticGaussianModel
from feature_3dgs.prepare import prepare_trainer, modes

from .extractor import SequenceFeatureCameraDataset
from .prepare import prepare_datasets_and_decoder, prepare_gaussians_sequence


def prepare_training(
        name: str, sh_degree: int, mode: str, sources: List[str], embed_dim: int,
        device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_plys: List[str] = None, load_decoder: str = None, load_cameras: List[str] = None,
        load_mask=True, load_depth=True, load_semantic: bool = True,
        preload_cache: bool = True, configs={}, extractor_configs={},
) -> Tuple[SequenceFeatureCameraDataset, List[SemanticGaussianModel], List[AbstractTrainer]]:
    """Prepare a sequence dataset, per-timestep gaussians and trainers.

    All timesteps share a single decoder (owned by the extractor/decoder
    factory) that is attached to every :class:`SemanticGaussianModel`.
    """
    datasets, decoder = prepare_datasets_and_decoder(
        name=name, sources=sources, embed_dim=embed_dim, device=device, dataset_cache_device=dataset_cache_device,
        trainable_camera=trainable_camera, load_cameras=load_cameras,
        load_mask=load_mask, load_depth=load_depth, preload_cache=preload_cache, configs=extractor_configs,
    )
    gaussians_list = prepare_gaussians_sequence(
        decoder=decoder, sh_degree=sh_degree, sources=sources, datasets=datasets, device=device,
        trainable_camera=trainable_camera, load_plys=load_plys, load_semantic=load_semantic, load_decoder=load_decoder,)
    trainers = [prepare_trainer(
        gaussians=gaussians, dataset=dataset, mode=mode,
        trainable_camera=trainable_camera, configs=configs,
    ) for gaussians, dataset in zip(gaussians_list, datasets)]
    return datasets, gaussians_list, trainers


def training(datasets: SequenceFeatureCameraDataset, gaussians_list: List[SemanticGaussianModel], trainers: List[AbstractTrainer], destinations: List[str], iteration: int, save_iterations: List[int], empty_cache_every_step=False):
    assert len(datasets) == len(gaussians_list) == len(trainers) == len(destinations), \
        "datasets, gaussians_list, trainers and destinations must have the same length (number of timesteps)"
    for destination in destinations:
        shutil.rmtree(os.path.join(destination, "point_cloud"), ignore_errors=True)
    pbar = tqdm(range(1, iteration+1), dynamic_ncols=True, desc="Training")
    epoch = [list(range(len(dataset))) for dataset in datasets]
    epoch_psnr = [torch.empty(3, 0) for _ in range(len(datasets))]
    epoch_maskpsnr = [torch.empty(3, 0) for _ in range(len(datasets))]
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    avg_maskpsnr_for_log = 0.0
    avg_psnrs = [None] * len(datasets)
    avg_maskpsnrs = [None] * len(datasets)
    for step in pbar:
        step_loss_sum = 0.0
        for i in random.sample(range(len(datasets)), len(datasets)):
            dataset, trainer = datasets[i], trainers[i]
            epoch_idx = step % len(dataset)
            if epoch_idx == 0:
                avg_psnrs[i] = epoch_psnr[i].mean().item() if epoch_psnr[i].numel() > 0 else None
                avg_maskpsnrs[i] = epoch_maskpsnr[i].mean().item() if epoch_maskpsnr[i].numel() > 0 else None
                valid_psnrs = [v for v in avg_psnrs if v is not None]
                valid_maskpsnrs = [v for v in avg_maskpsnrs if v is not None]
                avg_psnr_for_log = sum(valid_psnrs) / len(valid_psnrs) if valid_psnrs else 0.0
                avg_maskpsnr_for_log = sum(valid_maskpsnrs) / len(valid_maskpsnrs) if valid_maskpsnrs else 0.0
                epoch_psnr[i] = torch.empty(3, 0)
                epoch_maskpsnr[i] = torch.empty(3, 0)
                random.shuffle(epoch[i])
            idx = epoch[i][epoch_idx]
            loss, out = trainer.step(dataset[idx])
            with torch.no_grad():
                ground_truth_image = dataset[idx].ground_truth_image
                rendered_image = out["render"].detach()
                epoch_psnr[i] = torch.concat([epoch_psnr[i], psnr(rendered_image, ground_truth_image).cpu()], dim=1)
                if dataset[idx].ground_truth_image_mask is not None:
                    ground_truth_maskimage = ground_truth_image * dataset[idx].ground_truth_image_mask
                    rendered_maskimage = rendered_image * dataset[idx].ground_truth_image_mask
                    epoch_maskpsnr[i] = torch.cat([epoch_maskpsnr[i], psnr(rendered_maskimage, ground_truth_maskimage).cpu()], dim=1)
                step_loss_sum += loss.item()
            del loss, out
        ema_loss_for_log = 0.4 * (step_loss_sum / len(datasets)) + 0.6 * ema_loss_for_log
        if empty_cache_every_step:
            torch.cuda.empty_cache()
        if step % 10 == 0:
            postfix = {'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'masked psnr': avg_maskpsnr_for_log, 'n': sum(g._xyz.shape[0] for g in gaussians_list)}
            if avg_maskpsnr_for_log <= 0:
                del postfix['masked psnr']
            pbar.set_postfix(postfix)
        if step in save_iterations:
            for destination, gaussians, dataset in zip(destinations, gaussians_list, datasets):
                save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
                os.makedirs(save_path, exist_ok=True)
                gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
                dataset.save_cameras(os.path.join(destination, "cameras.json"))
    for destination, gaussians, dataset in zip(destinations, gaussians_list, datasets):
        save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
        os.makedirs(save_path, exist_ok=True)
        gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
        dataset.save_cameras(os.path.join(destination, "cameras.json"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--embed_dim", required=True, type=int)
    parser.add_argument("-s", "--sources", required=True, nargs='+', type=str)
    parser.add_argument("-d", "--destinations", required=True, nargs='+', type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("-l", "--load_plys", default=None, nargs='+', type=str)
    parser.add_argument("--load_decoder", default=None, type=str)
    parser.add_argument("--load_cameras", default=None, nargs='+', type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--no_load_semantic", action="store_true")
    parser.add_argument("--mode", choices=sorted(modes.keys()), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("--no_preload_dataset_cache", action="store_true")
    parser.add_argument("--empty_cache_every_step", action='store_true')
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("-e", "--option_extractor", default=[], action='append', type=str)
    args = parser.parse_args()
    assert len(args.destinations) == len(args.sources), "len(destinations) must equal len(sources)"
    if args.load_plys is not None:
        assert len(args.load_plys) == len(args.sources), "len(load_plys) must equal len(sources)"
    if args.load_cameras is not None:
        assert len(args.load_cameras) == len(args.sources), "len(load_cameras) must equal len(sources)"
    for destination, source in zip(args.destinations, args.sources):
        save_cfg_args(destination, args.sh_degree, source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    datasets, gaussians_list, trainers = prepare_training(
        name=args.name, sh_degree=args.sh_degree, mode=args.mode,
        sources=args.sources, embed_dim=args.embed_dim,
        device=args.device, dataset_cache_device=args.dataset_cache_device,
        trainable_camera="camera" in args.mode,
        load_plys=args.load_plys, load_decoder=args.load_decoder, load_cameras=args.load_cameras,
        load_mask=not args.no_image_mask, load_depth=not args.no_depth_data, load_semantic=not args.no_load_semantic,
        preload_cache=not args.no_preload_dataset_cache, configs=configs, extractor_configs=extractor_configs)
    for dataset in datasets:
        dataset.save_cameras(os.path.join(args.destinations[0], "cameras.json"))
    torch.cuda.empty_cache()
    training(
        datasets=datasets, gaussians_list=gaussians_list, trainers=trainers,
        destinations=args.destinations, iteration=args.iteration, save_iterations=args.save_iterations,
        empty_cache_every_step=args.empty_cache_every_step)
