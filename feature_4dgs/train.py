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
        trainable_camera: bool = False, load_plys: List[str] = None, load_cameras: List[str] = None,
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
        trainable_camera=trainable_camera, load_plys=load_plys, load_semantic=load_semantic,
    )
    trainers = [prepare_trainer(
        gaussians=gaussians, dataset=dataset, mode=mode,
        trainable_camera=trainable_camera, configs=configs,
    ) for gaussians, dataset in zip(gaussians_list, datasets)]
    return datasets, gaussians_list, trainers


def training(dataset: CameraDataset, gaussians: GaussianModel, trainer: AbstractTrainer, destination: str, iteration: int, save_iterations: List[int], empty_cache_every_step=False):
    shutil.rmtree(os.path.join(destination, "point_cloud"), ignore_errors=True)  # remove the previous point cloud
    pbar = tqdm(range(1, iteration+1), dynamic_ncols=True, desc="Training")
    epoch = list(range(len(dataset)))
    epoch_psnr = torch.empty(3, 0)
    epoch_maskpsnr = torch.empty(3, 0)
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    avg_maskpsnr_for_log = 0.0
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            avg_psnr_for_log = epoch_psnr.mean().item()
            avg_maskpsnr_for_log = epoch_maskpsnr.mean().item()
            epoch_psnr = torch.empty(3, 0)
            epoch_maskpsnr = torch.empty(3, 0)
            random.shuffle(epoch)
        idx = epoch[epoch_idx]
        loss, out = trainer.step(dataset[idx])
        with torch.no_grad():
            ground_truth_image = dataset[idx].ground_truth_image
            rendered_image = out["render"].detach()
            epoch_psnr = torch.concat([epoch_psnr, psnr(rendered_image, ground_truth_image).cpu()], dim=1)
            if dataset[idx].ground_truth_image_mask is not None:
                ground_truth_maskimage = ground_truth_image * dataset[idx].ground_truth_image_mask
                rendered_maskimage = rendered_image * dataset[idx].ground_truth_image_mask
                epoch_maskpsnr = torch.concat([epoch_maskpsnr, psnr(rendered_maskimage, ground_truth_maskimage).cpu()], dim=1)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        # Free GPU memory held by the output dict (rasterization intermediates, rendered
        # images, viewspace_points, etc.) before the next forward pass.
        del loss, out
        if empty_cache_every_step:
            torch.cuda.empty_cache()
        if step % 10 == 0:
            postfix = {'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'masked psnr': avg_maskpsnr_for_log, 'n': gaussians._xyz.shape[0]}
            if avg_maskpsnr_for_log <= 0:
                del postfix['masked psnr']
            pbar.set_postfix(postfix)
        if step in save_iterations:
            save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
            os.makedirs(save_path, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
            dataset.save_cameras(os.path.join(destination, "cameras.json"))
    save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    dataset.save_cameras(os.path.join(destination, "cameras.json"))
