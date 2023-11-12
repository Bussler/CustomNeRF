from typing import Tuple

import numpy as np
import torch

from model.RadianceFieldEncoder import RadianceFieldEncoder
from training.early_stopping import EarlyStopping
from training.nerf_inference import nerf_forward
from training.setup_stuff import init_models
from training.utils import crop_center
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rays_rgb_dataset import Ray_Rgb_Dataset
from volume_handling.rendering import Differentiable_Volume_Renderer
from volume_handling.sampling import NeRF_Sampler


def validate_model():
    pass


def training_session(
    device: torch.device,
    data_loader: NeRF_Data_Loader,
    n_iters: int,
    optimizer: torch.optim.Optimizer,
    warmup_stopper: EarlyStopping,
    renderer: Differentiable_Volume_Renderer,
    model: RadianceFieldEncoder,
    nerf_sampler_coarse: NeRF_Sampler,
    fine_model: RadianceFieldEncoder = None,
    nerf_sampler_fine: NeRF_Sampler = None,
    batch_chunksize: int = 2**15,
    one_image_per_step: bool = False,
    center_crop: bool = True,
    center_crop_iters: int = 50,
    warmup_iters: int = 100,
    warmup_min_fitness: float = 10.0,
    display_rate: int = 100,
) -> Tuple[bool, list, list]:
    # M: Gather and shuffle rays across all images.
    one_image_per_step = False  # TODO M: Remove this line
    if not one_image_per_step:
        rays_rgb = data_loader.get_training_rays()
        rays_rgb_dataset = Ray_Rgb_Dataset(rays_rgb)

    i_batch = 0
    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in range(n_iters):
        model.train()

        if one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(data_loader.images.shape[0])
            target_img = torch.from_numpy(data_loader.images[target_img_idx]).to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = torch.from_numpy(data_loader.poses[target_img_idx]).to(device)
            rays_o, rays_d = data_loader.get_rays(height, width, data_loader.focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            rays_o, rays_d, target_img, i_batch = rays_rgb_dataset[i_batch]
            height, width = target_img.shape[:2]  # TODO M: Check if this is correct: make image 2D again? Atm: id 1D
        target_img = target_img.reshape([-1, 3])

        # Forward pass through model.
        outputs = nerf_forward(
            rays_o,
            rays_d,
            data_loader,
            renderer,
            model,
            nerf_sampler_coarse,
            fine_model,
            nerf_sampler_fine,
            batch_chunksize,
        )

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backpropagate loss.
        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        psnr = -10.0 * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Validate model.
        val_psnr = 0.0
        if i % display_rate == 0:
            model.eval()
            with torch.no_grad():
                # val_psnr = validate_model(
                #     device,
                #     data_loader,
                #     renderer,
                #     model,
                #     nerf_sampler_coarse,
                #     fine_model,
                #     nerf_sampler_fine,
                #     batch_chunksize,
                #     center_crop,
                #     center_crop_iters,
                # )
                pass
            val_psnrs.append(val_psnr.item())

        # Stop training if warmup issues with psnr metric
        if i == warmup_iters - 1:
            if val_psnr is not 0.0 and val_psnr < warmup_min_fitness:
                print(f"Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...")
                return False, train_psnrs, val_psnrs
        elif i < warmup_iters:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(f"Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...")
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs


def train(args: dict) -> bool:
    """Perform training of NeRF model.

    Args:
        args (dict): parsed input arguments from experiment file.

    Returns:
        bool: Whether training was successful.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    success = False
    for i in range(args["n_restarts"]):
        print(f"Starting training session {i+1}/{args['n_restarts']}")

        (
            model,
            fine_model,
            data_loader,
            nerf_sampler_coarse,
            nerf_sampler_fine,
            renderer,
            optimizer,
            warmup_stopper,
        ) = init_models(
            device,
            args["data_path"],
            args["near"],
            args["far"],
            args["use_viewdirs"],
            args["d_input"],
            args["n_freqs"],
            args["n_freqs_views"],
            args["log_space"],
            args["n_samples"],
            args["perturb"],
            args["inverse_depth"],
            args["use_fine_model"],
            args["n_layers"],
            args["d_Weights"],
            args["n_layers_fine"],
            args["d_Weights_fine"],
            args["skip"],
            args["lr"],
        )

        success, train_psnrs, val_psnrs = training_session(
            device,
            data_loader,
            args["n_iters"],
            optimizer,
            warmup_stopper,
            renderer,
            model,
            nerf_sampler_coarse,
            fine_model,
            nerf_sampler_fine,
            args["chunksize"],
            args["one_image_per_step"],
            args["center_crop"],
            args["center_crop_iters"],
            args["warmup_iters"],
            args["warmup_min_fitness"],
            args["display_rate"],
        )

        if success and val_psnrs[-1] >= args["warmup_min_fitness"]:
            print("Training successful!")
            # TODO M: Store model params?
            break

    print("Training complete!")
    return success
