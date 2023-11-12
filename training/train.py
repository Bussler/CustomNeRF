from typing import Tuple

import numpy as np
import torch

from model.RadianceFieldEncoder import RadianceFieldEncoder
from training.setup_stuff import init_models
from training.utils import crop_center
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rays_rgb_dataset import Ray_Rgb_Dataset


def training_session(
    device: torch.device,
    data_loader: NeRF_Data_Loader,
    n_iters: int,
    model: RadianceFieldEncoder,
    one_image_per_step: bool = False,
    center_crop: bool = True,
    center_crop_iters: int = 50,
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

        # Backpropagate loss.

        # Validate model.

        # Stop training if warmup issues

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
            model,
            args["one_image_per_step"],
            args["center_crop"],
            args["center_crop_iters"],
        )
        if success and val_psnrs[-1] >= args["warmup_min_fitness"]:
            print("Training successful!")
            # TODO M: Store model params?
            break

    print("Training complete!")
    return success
