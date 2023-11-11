from typing import Tuple

import torch

from model.RadianceFieldEncoder import RadianceFieldEncoder
from training.setup_stuff import init_models
from volume_handling.data_handling import NeRF_Data_Loader


def training_session(
    device: torch.device,
    data_loader: NeRF_Data_Loader,
    n_iters: int,
    model: RadianceFieldEncoder,
    one_image_per_step: bool = False,
) -> Tuple[bool, list, list]:
    # M: move data from loader to device
    n_training = 100  # TODO M: n_training should be a parameter, not a constant
    data_loader.data_to_device(device, n_training)

    # M: Gather and shuffle rays across all images.
    one_image_per_step = False
    if not one_image_per_step:
        height, width = data_loader.train_images.shape[1:3]
        all_rays = torch.stack(
            [
                torch.stack(data_loader.get_rays(height, width, data_loader.focal, p), 0)
                for p in data_loader.poses[:n_training]
            ],
            0,
        )  # M: stack for all training images the rays_o and rays_d; shape [n_training, 2, width, height, 3]
        rays_rgb = torch.cat(
            [all_rays, data_loader.train_images[:, None]], 1
        )  # M: add rgb values to rays; shape [n_training, 3, width, height, 3]
        rays_rgb = torch.permute(
            rays_rgb, [0, 2, 3, 1, 4]
        )  # M: reorder o, d, rgb values to singular rays; reshape to [n_training, width, height, 3, 3]
        rays_rgb = rays_rgb.reshape([-1, 3, 3])  # M: flatten; reshape to [n_training*width*height, 3, 3]
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]  # M: shuffle rays
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in range(n_iters):
        model.train()
        pass

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
        print(f"Restarting training session {i+1}/{args['n_restarts']}")

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
            device, data_loader, args["n_iters"], model, args["one_image_per_step"]
        )
        if success and val_psnrs[-1] >= args["warmup_min_fitness"]:
            print("Training successful!")
            # TODO M: Store model params?
            break

    print("Training complete!")
    return success
