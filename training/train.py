from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model.RadianceFieldEncoder import RadianceFieldEncoder
from training.early_stopping import EarlyStopping
from training.nerf_inference import nerf_forward
from training.setup_stuff import init_models
from training.utils import crop_center, plot_samples, write_dict_to_file
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rays_rgb_dataset import Ray_Rgb_Dataset
from volume_handling.rendering import Differentiable_Volume_Renderer
from volume_handling.sampling import NeRF_Ray_Generator, NeRF_Sampler

writer: SummaryWriter = None


def validate_model(
    index: int,
    train_psnrs: list,
    val_psnrs: list,
    val_losses: list,
    iternums: list,
    data_loader: NeRF_Data_Loader,
    renderer: Differentiable_Volume_Renderer,
    model: RadianceFieldEncoder,
    nerf_sampler_coarse: NeRF_Sampler,
    fine_model: RadianceFieldEncoder = None,
    nerf_sampler_fine: NeRF_Sampler = None,
    batch_chunksize: int = 2**15,
    n_samples: int = 64,
    n_samples_hierarchical: int = 64,
    show_graph: bool = True,
) -> Tuple[float, list, list, list]:
    testimg, testpose = data_loader.get_validation_image_pose()

    height, width = testimg.shape[:2]
    rays_o, rays_d = NeRF_Ray_Generator.get_rays(height, width, data_loader.focal, testpose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
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

    rgb_predicted = outputs["rgb_map"]
    loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
    val_losses.append(loss.item())
    print("Validation Loss:", loss.item())
    writer.add_scalar("Loss/val", loss.item(), index)

    val_psnr = -10.0 * torch.log10(loss)
    val_psnrs.append(val_psnr.item())
    print("Validation psnr:", val_psnr.item())
    writer.add_scalar("PSNR/val", val_psnr.item(), index)

    iternums.append(index)

    predicted_img = rgb_predicted.reshape([height, width, 3])
    writer.add_image("Predicted Image", predicted_img.permute(2, 0, 1), index)
    writer.add_image("Target Image", testimg.permute(2, 0, 1), index)

    # Plot example outputs
    if show_graph:
        fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]})
        ax[0].imshow(predicted_img.detach().cpu().numpy())
        ax[0].set_title(f"Prediction Iteration: {index}")
        ax[1].imshow(testimg.detach().cpu().numpy())
        ax[1].set_title(f"Target")
        ax[2].plot(range(0, index + 1), train_psnrs, "r")
        ax[2].plot(iternums, val_psnrs, "b")
        ax[2].set_title("PSNR (train=red, val=blue")
        z_vals_strat = outputs["z_vals_stratified"].view((-1, n_samples))
        z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
        if "z_vals_hierarchical" in outputs:
            z_vals_hierarch = outputs["z_vals_hierarchical"].view((-1, n_samples_hierarchical))
            z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
        else:
            z_sample_hierarch = None
        _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
        ax[3].margins(0)
        plt.show()  # TODO M: Instead of showing, safe to file?

    return val_psnr, val_psnrs, val_losses, iternums


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
    batch_size: int = 2**15,
    batch_chunksize: int = 2**15,
    one_image_per_step: bool = False,
    center_crop: bool = True,
    center_crop_iters: int = 50,
    warmup_iters: int = 100,
    warmup_min_fitness: float = 10.0,
    display_rate: int = 100,
    model_storage_rate: int = 5001,
    out_path: str = "./",
) -> Tuple[bool, list, list, list]:
    # M: Gather and shuffle rays across all images.
    if not one_image_per_step:
        rays_rgb = data_loader.get_training_rays()
        rays_rgb_dataset = Ray_Rgb_Dataset(rays_rgb, batch_size)

    i_batch = 0
    train_psnrs = []
    val_psnrs = []
    val_losses = []
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
            rays_o, rays_d = NeRF_Ray_Generator.get_rays(height, width, data_loader.focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            rays_o, rays_d, target_img = rays_rgb_dataset[i_batch]
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
        writer.add_scalar("Loss/train", loss.item(), i)
        writer.add_scalar("PSNR/train", psnr.item(), i)

        # Validate model.
        val_psnr = 0.0
        if i % display_rate == 0:
            print(f"--- Iteration: {i} / {n_iters} | Train PSNR: {psnr} | Train Loss: {loss} ---")

            model.eval()
            with torch.no_grad():
                val_psnr, val_psnrs, val_losses, iternums = validate_model(
                    i,
                    train_psnrs,
                    val_psnrs,
                    val_losses,
                    iternums,
                    data_loader,
                    renderer,
                    model,
                    nerf_sampler_coarse,
                    fine_model,
                    nerf_sampler_fine,
                    batch_chunksize,
                    show_graph=False,
                )

        if i != 0 and i % model_storage_rate == 0:
            torch.save(model.state_dict(), out_path + "Iter_" + str(i) + "_nerf.pt")
            torch.save(fine_model.state_dict(), out_path + "Iter_" + str(i) + "_nerf-fine.pt")

        # Stop training if warmup issues with psnr metric
        if i == warmup_iters - 1:
            if psnr != 0.0 and psnr < warmup_min_fitness:
                print(f"PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...")
                return False, train_psnrs, val_psnrs, val_losses
        elif i < warmup_iters:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(f"Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...")
                return False, train_psnrs, val_psnrs, val_losses

    return True, train_psnrs, val_psnrs, val_losses


def train(args: dict) -> bool:
    """Perform training of NeRF model.

    Args:
        args (dict): parsed input arguments from experiment file.

    Returns:
        bool: Whether training was successful.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    global writer
    if args["tensorboard_log_dir"]:
        writer = SummaryWriter(args["tensorboard_log_dir"] + args["expname"])
    else:
        writer = SummaryWriter("runs/" + args["expname"])

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
            args["n_training"],
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

        success, train_psnrs, val_psnrs, val_losses = training_session(
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
            args["batch_size"],
            args["chunksize"],
            args["one_image_per_step"],
            args["center_crop"],
            args["center_crop_iters"],
            args["warmup_iters"],
            args["warmup_min_fitness"],
            args["display_rate"],
            args["model_storage_rate"],
            args["out_path"],
        )

        if success and val_psnrs[-1] >= args["warmup_min_fitness"]:
            print("Training successful!")
            write_dict_to_file(
                args["out_path"] + "train_results.txt",
                {"train_psnr": str(train_psnrs[-1]), "val_psnr": str(val_psnrs[-1]), "val_loss": str(val_losses[-1])},
            )

            # M: Store model params
            torch.save(model.state_dict(), args["out_path"] + "nerf.pt")
            torch.save(fine_model.state_dict(), args["out_path"] + "nerf-fine.pt")
            break

    print("Training complete!")
    return success
