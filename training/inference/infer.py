import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from training.inference.render_path_generator import generate_circular_renderpath
from training.nerf_inference import nerf_forward
from training.setup_stuff import load_model
from volume_handling.sampling import NeRF_Ray_Generator


def infer(args: dict) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model, fine_model, data_loader, nerf_sampler_coarse, nerf_sampler_fine, renderer = load_model(
        device,
        args["data_path"],
        args["model_path"],
        args["fine_model_path"],
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
    )

    # TODO M: create poses
    testimg, testpose = (
        data_loader.train_images[0:1].squeeze(0),
        data_loader.train_poses[0:1].squeeze(0),
    )
    render_path = generate_circular_renderpath(
        data_loader.train_poses[0:1].cpu(), data_loader.focal.cpu(), 120, 2, 0.1, 0.1
    )
    images = []

    model.eval()
    with torch.no_grad():
        # infer model: go over all poses from render path and generate images
        for pose in tqdm(render_path):
            testpose = pose.to(device)

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
                args["chunksize"],
            )

            rgb_predicted = outputs["rgb_map"]
            predicted_img = rgb_predicted.reshape([height, width, 3])
            predicted_img = (predicted_img.detach().cpu().numpy() * 255).astype(
                np.uint8
            )  # Convert from float in [0, 1] to uint in [0, 255]
            images.append(predicted_img)

    # turn images into video/ gif
    imageio.mimsave("output.gif", images, fps=30)

    return True
