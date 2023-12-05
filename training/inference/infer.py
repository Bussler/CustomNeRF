import torch

from training.nerf_inference import nerf_forward
from training.setup_stuff import load_model


def infer(args: dict) -> bool:
    # TODO M: create poses

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model, fine_model, data_loader, nerf_sampler_coarse, nerf_sampler_fine, renderer = load_model(
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
    )

    # infer model: go over all poses from render path and generate images
    testimg, testpose = data_loader.get_validation_image_pose()  # TODO M: get later from the generated paths

    height, width = testimg.shape[:2]
    rays_o, rays_d = data_loader.get_rays(height, width, data_loader.focal, testpose)
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

    # turn images into video/ gif

    return True
