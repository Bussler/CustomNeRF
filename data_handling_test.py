from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.feature_embedding import PositionalEmbedding
from model.NeRF import NeRF
from training.nerf_inference import nerf_forward
from training.setup_stuff import init_models
from training.utils import dict_from_file
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rendering import Differentiable_Volume_Renderer
from volume_handling.sampling import NeRF_Hierarchical_Sampler, NeRF_Stratified_Sampler

nerf_data_loader = NeRF_Data_Loader("data/tiny_nerf_data.npz")
device = torch.device("cpu")  # can also try "cuda"
testimg_idx = 101


def debug_testimg_show(show=True) -> Tuple[List, List]:
    testimg = nerf_data_loader.images[testimg_idx]
    testpose = nerf_data_loader.poses[testimg_idx]
    imgplot = plt.imshow(testimg)
    if show:
        plt.show()
    print("Pose")
    print(testpose)
    return testimg, testpose


def debug_cam_directions_origins(show=True) -> Tuple[List, List]:
    cam_origins = nerf_data_loader.cam_origins
    cam_dirs = nerf_data_loader.cam_dirs

    if show:
        ax = plt.figure(figsize=(12, 8)).add_subplot(projection="3d")
        _ = ax.quiver(
            cam_origins[..., 0].flatten(),
            cam_origins[..., 1].flatten(),
            cam_origins[..., 2].flatten(),
            cam_dirs[..., 0].flatten(),
            cam_dirs[..., 1].flatten(),
            cam_dirs[..., 2].flatten(),
            length=0.5,
            normalize=True,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("z")
        plt.show()
    return cam_origins, cam_dirs


def debug_rays_generation(
    show=True,
) -> Tuple[torch.Size, torch.Tensor, torch.Size, torch.Tensor, torch.Size, torch.Size]:
    """Testfunction. Generate Ray origin positions and direction vector for a single testimage.
    Then generate sample points along the rays.
    """
    testimg, testpose = debug_testimg_show(show)
    testimg = torch.from_numpy(testimg).to(device)
    testpose = torch.from_numpy(testpose).to(device)

    with torch.no_grad():
        ray_origin, ray_direction = nerf_data_loader.get_rays(
            nerf_data_loader.height, nerf_data_loader.width, nerf_data_loader.focal, testpose
        )

    print("Ray Origin")
    print(ray_origin.shape)
    ray_slice = ray_origin[nerf_data_loader.height // 2, nerf_data_loader.width // 2, :]
    print(ray_slice)
    print("")

    print("Ray Direction")
    print(ray_direction.shape)
    raydirection_slice = ray_direction[nerf_data_loader.height // 2, nerf_data_loader.width // 2, :]
    print(raydirection_slice)
    print("")

    # Draw stratified samples from example
    rays_o = ray_origin.view([-1, 3])
    rays_d = ray_direction.view([-1, 3])
    n_samples = 8
    perturb = True
    inverse_depth = False

    nerf_sampler = NeRF_Stratified_Sampler(
        near=nerf_data_loader.near,
        far=nerf_data_loader.far,
        n_samples=n_samples,
        perturb=perturb,
        inverse_depth=inverse_depth,
    )

    with torch.no_grad():
        pts, z_vals = nerf_sampler.sample(
            rays_o,
            rays_d,
        )

    print("Input Points")
    print(pts.shape)
    print("")
    print("Distances Along Ray")
    print(z_vals.shape)

    y_vals = torch.zeros_like(z_vals)

    nerf_sampler_unperturbed = NeRF_Stratified_Sampler(
        near=nerf_data_loader.near,
        far=nerf_data_loader.far,
        n_samples=n_samples,
        perturb=False,
        inverse_depth=inverse_depth,
    )

    _, z_vals_unperturbed = nerf_sampler_unperturbed.sample(
        rays_o,
        rays_d,
    )

    if show:
        plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), "b-o")
        plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), "r-o")
        plt.ylim([-1, 2])
        plt.title("Stratified Sampling (blue) with Perturbation (red)")
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        plt.grid(True)
        plt.show()

    return (ray_origin.shape, ray_slice, ray_direction.shape, raydirection_slice, pts.shape, z_vals.shape)


def debug_NeRF_model() -> Tuple[int, int, list[int]]:
    tensor_pos = torch.tensor([1.0, 1.0, 1.0])
    tensor_dir = torch.tensor([1.0, 1.0, 1.0])

    pos_encoder = PositionalEmbedding(n_freqs=10, input_dim=3)
    vdir_encoder = PositionalEmbedding(n_freqs=4, input_dim=3)

    d_pos = pos_encoder.out_dim
    d_vdir = vdir_encoder.out_dim

    tensor_pos_encoded = pos_encoder(tensor_pos).unsqueeze(0)
    tensor_vdir_encoded = vdir_encoder(tensor_dir).unsqueeze(0)

    nerf_model = NeRF(d_input_pos=d_pos, d_viewdirs=d_vdir)
    result = nerf_model(tensor_pos_encoded, tensor_vdir_encoded)

    d_result = list(result.shape)

    return d_pos, d_vdir, d_result


def debug_NeRF_renderer():
    raw = torch.tensor([1.0, 1.0, 1.0, 0.5]).unsqueeze(0)
    z_vals = torch.tensor(torch.linspace(2.0, 6.0, 8)).unsqueeze(0)
    ray_dir = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
        ]
    ).unsqueeze(0)

    renderer = Differentiable_Volume_Renderer()
    rgb_map, depth_map, acc_map, weights = renderer.raw_to_outputs(raw, z_vals, ray_dir)

    print("Testrendering done")
    pass


def debug_NeRF_forward_pass():
    testimg, testpose = debug_testimg_show(False)
    testimg = torch.from_numpy(testimg).to(device)
    testpose = torch.from_numpy(testpose).to(device)

    with torch.no_grad():
        ray_origin, ray_direction = nerf_data_loader.get_rays(
            nerf_data_loader.height, nerf_data_loader.width, nerf_data_loader.focal, testpose
        )

    rays_o = ray_origin.view([-1, 3])
    rays_d = ray_direction.view([-1, 3])

    renderer = Differentiable_Volume_Renderer()

    d_pos = nerf_data_loader.pos_embedder.out_dim
    d_vdir = nerf_data_loader.viewdir_embedder.out_dim
    nerf_model_coarse = NeRF(d_input_pos=d_pos, d_viewdirs=d_vdir)
    nerf_model_fine = NeRF(d_input_pos=d_pos, d_viewdirs=d_vdir)

    n_samples = 8
    perturb = True
    inverse_depth = False
    nerf_sampler_coarse = NeRF_Stratified_Sampler(
        near=nerf_data_loader.near,
        far=nerf_data_loader.far,
        n_samples=n_samples,
        perturb=perturb,
        inverse_depth=inverse_depth,
    )

    nerf_sampler_fine = NeRF_Hierarchical_Sampler(
        n_samples=n_samples,
        perturb=perturb,
    )

    outputs = nerf_forward(
        rays_o,
        rays_d,
        nerf_data_loader,
        renderer,
        nerf_model_coarse,
        nerf_sampler_coarse,
        fine_model=nerf_model_fine,
        sampler_fine=nerf_sampler_fine,
    )

    pass


def debug_model_init():
    experiment_file = "experiments/testexperiment.txt"
    args = dict_from_file(experiment_file)

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
    pass


def test_basic_image_information():
    num_img, img_shape, pose_shape, focal = nerf_data_loader.debug_information()
    assert num_img == 106
    assert img_shape == (106, 100, 100, 3)
    assert pose_shape == (106, 4, 4)
    assert abs(focal - 138.8) <= 1.0


def test_testimage():
    testimg, testpose = debug_testimg_show(False)
    assert testimg.shape == (100, 100, 3)
    assert testpose.shape == (4, 4)


def test_cam_directions_origins():
    cam_origins, cam_dirs = debug_cam_directions_origins(False)
    assert cam_origins.shape == (106, 3)
    assert cam_dirs.shape == (106, 3)


def test_rays_generation():
    (
        ray_origin_shape,
        ray_slice,
        ray_direction_shape,
        raydirection_slice,
        pts_ray_shape,
        z_vals_shape,
    ) = debug_rays_generation(False)
    assert list(ray_origin_shape) == [100, 100, 3]
    assert abs((ray_slice.numpy() - np.array([-1.9745, -1.8789, 2.9700])).sum()) <= 1.0
    assert list(ray_direction_shape) == [100, 100, 3]
    assert abs((raydirection_slice.numpy() - np.array([0.4898, 0.4661, -0.7368])).sum()) <= 1.0
    assert list(pts_ray_shape) == [10000, 8, 3]
    assert list(z_vals_shape) == [10000, 8]


# TODO NeRF debug test, renderer Test

# TODO NeRF inference test

# def test_NeRF_model():
#     d_pos, d_vdir, d_result = debug_NeRF_model()
#     assert d_pos == 60
#     assert d_vdir == 24
#     assert d_result == [1, 4]


if __name__ == "__main__":
    # debug_testimg_show(True)
    # debug_cam_directions_origins(True)
    debug_rays_generation(True)

    # debug_NeRF_model()
    # debug_NeRF_renderer()
    # debug_NeRF_forward_pass()
    # debug_model_init()
    pass
