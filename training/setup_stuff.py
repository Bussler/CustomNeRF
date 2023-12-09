from typing import Tuple

import torch

from model.feature_embedding import PositionalEmbedding
from model.NeRF import NeRF
from model.RadianceFieldEncoder import RadianceFieldEncoder
from training.early_stopping import EarlyStopping
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rendering import Differentiable_Volume_Renderer
from volume_handling.sampling import (
    NeRF_Hierarchical_Sampler,
    NeRF_Sampler,
    NeRF_Stratified_Sampler,
)


def init_models(
    device: torch.device,
    data_path: str = "data/tiny_nerf_data.npz",
    near: float = 2.0,
    far: float = 6.0,
    n_training: int = 100,
    use_viewdirs: bool = True,
    d_input: int = 3,
    n_freqs: int = 10,
    n_freqs_views: int = 4,
    log_space: bool = True,
    n_samples: int = 64,
    perturb: bool = True,
    inverse_depth: bool = False,
    use_fine_model: bool = True,
    n_layers: int = 2,
    d_Weights: int = 128,
    n_layers_fine: int = 6,
    d_Weights_fine: int = 128,
    skip: Tuple[int] = (),
    lr: float = 5e-4,
) -> Tuple[
    RadianceFieldEncoder,
    RadianceFieldEncoder,
    NeRF_Data_Loader,
    NeRF_Sampler,
    NeRF_Sampler,
    torch.optim.Optimizer,
    EarlyStopping,
]:
    """
    Initialize models, encoders, dataloaders, renderer and optimizer for NeRF training.

    Args:
        device (torch.device): The device to run the models on.
        data_path (str, optional): Path to the data file. Defaults to "data/tiny_nerf_data.npz".
        near (float, optional): Near clipping distance. Defaults to 2.0.
        far (float, optional): Far clipping distance. Defaults to 6.0.
        use_viewdirs (bool, optional): Whether to use view directions as input for NeRF model. Defaults to True.
        d_input (int, optional): Number of input dimensions to Embedder Function (for postion and viewdir). Defaults to 3.
        n_freqs (int, optional): Number of encoding functions for positional samples. Defaults to 10.
        n_freqs_views (int, optional): Number of encoding functions for view dirs. Defaults to 4.
        log_space (bool, optional): If set, frequencies scale in log space. Defaults to True.
        n_samples (int, optional): Number of spatial samples per ray. Defaults to 64.
        perturb (bool, optional): If set, applies noise to sample positions. Defaults to True.
        inverse_depth (bool, optional): If set, samples points linearly in inverse depth. Defaults to False.
        use_fine_model (bool, optional): If set, creates a fine model (hierarchical after genera, coarse model). Defaults to True.
        n_layers (int, optional): Number of layers in NeRF main corpus. Defaults to 2.
        d_Weights (int, optional): Dimensions of linear layers in NeRF. Defaults to 128.
        n_layers_fine (int, optional): Number of layers in fine network bottleneck. Defaults to 6.
        d_Weights_fine (int, optional): Dimensions of linear layers of fine network. Defaults to 128.
        skip (Tuple[int], optional): Layers at which to apply input residual (resnet skip connetion) in NeRF. Defaults to ().
        lr (float, optional): Learning rate. Defaults to 5e-4.

    Returns:
        Tuple[RadianceFieldEncoder, RadianceFieldEncoder, NeRF_Data_Loader, NeRF_Sampler, NeRF_Sampler, Differentiable_Volume_Renderer]:
        model, fine_model, data_loader, nerf_sampler_coarse, nerf_sampler_fine, renderer, optimizer, warmup_stopper
    """
    # Embedders
    encoder = PositionalEmbedding(n_freqs, d_input, log_space=log_space)

    # View direction embedder
    if use_viewdirs:
        encoder_viewdirs = PositionalEmbedding(n_freqs_views, d_input, log_space=log_space)
        d_viewdirs = encoder_viewdirs.out_dim
    else:
        encoder_viewdirs = None
        d_viewdirs = None

    # Data Loader
    data_loader = NeRF_Data_Loader(data_path, encoder, encoder_viewdirs, device, n_training, near, far)

    # Samplers
    nerf_sampler_coarse = NeRF_Stratified_Sampler(
        near=data_loader.near,
        far=data_loader.far,
        n_samples=n_samples,
        perturb=perturb,
        inverse_depth=inverse_depth,
    )

    if use_fine_model:
        nerf_sampler_fine = NeRF_Hierarchical_Sampler(
            n_samples=n_samples,
            perturb=perturb,
        )
    else:
        nerf_sampler_fine = None

    # Models
    model = NeRF(encoder.out_dim, n_layers=n_layers, d_Weights=d_Weights, skip=skip, d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(
            encoder.out_dim, n_layers=n_layers_fine, d_Weights=d_Weights_fine, skip=skip, d_viewdirs=d_viewdirs
        )
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, data_loader, nerf_sampler_coarse, nerf_sampler_fine, optimizer, warmup_stopper


def load_model(
    device: torch.device,
    data_path: str = "data/tiny_nerf_data.npz",
    model_path: str = "experiments/test_exp/nerf.pt",
    fine_model_path: str = "experiments/test_exp/nerf-fine.pt",
    near: float = 2.0,
    far: float = 6.0,
    n_training: int = 100,
    use_viewdirs: bool = True,
    d_input: int = 3,
    n_freqs: int = 10,
    n_freqs_views: int = 4,
    log_space: bool = True,
    n_samples: int = 64,
    perturb: bool = True,
    inverse_depth: bool = False,
    use_fine_model: bool = True,
    n_layers: int = 2,
    d_Weights: int = 128,
    n_layers_fine: int = 6,
    d_Weights_fine: int = 128,
    skip: Tuple[int] = (),
) -> Tuple[RadianceFieldEncoder, RadianceFieldEncoder, NeRF_Data_Loader, NeRF_Data_Loader, NeRF_Sampler, NeRF_Sampler,]:
    # Embedders
    encoder = PositionalEmbedding(n_freqs, d_input, log_space=log_space)

    # View direction embedder
    if use_viewdirs:
        encoder_viewdirs = PositionalEmbedding(n_freqs_views, d_input, log_space=log_space)
        d_viewdirs = encoder_viewdirs.out_dim
    else:
        encoder_viewdirs = None
        d_viewdirs = None

    # Data Loader
    data_loader = NeRF_Data_Loader(data_path, encoder, encoder_viewdirs, device, n_training, near, far)

    # Samplers
    nerf_sampler_coarse = NeRF_Stratified_Sampler(
        near=data_loader.near,
        far=data_loader.far,
        n_samples=n_samples,
        perturb=perturb,
        inverse_depth=inverse_depth,
    )

    if use_fine_model:
        nerf_sampler_fine = NeRF_Hierarchical_Sampler(
            n_samples=n_samples,
            perturb=perturb,
        )
    else:
        nerf_sampler_fine = None

    # Models: load from checkpoint
    model = NeRF(encoder.out_dim, n_layers=n_layers, d_Weights=d_Weights, skip=skip, d_viewdirs=d_viewdirs)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    if use_fine_model:
        fine_model = NeRF(
            encoder.out_dim, n_layers=n_layers_fine, d_Weights=d_Weights_fine, skip=skip, d_viewdirs=d_viewdirs
        )
        fine_model.to(device)

        fine_checkpoint = torch.load(fine_model_path)
        fine_model.load_state_dict(fine_checkpoint)
    else:
        fine_model = None

    return model, fine_model, data_loader, nerf_sampler_coarse, nerf_sampler_fine
