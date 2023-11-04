from typing import Callable, Optional, Tuple

import torch
from torch import nn

from model.RadianceFieldEncoder import RadianceFieldEncoder
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.rendering import Differentiable_Volume_Renderer
from volume_handling.sampling import NeRF_Sampler


def nerf_inference(
    query_points: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    model: RadianceFieldEncoder,
    data_loader: NeRF_Data_Loader,
    chunksize: int,
    renderer: Differentiable_Volume_Renderer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate pixel rgb values from sampling points along ray with NeRF inference

    Args:
        query_points (torch.Tensor): sampled points positions
        rays_d (torch.Tensor): direction vectors of the corresponding rays
        z_vals (torch.Tensor): depth values of the sampled points along the rays
        model (RadianceFieldEncoder): NeRF model
        data_loader (NeRF_Data_Loader): data loader for batching
        chunksize (int): chunksize of batches
        renderer (Differentiable_Volume_Renderer): differentiable volume renderer to interpret the raw NeRF outputs

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    # Batchifz the points and put them through the embedding function
    batches = data_loader.prepare_position_chunks(query_points, chunksize=chunksize)

    # TODO M: find something to make viewdirs optional! Here the embedder is always set!
    if data_loader.viewdir_embedder is not None:
        batches_viewdirs = data_loader.prepare_viewdirs_chunks(query_points, rays_d, chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(model(batch, view_dirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = renderer.raw_to_outputs(raw, z_vals, rays_d)

    return rgb_map, depth_map, acc_map, weights


def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    data_loader: NeRF_Data_Loader,
    renderer: Differentiable_Volume_Renderer,
    coarse_model: RadianceFieldEncoder,
    sampler_coarse: NeRF_Sampler,
    fine_model: RadianceFieldEncoder = None,
    sampler_fine: NeRF_Sampler = None,
    chunksize: int = 2**15,
) -> dict[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward pass through coarse, fine model(s).

    Args:
        rays_o (torch.Tensor): origin points of rays
        rays_d (torch.Tensor): direction vector of rays
        data_loader (NeRF_Data_Loader): data loader used for batching positions and view_dirs
        renderer (Differentiable_Volume_Renderer): differentiable volume renderer to genereate pixel values from raw model output
        coarse_model (nn.Module): first, coarse structur predicting model
        sampler_coarse (NeRF_Sampler): Coarse sampler to generate position samples along the ray for input of coarse model.
        fine_model (_type_, optional): second, fine detailed model. Defaults to None.
        sampler_fine (NeRF_Sampler): Fine sampler to generate position samples along the ray for input of fine model. Defaults to None.
        chunksize (int, optional): chunksize of batches for training. Defaults to 2**15.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        dict containing rgb_map, depth_map, acc_map, weights (where to focus sampling, points of high alpha)
    """
    # Sample query points along each ray.
    query_points, z_vals = sampler_coarse.sample(rays_o, rays_d)
    rgb_map, depth_map, acc_map, weights = nerf_inference(
        query_points, rays_d, z_vals, coarse_model, data_loader, chunksize, renderer
    )

    outputs = {"z_vals_stratified": z_vals}

    # Fine model pass.
    if sampler_fine is not None:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        kwargs_sample_hierarchical = {"z_vals": z_vals, "weights": weights}
        query_points, z_vals_combined, z_hierarch = sampler_fine.sample(rays_o, rays_d, **kwargs_sample_hierarchical)

        fine_model = fine_model if fine_model is not None else coarse_model
        rgb_map, depth_map, acc_map, weights = nerf_inference(
            query_points, rays_d, z_vals_combined, fine_model, data_loader, chunksize, renderer
        )

        # Store outputs.
        outputs["z_vals_hierarchical"] = z_hierarch
        outputs["rgb_map_0"] = rgb_map_0
        outputs["depth_map_0"] = depth_map_0
        outputs["acc_map_0"] = acc_map_0

    # Store outputs.
    outputs["rgb_map"] = rgb_map
    outputs["depth_map"] = depth_map
    outputs["acc_map"] = acc_map
    outputs["weights"] = weights
    return outputs
