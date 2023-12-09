from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Differentiable_Volume_Renderer:
    """
    Apply volume integration to convert rgbo values of NeRF samples into pixel values.

    Take the weighted sum of all samples along the ray of each pixel to get the estimated color value at that pixel.

    Each RGB sample is weighted by its alpha value.
    Higher alpha values indicate higher likelihood that the sampled area is opaque,
    therefore points further along the ray are likelier to be occluded.
    The cumulative product ensures that those further points are dampened.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Calculate cumulative product of a tensor: yi = x1 * x2 * x3 * ... xi.

        (Courtesy of https://github.com/krrish94/nerf-pytorch)

        Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

        Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
            is to be computed.
        Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
            tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
        """

        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, -1)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, -1)
        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        cumprod[..., 0] = 1.0

        return cumprod

    @classmethod
    def raw_to_outputs(
        cls,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw rgbo samples from nerf nw to pixel color values

        Args:
            raw (torch.Tensor): nerf output
            z_vals (torch.Tensor): depth values of samples along the rays
            rays_d (torch.Tensor): direction vector of ray
            raw_noise_std (float, optional): noise to predicted opacity values, for more stability during training. Defaults to 0.0.
            white_bkgd (bool, optional): If the background is white or black. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: rgb_map, depth_map, acc_map, weights
        """
        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        # This is done by looking at the predicted opacity and multiplying with dist to prev sample point.
        alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)

        # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
        # The higher the alpha, the lower subsequent weights are driven, since an opaque surface was reached.
        weights = alpha * cls.cumprod_exclusive(1.0 - alpha + 1e-10)

        # Compute weighted RGB map: Weight predicted colors by alpha values and sum to pixel value.
        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

        # Estimated depth map is predicted distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

        # Sum of weights along each ray. In [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights
