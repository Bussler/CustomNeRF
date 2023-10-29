from typing import Optional, Tuple

import torch


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

    def raw_to_outputs(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
