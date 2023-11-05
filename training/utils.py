from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_samples(
    z_vals: torch.Tensor, z_hierarch: Optional[torch.Tensor] = None, ax: Optional[np.ndarray] = None
) -> None:
    """Plot stratified and (optional) hierarchical samples on a plt subplot.

    Args:
        z_vals (torch.Tensor): depth values of samples along the ray of stratified sampling
        z_hierarch (Optional[torch.Tensor], optional): depth values of samples along the ray of hierarchical sampling. Defaults to None.
        ax (Optional[np.ndarray], optional): plt subplot. Defaults to None.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, "b-o")
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, "r-o")
    ax.set_ylim([-1, 2])
    ax.set_title("Stratified  Samples (blue) and Hierarchical Samples (red)")
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax


def crop_center(img: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """Crop center square from image.

    Args:
        img (torch.Tensor): image as tensor
        frac (float, optional): fraction to crop. Defaults to 0.5.

    Returns:
        torch.Tensor: cropped center square
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]
