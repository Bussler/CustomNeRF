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


def dict_from_file(filename) -> dict:
    """Generate dictionary from experimental description file.
    File can hold floats, ints, lists and strings.
    Values have to be separated by "=".
    Comments can be added by "#" and are filtered in the parsing.

    Args:
        filename (_type_): file to read from

    Returns:
        dict: argument dictionary
    """
    file = open(filename, "r")
    Lines = file.readlines()

    d = {}
    for line in Lines:
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        line = line.replace("\t", "")

        line = line.split("#")[0]  # M: remove comments from line
        lineParts = line.split("=")  # M: split line into key and value

        if len(lineParts) <= 1:
            continue

        key = lineParts[0]
        value = lineParts[1]

        # M: parse int, float, list or string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if "[" in value or "]" in value:
                    value = value.replace("[", "")
                    value = value.replace("]", "")
                    value = value.split(",")

                    try:
                        value = [int(x) for x in value]
                    except ValueError:
                        try:
                            value = [float(x) for x in value]
                        except ValueError:  # M: empty list
                            value = []
                else:
                    if value == "True" or value == "False":
                        value = bool(value)
                    else:  # M: normal string
                        value = value.replace('"', "")
                        value = str(value)

        d[key] = value

    return d
