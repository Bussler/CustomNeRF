import numpy as np
import torch
from torch import nn


class RadianceFieldEncoder(nn.Module):
    """Basis class for implicitly representing a radiance field of a scene with a neural network.

    Args:
        nn (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract Method, child should implement this!")
