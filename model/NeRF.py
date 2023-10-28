from typing import Optional

import numpy as np
import torch
from RadianceFieldEncoder import RadianceFieldEncoder
from torch import nn


class NeRF(RadianceFieldEncoder):
    def __init__(self) -> None:
        super().__init__()
        # TODO M: add attributes for holding input sizes, network layers...

    def forward(self, position: torch.Tensor, view_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO M: put position (for opacity) and optionally view_dir (for color) through newtwork to get opacity and radiance
        pass
