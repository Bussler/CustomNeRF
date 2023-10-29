from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .RadianceFieldEncoder import RadianceFieldEncoder


class NeRF(RadianceFieldEncoder):
    """Implementation of the original NeRF encoder according to Mildenhall et al."""

    def __init__(
        self,
        d_input_pos: int = 3,
        n_layers: int = 8,
        d_Weights: int = 256,
        skip: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None,
    ) -> None:
        """Create NeRF encoder.

        Args:
            d_input_pos (int, optional): Dimension of positional input (usually xyz). Defaults to 3.
            n_layers (int, optional): Number of middle layers. Defaults to 8.
            d_Weights (int, optional): Dimension of middle layers. Defaults to 256.
            skip (Tuple[int], optional): Where to place residual layers.
            These layers concatenate the original input with intermediate features
            to prevent information loss. Defaults to (4,).
            d_viewdirs (Optional[int], optional): Dimension of viewdir input (from sampling rays).
            Defaults to None.
        """
        super().__init__()
        self.d_input_pos = d_input_pos
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input_pos, d_Weights)]
            + [
                nn.Linear(d_Weights + self.d_input_pos, d_Weights) if i in skip else nn.Linear(d_Weights, d_Weights)
                for i in range(n_layers - 1)
            ]
        )

        # Final linear output layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_Weights, 1)
            self.rgb_feature = nn.Linear(d_Weights, d_Weights)
            self.rgb_branch = nn.Linear(d_Weights + self.d_viewdirs, d_Weights // 2)
            self.rgb_out = nn.Linear(d_Weights // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_Weights, 4)

    def forward(self, x: torch.Tensor, view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply forward pass through NeRF nw, generates rgbo from xyz and view direction.

        Args:
            x (torch.Tensor): (x,y,z) position vector of sampled points.
            view_dirs (Optional[torch.Tensor], optional): view direction from generating rays. Defaults to None.

        Returns:
            torch.Tensor: (r,g,b,o)
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and view_dirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

        # Apply forward pass through main nw
        x_input = x
        for i, layer in enumerate(self.layers):
            # x = self.act(layer(x)) # TODO M: try this: better way to use activation function?
            x = layer(x)
            x = F.relu(x)
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply output layers
        if self.d_viewdirs is not None:
            # Split alpha from first network output
            alpha = self.alpha_out(x)

            # Generate features for rgb from position
            x = self.rgb_feature(x)
            x = torch.cat([x, view_dirs], dim=-1)

            # Get rgb value from input viewdir and position features
            # x = self.act(self.branch(x))
            x = self.rgb_branch(x)
            x = F.relu(x)
            rgb = self.rgb_out(x)

            # Concatenate alphas to output
            x = torch.cat([rgb, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x
