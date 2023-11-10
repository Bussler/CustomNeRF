import numpy as np
import torch
from torch import nn


class Embedder(nn.Module):
    """Class to embed input into higher-dimensional frequency space"""

    def __init__(self):
        super().__init__()

        self.embed_functions = []
        self.out_dim = 0

    def create_embedding_function(self, n_freqs: int, input_dim: int, log_space: bool = False):
        """Abstract function for concrete child Embedder classes to
        create the high frequency embedding functions for the input.
        The embedding functions are stored into self.embed_functions
        and used in the forward pass on the model input.

        Args:
            n_freqs (int): frequencies to use for embedding.
            More frequencies means larger embedding.
            input_dim (int): dimensionality of original input.
            log_space (bool, optional): Whether to sample frequencies
            in linspace or logspace. Defaults to False.
        """
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply embedding functions to all inputs and concatenate results to matrix.

        Args:
            inputs (torch.Tensor): model input

        Returns:
            torch.Tensor: input embedded into higher dimensional space
        """
        embedded = torch.cat([fn(inputs) for fn in self.embed_functions], -1)
        return embedded


class PositionalEmbedding(Embedder):
    """Original positional embedding as described by Mildenhall NeRF paper

    Args:
        Embedder (_type_): _description_
    """

    def __init__(self, n_freqs: int, input_dim: int, log_space: bool = False):
        super(PositionalEmbedding, self).__init__()
        self.periodic_functions = [torch.sin, torch.cos]
        self.create_embedding_function(n_freqs, input_dim, log_space)

    def create_embedding_function(self, n_freqs: int, input_dim: int, log_space: bool = False):
        if log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, n_freqs - 1, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2**0.0, 2 ** (n_freqs - 1), steps=n_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_functions:
                self.embed_functions.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += input_dim


class FourierEmbedding(Embedder):
    """Fourier embedding like described in Weiss fv-SRN paper

    Args:
        Embedder (_type_): _description_
    """

    def __init__(self, n_freqs: int, input_dim: int):
        super(FourierEmbedding, self).__init__()

        self.periodic_functions = [torch.sin, torch.cos]
        self.create_embedding_function(n_freqs, input_dim)

    def create_embedding_function(self, n_freqs: int, input_dim: int):
        freq_bands = 2.0 ** torch.linspace(0.0, n_freqs - 1, steps=n_freqs)
        freq_bands = freq_bands * 2.0 * np.pi

        for freq in freq_bands:
            for p_fn in self.periodic_functions:
                self.embed_functions.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += input_dim
