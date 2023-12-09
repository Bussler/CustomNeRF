from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class NeRF_Ray_Generator:
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_rays(
        cls, height: int, width: int, focal_length: float, c2w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find origin and direction of rays through every pixel and camera origin.

        Args:
            height: int: height of input image
            width: int: width of input image
            focal_length: int: focal length of camera model
            c2w: torch.Tensor: camera to world matrix, camera pose to get projection lines for each image pixel

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Ray Origin (for each pixel, xyz), Ray Direction (for each pixel, xyz for dir)
        """

        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=c2w.device),
            torch.arange(height, dtype=torch.float32, device=c2w.device),
            indexing="ij",
        )
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - width * 0.5) / focal_length,
                -(j - height * 0.5) / focal_length,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

        # Origin is same for all pixels/ directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d


class NeRF_Sampler(ABC):
    def __init__(
        self,
        n_samples: int,
        perturb: Optional[bool] = True,
    ) -> None:
        """Abstract class for sampler to generate point (position) values along a ray

        Args:
            n_samples (int): How many points to sample along each ray. Influences the space integration along the ray
            perturb (Optional[bool], optional): Determines whether to sample points uniformly from each bin or to simply use the bin center as the point. Defaults to True.
        """
        self.n_samples = n_samples
        self.perturb = perturb

    @abstractmethod
    def sample(self, *args) -> None:
        raise NotImplementedError("Abstract Method, child should implement this!")


class NeRF_Stratified_Sampler(NeRF_Sampler):
    """Sampler for broad, scene structure focused NeRF

    Args:
        NeRF_Sampler (_type_): _description_
    """

    def __init__(
        self,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False,
    ) -> None:
        """Sampler for broad, scene structure focused NeRF

        Args:
            near (float): near plane z value
            far (float): far plane z value
            inverse_depth (bool, optional): _description_. Defaults to False.
        """
        super().__init__(n_samples, perturb)
        self.near = near
        self.far = far
        self.inverse_depth = inverse_depth

    def sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample along ray from regularly-spaced bins.
        Divide space into regular sized bins and sample uniformly from each bin.
        This allows for samples with uneven spacing (if perturb is enabled).

        Args:
            rays_o (torch.Tensor): origin position of rays
            rays_d (torch.Tensor): direction vector of rays

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: xyz for all sampled points along each ray; distance along ray of each point
        """

        # Grab samples for space integration along ray
        t_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        if not self.inverse_depth:
            # Generate depth values. Sample linearly between `near` and `far`
            z_vals = self.near * (1.0 - t_vals) + self.far * (t_vals)
        else:
            # Generate depth values. Sample linearly in inverse depth (disparity) (back to front)
            z_vals = 1.0 / (1.0 / self.near * (1.0 - t_vals) + 1.0 / self.far * (t_vals))

        # Draw uniform samples from bins along ray
        if self.perturb:
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)
            lower = torch.concat([z_vals[:1], mids], dim=-1)
            t_rand = torch.rand([self.n_samples], device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        # Expand the n_samples depth values to all rays
        z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [self.n_samples])

        # Generate Samplepoints: Apply scale from `rays_d` and offset from `rays_o` to distance offsets along ray
        # pts: (width * height, n_samples, 3)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals


class NeRF_Hierarchical_Sampler(NeRF_Sampler):
    """Sampler for fine, detailed NeRF

    Args:
        NeRF_Sampler (_type_): _description_
    """

    def __init__(
        self,
        n_samples: int,
        perturb: bool = False,
    ) -> None:
        super().__init__(n_samples, perturb)

    def sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply hierarchical sampling to the rays.
        NeRF uses two MLPs for scene representaion: Coarse for broad structure, Fine for details.
        Coarse model receives broad sampling (more or less evenly spaced: stratified sampling),
        Fine level honing in on areas with strong priors for salient information.
        Honing in is done by hierarchical vol rendering: oversample regions with high likelyhood to contribute to signal,
        since large areas of volume are empty.
        Apply learned, normalized weights to the first set of samples to create a PDF across the ray.
        Then apply inverse transform sampling to pdf to get second, finer batch of samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: new sample points, old and new z vals combined, new z vals
        """

        # Draw samples from PDF using z_vals as bins and weights as probabilities.
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self.sample_pdf(z_vals_mid, weights[..., 1:-1], self.n_samples, perturb=self.perturb)
        new_z_samples = new_z_samples.detach()

        # Resample points from ray based on PDF.
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        )  # [N_rays, N_samples + n_samples, 3]
        return pts, z_vals_combined, new_z_samples

    def sample_pdf(
        self,
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
    ) -> torch.Tensor:
        r"""
        Apply inverse transform sampling to a weighted set of points.
        """

        # Normalize weights to get PDF.
        pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
        cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not perturb:
            u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)  # [n_rays, n_samples]

        # Find indices along CDF where values in u would be placed.
        u = u.contiguous()  # Returns contiguous tensor with same values.
        inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

        # Clamp indices that are out of bounds.
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

        # Sample from cdf and the corresponding bin centers.
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

        # Convert samples to ray length.
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples  # [n_rays, n_samples]
