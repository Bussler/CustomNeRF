import torch
from torch.utils.data.dataset import Dataset


class Ray_Rgb_Dataset(Dataset):
    """Dataset for getting batches of rays (origin, direction) and corresponding rgb values.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        rays_rgb: torch.Tensor,
        batch_size: int = 2**15,
    ):
        """_summary_

        Args:
            rays_rgb (torch.Tensor): Tensor of all available rays (origin, direction) and corresponding rgb values for training.
            batch_size (int, optional): Size of batches for training. Defaults to 2**15.
        """
        self.rays_rgb = rays_rgb
        self.batch_size = batch_size
        self.ibatch = 0

    def __len__(self):
        return self.rays_rgb.shape[0] // self.batch_size

    def __getitem__(self, index):
        # In batch_size steps over rays.
        batch = self.rays_rgb[self.ibatch : self.ibatch + self.batch_size]
        batch = torch.transpose(batch, 0, 1)
        self.ibatch += self.batch_size
        # Shuffle after one epoch
        if self.ibatch >= self.rays_rgb.shape[0]:
            self.rays_rgb = self.rays_rgb[torch.randperm(self.rays_rgb.shape[0])]
            self.ibatch = 0

        rays_o, rays_d, target_img = batch
        return rays_o, rays_d, target_img
