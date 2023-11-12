import torch
from torch.utils.data.dataset import Dataset


class Ray_Rgb_Dataset(Dataset):
    def __init__(
        self,
        rays_rgb: torch.Tensor,
        batch_size: int = 2**15,
    ):
        self.rays_rgb = rays_rgb
        self.batch_size = batch_size

    def __len__(self):
        return self.rays_rgb.shape[0] // self.batch_size

    def __getitem__(self, index):
        # In batch_size steps over rays.
        batch = self.rays_rgb[index * self.batch_size : (index + 1) * self.batch_size]
        batch = torch.transpose(batch, 0, 1)
        index += 1
        # Shuffle after one epoch
        if (index + 1) * self.batch_size >= self.rays_rgb.shape[0]:
            self.rays_rgb = self.rays_rgb[torch.randperm(self.rays_rgb.shape[0])]
            index = 0

        rays_o, rays_d, target_img = batch
        return rays_o, rays_d, target_img, index
