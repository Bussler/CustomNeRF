from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class NeRF_Data_Loader:
    def __init__(self, data_path="data/tiny_nerf_data.npz") -> None:
        # load data images
        self.data = np.load(data_path)
        self.images = self.data["images"]
        self.poses = self.data["poses"]
        self.focal = self.data["focal"]

        # camera parameters
        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2.0, 6.0

        # convert 4*4 camera pose mat into origin camera pos + dir vector to indicate where the camera is pointing
        self.cam_dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.poses])
        self.cam_origins = self.poses[:, :3, -1]

    def data_to_device(self, device: torch.device, n_training: int = 100):
        """move data (images, poses, focal) to torch device (cpu or cuda)

        Args:
            device (torch.device): device to move to
            n_training (int, optional): how many training images to move. Defaults to 100.
        """

        # Gather as torch tensors
        self.images = torch.from_numpy(self.data["images"][:n_training]).to(device)
        self.poses = torch.from_numpy(self.data["poses"]).to(device)
        self.focal = torch.from_numpy(self.data["focal"]).to(device)

    def get_rays(
        self, height: int, width: int, focal_length: float, c2w: torch.Tensor
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
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
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
        ).to(c2w)

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

        # Origin is same for all pixels/ directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def debug_information(self) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int], float]:
        print(f"Num images: {self.images.shape[0]}")
        print(f"Images shape: {self.images.shape}")
        print(f"Poses shape: {self.poses.shape}")
        print(f"Focal length: {self.focal}")
        return self.images.shape[0], self.images.shape, self.poses.shape, self.focal
