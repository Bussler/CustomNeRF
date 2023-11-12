from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.feature_embedding import Embedder, PositionalEmbedding

from .sampling import NeRF_Sampler, NeRF_Stratified_Sampler


class NeRF_Data_Loader:
    def __init__(
        self,
        data_path="data/tiny_nerf_data.npz",
        pos_embedder: Embedder = None,
        viewdir_embedder: Embedder = None,
        device: torch.device = torch.device("cpu"),
        n_training: int = 100,
        near=2.0,
        far=6.0,
    ) -> None:
        # load data images
        self.data = np.load(data_path)
        self.images = self.data["images"]
        self.train_images = []
        self.validation_images = []
        self.poses = self.data["poses"]
        self.train_poses = []
        self.validation_poses = []
        self.focal = self.data["focal"]

        # camera parameters
        self.height, self.width = self.images.shape[1:3]
        self.near = near
        self.far = far

        # convert 4*4 camera pose mat into origin camera pos + dir vector to indicate where the camera is pointing
        self.cam_dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.poses])
        self.cam_origins = self.poses[:, :3, -1]

        # TODO should this class have a method like sample() that instantly generates the samples along each ray?
        # set class for sampling points from volume
        # if point_sampler is not None:
        #     self.point_sampler = point_sampler
        # else:
        #     self.point_sampler = NeRF_Stratified_Sampler()

        # set high frequency embedders for input to network
        if pos_embedder is not None:
            self.pos_embedder = pos_embedder
        else:
            self.pos_embedder = PositionalEmbedding(n_freqs=10, input_dim=3)

        if viewdir_embedder is not None:
            self.viewdir_embedder = viewdir_embedder
        else:
            self.viewdir_embedder = PositionalEmbedding(n_freqs=4, input_dim=3)

        # move data to torch device
        self.device = device
        self.n_training = n_training
        self.data_to_device(self.device, self.n_training)

    def data_to_device(self, device: torch.device, n_training: int = 100):
        """move data (images, poses, focal) to torch device (cpu or cuda)

        Args:
            device (torch.device): device to move to
            n_training (int, optional): how many training images to move. Defaults to 100.
        """
        self.train_images = torch.from_numpy(self.data["images"][:n_training]).to(device)
        self.validation_images = torch.from_numpy(self.data["images"][n_training:]).to(device)
        self.train_poses = torch.from_numpy(self.data["poses"][:n_training]).to(device)
        self.validation_poses = torch.from_numpy(self.data["poses"][n_training:]).to(device)
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

    def get_training_rays(self) -> torch.Tensor:
        """Generate a tensor of all rays o, d, rgb values for training.

        Returns:
            torch.Tensor: shuffled tensor of rays o, d, rgb values in shape [n_training*width*height, 3, 3]
        """
        all_rays = torch.stack(
            [torch.stack(self.get_rays(self.height, self.width, self.focal, p), 0) for p in self.train_poses],
            0,
        )  # M: stack for all training images the rays_o and rays_d; shape [n_training, 2, width, height, 3]
        rays_rgb = torch.cat(
            [all_rays, self.train_images[:, None]], 1
        )  # M: add rgb values to rays; shape [n_training, 3, width, height, 3]
        rays_rgb = torch.permute(
            rays_rgb, [0, 2, 3, 1, 4]
        )  # M: reorder o, d, rgb values to singular rays; reshape to [n_training, width, height, 3, 3]
        rays_rgb = rays_rgb.reshape([-1, 3, 3])  # M: flatten; reshape to [n_training*width*height, 3, 3]
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]  # M: shuffle rays
        return rays_rgb

    def get_chunks(self, inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
        """Helper function to divide an input into chunks.

        Args:
            inputs (torch.Tensor): input to chunk (either positions or view_dirs)
            chunksize (int, optional): size of the chunks. Defaults to 2**15.

        Returns:
            List[torch.Tensor]: list of chunks
        """
        return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    def prepare_position_chunks(
        self, points: torch.Tensor, encoding_function: Embedder = None, chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        """Feature-Encode and chunkify sampled points to prepare for NeRF model.

        Args:
            points (torch.Tensor): sampled input points
            encoding_function (Embedder), torch.Tensor]): frequency embedder
            chunksize (int, optional): size of the chunks. Defaults to 2**15.

        Returns:
            List[torch.Tensor]: list of chunks
        """
        points = points.reshape((-1, 3))

        if encoding_function is None:
            points = self.pos_embedder(points)
        else:
            points = encoding_function(points)

        points = self.get_chunks(points, chunksize=chunksize)
        return points

    def prepare_viewdirs_chunks(
        self,
        points: torch.Tensor,
        rays_d: torch.Tensor,
        encoding_function: Embedder = None,
        chunksize: int = 2**15,
    ) -> List[torch.Tensor]:
        """Feature-Encode and chunkify viewdirs to prepare for NeRF model.

        Args:
            points (torch.Tensor): sampled input points as reference for tensor size
            rays_d (torch.Tensor): sampled view directions
            encoding_function (Embedder): frequency embedder
            chunksize (int, optional): size of the chunks. Defaults to 2**15.

        Returns:
            List[torch.Tensor]: list of chunks
        """
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))

        if encoding_function is None:
            viewdirs = self.viewdir_embedder(viewdirs)
        else:
            viewdirs = encoding_function(viewdirs)

        viewdirs = self.get_chunks(viewdirs, chunksize=chunksize)
        return viewdirs

    def debug_information(self) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int], float]:
        print(f"Num images: {self.images.shape[0]}")
        print(f"Images shape: {self.images.shape}")
        print(f"Poses shape: {self.poses.shape}")
        print(f"Focal length: {self.focal}")
        return self.images.shape[0], self.images.shape, self.poses.shape, self.focal
