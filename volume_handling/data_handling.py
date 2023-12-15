from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.feature_embedding import Embedder, PositionalEmbedding
from volume_handling.sampling import NeRF_Ray_Generator
from volume_handling.utils import load_npy_data


class NeRF_Data_Loader:
    def __init__(
        self,
        data_path="data/tiny_nerf_data.npz",
        poses_bounds=False,
        pos_embedder: Embedder = None,
        viewdir_embedder: Embedder = None,
        device: torch.device = torch.device("cpu"),
        n_training: int = 100,
        near=2.0,
        far=6.0,
    ) -> None:
        # load data images
        self.data = self.load_data(data_path, boundsfile=poses_bounds)
        self.images = self.data["images"]
        self.train_images = []
        self.validation_images = []
        self.poses = self.data["poses"]
        self.train_poses = []
        self.validation_poses = []

        # camera parameters
        if "hwf" in self.data:
            H, W, self.focal = self.data["hwf"]
            self.focal = np.array([self.focal])
        else:
            self.focal = self.data["focal"]
        self.height, self.width = self.images.shape[1:3]

        if "near" in self.data and "far" in self.data:
            self.near = self.data["near"]
            self.far = self.data["far"]
        else:
            self.near = near
            self.far = far

        # convert 4*4 camera pose mat into origin camera pos + dir vector to indicate where the camera is pointing
        # self.cam_dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.poses])
        # self.cam_origins = self.poses[:, :3, -1]

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

    def load_data(self, data_path: str, boundsfile=False) -> dict:
        """Load data from npz or npy file.

        Args:
            data_path (str): directory to data file
            boundsfile (bool, optional): False: npz, True: npy. Defaults to False.

        Returns:
            dict: dict containing images, poses, focal length, near and far clipping planes
        """
        if boundsfile:
            data = load_npy_data(data_path)
        else:
            data = np.load(data_path)
        return data

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
        self.focal = torch.from_numpy(self.focal).to(device)

    def get_training_rays(self) -> torch.Tensor:
        """Generate a tensor of all rays o, d, rgb values for training.

        Returns:
            torch.Tensor: shuffled tensor of rays o, d, rgb values in shape [n_training*width*height, 3, 3]
        """
        all_rays = torch.stack(
            [
                torch.stack(NeRF_Ray_Generator.get_rays(self.height, self.width, self.focal, p), 0)
                for p in self.train_poses
            ],
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

    def get_validation_image_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        testimg_idx = np.random.randint(self.validation_images.shape[0])
        return self.validation_images[testimg_idx], self.validation_poses[testimg_idx]

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
