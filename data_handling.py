import matplotlib.pyplot as plt
import numpy as np


class NeRF_Data_Loader:
    def __init__(self, data_path="data/tiny_nerf_data.npz") -> None:
        # load data images
        self.data = np.load(data_path)
        self.images = self.data["images"]
        self.poses = self.data["poses"]
        self.focal = self.data["focal"]

        # camera parameters
        height, width = self.images.shape[1:3]
        self.near, self.far = 2.0, 6.0

    def debug_information(self):
        print(f"Num images: {self.images.shape[0]}")
        print(f"Images shape: {self.images.shape}")
        print(f"Poses shape: {self.poses.shape}")
        print(f"Focal length: {self.focal}")

    def testimg_show(self):
        testimg_idx = 101
        testimg, testpose = self.images[testimg_idx], self.poses[testimg_idx]
        imgplot = plt.imshow(testimg)
        plt.show()
        print("Pose")
        print(testpose)
