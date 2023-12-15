import os
from subprocess import check_output

import imageio
import numpy as np


def load_npy_data(data_path: str) -> dict:
    """
    Loads the numpy bounds data from the given path.

    This function reads the data from the provided path, corrects the rotation matrix ordering,
    moves the variable dimension to axis 0, and calculates the near and far bounds based on the
    loaded data. The loaded data includes poses, bounds, images, hwf (height, width, focal),
    near and far values.

    Args:
        data_path (str): The path to the folder with the images and numpy bounds data file.

    Returns:
        dict: A dictionary containing the loaded data. The keys in the dictionary are:
            - 'poses': The poses data as a numpy array.
            - 'bounds': The bounds data as a numpy array.
            - 'images': The images data as a numpy array.
            - 'hwf': The height, width, focal data as a numpy array.
            - 'near': The near value as a float.
            - 'far': The far value as a float.
    """
    poses, bounds, imgs = _load_data(data_path)

    print("Loaded", data_path)

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)
    print("bds:", bounds[0])

    print(f"Data: {poses.shape}, {imgs.shape}, {bounds.min()}, {bounds.max()}")

    hwf = poses[0, :3, -1]
    poses = poses[:, :, :4]

    near = np.ndarray.min(bounds) * 0.9
    far = np.ndarray.max(bounds) * 1.0

    return {"poses": poses, "bounds": bounds, "images": imgs, "hwf": hwf, "near": near, "far": far}


########## Slightly modified version of LLFF data loading code
########## See https://github.com/Fyusion/LLFF for original
########## Adapted from DSNERF https://github.com/dunbar12138/DSNeRF


def _minify(basedir, factors=[], resolutions=[]) -> None:
    """
    Minifies the images in the given directory based on the provided factors and resolutions.

    This function checks if the minified versions of the images already exist.
    If they do not exist, it creates new directories for the minified images, copies the original images into them,
    and then resizes the images according to the provided factors and resolutions.

    Args:
        basedir (str): The base directory where the original images are stored.
        factors (list, optional): A list of factors by which to reduce the size of the images.
            Each factor is an integer, and the size of the image is divided by this factor.
            Defaults to an empty list.
        resolutions (list, optional): A list of resolutions to which to resize the images.
            Each resolution is a tuple of two integers representing the width and height.
            Defaults to an empty list.

    """
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            print("Removed duplicates")
        print("Done")


# factor: downsample factor
def _load_data(
    basedir: str, factor: int = None, width: int = None, height: int = None, load_imgs: bool = True
) -> tuple[np.array, np.array, np.array]:
    """Loads image and pose data from a specified npy directory.

    Args:
        basedir (str): The base directory from which to load the data.
        factor (int, optional): The factor by which to minify the images. If not provided, the images will be minified based on the provided width or height.
        width (int, optional): The desired width of the images. If not provided, the images will be minified based on the provided factor or height.
        height (int, optional): The desired height of the images. If not provided, the images will be minified based on the provided factor or width.
        load_imgs (bool, optional): Whether to load the images. If False, only the poses and bounds will be returned. Default is True.

    Returns:
        poses (np.array): A 3D array of shape 4 x 5 x N containing the poses.
        bds (np.array): A 2D array (2 x N) containing the bounds.
        imgs (np.array, optional): A 4D array containing the images. Only returned if load_imgs is True.
    """
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3 x 5 x N

    bounds = poses_arr[:, -2:].transpose([1, 0])

    # M: extend poses to 4x4 matrices (add [0, 0, 0, 1] row)
    last_row = np.array([0, 0, 0, 1, 0]).reshape(1, 5, 1)
    last_row = np.repeat(last_row, poses.shape[2], axis=2)
    poses = np.concatenate((poses, last_row), axis=0)

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ""

    if factor is not None:
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if poses.shape[-1] != len(imgfiles):
        print("Mismatch between imgs {} and poses {} !!!!".format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if not load_imgs:
        return poses, bounds

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print("Loaded image data", imgs.shape, poses[:, -1, 0])
    return poses, bounds, imgs


if __name__ == "__main__":
    # M: test load_npy_data
    data_path = "C:\github\CustomNeRF\data\horns"
    data = load_npy_data(data_path)
    print(data)
