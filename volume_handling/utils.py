import os

import imageio
import numpy as np


# TODO M: sync with other data loading functions
def load_npy_data(data_path: str) -> None:
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
    poses = poses[:, :3, :4]

    H, W, focal = hwf
    H, W = int(H), int(W)

    near = np.ndarray.min(bounds) * 0.9
    far = np.ndarray.max(bounds) * 1.0

    return {"poses": poses, "bounds": bounds, "images": imgs}


########## Slightly modified version of LLFF data loading code
########## See https://github.com/Fyusion/LLFF for original
########## Adapted from DSNERF https://github.com/dunbar12138/DSNeRF


def _minify(basedir, factors=[], resolutions=[]):
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

    from shutil import copy
    from subprocess import check_output

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


# TODO refactor
# factor: downsample factor
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3 x 5 x N
    bounds = poses_arr[:, -2:].transpose([1, 0])

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
    # TODO M: test load_npy_data
    data_path = "C:\github\CustomNeRF\data\horns"
    data = load_npy_data(data_path)
    print(data)
