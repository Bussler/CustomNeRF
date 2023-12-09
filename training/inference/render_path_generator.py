import numpy as np
import torch


def normalize(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)

    # extend m with (0, 0, 0, 1) vector
    vector = np.array([0.0, 0.0, 0.0, 1.0])
    m = np.concatenate((m, vector[None, :]), axis=0)
    return m


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)

    return c2w


def generate_circular_renderpath(
    poses: list[torch.Tensor], focal: float, N_views: int = 120, N_rots: int = 2, zrate: float = 0.5, scale=1.0
) -> list[torch.Tensor]:
    """
    Generate a circular path around the object

    Args:
        poses (torch.Tensor): N x 3 x 4, representing input poses with positions
        focal (float): focal length of camera
        N_views (int, optional): Number of views to generate. Defaults to 120.
        N_rots (int, optional): Number of rotations on path. Defaults to 2.
        zrate (float, optional): Rate of change along z-coordinate. Defaults to 0.5.
        scale (float, optional): scaling factor for radius of rotation. Defaults to 1.0.

    Returns:
        list[torch.Tensor]: render poses to render the object from (c2w matrices)
    """
    c2w = poses_avg(poses)

    up = normalize(poses[:, :3, 1].sum(0))

    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0) * scale

    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N_views + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        pose = viewmatrix(z, up, c)
        pose = torch.from_numpy(pose).to(poses.device)
        pose = pose.type(torch.float32)
        render_poses.append(pose)
    return render_poses
