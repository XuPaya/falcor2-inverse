"""Rotations as axis-angle vectors (radians, the rotation axis times the angle), in PyTorch."""

import numpy as np
import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    """Cross-product matrices [v]x of vectors (..., 3)."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], zero, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], zero], -1)], -2)


def axis_angle_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) of axis-angle vectors (..., 3) (Rodrigues' formula, smooth at zero)."""
    theta2 = (r * r).sum(-1, keepdim=True)[..., None]
    theta = torch.sqrt(theta2.clamp_min(1e-24))
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2.clamp_min(1e-24))
    k = skew(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(k.shape)
    return eye + a * k + b * (k @ k)


def quaternion_to_axis_angle(q) -> np.ndarray:
    """Axis-angle vector of a unit quaternion (x, y, z, w)."""
    q = np.asarray(q, np.float64)
    q = q * np.sign(q[3]) if q[3] != 0 else q
    sin = np.linalg.norm(q[:3])
    if sin < 1e-12:
        return 2 * q[:3]
    return q[:3] / sin * 2 * np.arctan2(sin, q[3])


def axis_angle_to_quaternion(r) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of an axis-angle vector."""
    r = np.asarray(r, np.float64)
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([r[0] / 2, r[1] / 2, r[2] / 2, 1.0])
    axis = r / angle
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def rotation_vector_of(rotation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """The small world-space rotation vector w with rotation ~ exp([w]x) reference, to first order."""
    e = rotation @ reference.transpose(-1, -2)
    return 0.5 * torch.stack([e[..., 2, 1] - e[..., 1, 2], e[..., 0, 2] - e[..., 2, 0], e[..., 1, 0] - e[..., 0, 1]], -1)
