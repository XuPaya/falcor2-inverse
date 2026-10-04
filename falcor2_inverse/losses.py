"""Image losses."""

import torch


def mse(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((image - target) ** 2)


def l1(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(image - target))


def relative_mse(image: torch.Tensor, target: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Squared errors relative to the target's brightness, for high-dynamic-range images."""
    return torch.mean((image - target) ** 2 / (target.detach() ** 2 + eps))
