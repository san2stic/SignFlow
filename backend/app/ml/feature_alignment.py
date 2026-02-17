"""Utilities to align runtime feature tensors with model input dimension."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def resolve_model_feature_dim(model: Any) -> int | None:
    """Resolve expected input feature dimension from a model-like object."""
    candidate = getattr(model, "num_features", None)
    if candidate is None and hasattr(model, "module"):
        candidate = getattr(model.module, "num_features", None)
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def align_numpy_features(array: np.ndarray, target_dim: int | None) -> np.ndarray:
    """Crop/pad the last dimension of a numpy tensor to the requested size."""
    if target_dim is None or target_dim <= 0:
        return array
    if array.ndim == 0:
        return array

    current_dim = int(array.shape[-1])
    if current_dim == target_dim:
        return array
    if current_dim > target_dim:
        return array[..., :target_dim]

    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, target_dim - current_dim)
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=0.0)


def align_torch_features(tensor: torch.Tensor, target_dim: int | None) -> torch.Tensor:
    """Crop/pad the last dimension of a tensor to the requested size."""
    if target_dim is None or target_dim <= 0:
        return tensor
    if tensor.ndim == 0:
        return tensor

    current_dim = int(tensor.shape[-1])
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[..., :target_dim]

    pad_shape = (*tensor.shape[:-1], target_dim - current_dim)
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)
