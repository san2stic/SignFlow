"""GPU utilities for model training and inference."""

from __future__ import annotations

import contextlib
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.ml.gpu_manager import GPUManager, get_gpu_manager

logger = structlog.get_logger(__name__)


def move_to_device(
    obj: Any,
    device: torch.device | str,
    non_blocking: bool = True,
) -> Any:
    """
    Move tensor, model, or collection to device.

    Args:
        obj: Object to move (tensor, model, dict, list, etc.)
        device: Target device
        non_blocking: Use non-blocking transfer (faster with pinned memory)

    Returns:
        Object moved to device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in obj]
        return type(obj)(moved)

    # Return as-is for non-tensor types
    return obj


@contextlib.contextmanager
def gpu_memory_guard(gpu_manager: GPUManager | None = None):
    """
    Context manager to monitor and clear GPU memory.

    Usage:
        with gpu_memory_guard():
            # GPU operations here
            pass

    Args:
        gpu_manager: GPU manager instance. If None, uses global instance.
    """
    manager = gpu_manager or get_gpu_manager()

    # Log initial memory
    initial_stats = manager.get_memory_stats()
    logger.debug("gpu_memory_guard_start", stats=initial_stats)

    try:
        yield manager
    finally:
        # Clear cache and log final memory
        manager.empty_cache()
        final_stats = manager.get_memory_stats()
        logger.debug("gpu_memory_guard_end", stats=final_stats)


def optimize_model_for_inference(
    model: nn.Module,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Optimize model for inference.

    Applies:
    - Set to inference mode
    - Disable gradient computation
    - Move to device
    - Enable inference optimizations

    Args:
        model: PyTorch model
        device: Target device. If None, uses best available.

    Returns:
        Optimized model
    """
    if device is None:
        device = get_gpu_manager().get_device()

    # Set inference mode
    model.eval()

    # Move to device
    model = model.to(device)

    # Disable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable inference mode optimizations
    if hasattr(model, "set_inference_mode"):
        model.set_inference_mode()

    logger.debug(
        "model_optimized_for_inference",
        device=str(device),
        num_params=sum(p.numel() for p in model.parameters()),
    )

    return model


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | None = None,
    max_memory_fraction: float = 0.8,
    min_batch_size: int = 1,
    max_batch_size: int = 512,
) -> int:
    """
    Estimate optimal batch size for given model and input shape.

    Uses binary search to find largest batch size that fits in memory.

    Args:
        model: PyTorch model
        input_shape: Input shape (seq_len, features)
        device: Target device
        max_memory_fraction: Maximum GPU memory fraction to use
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try

    Returns:
        Optimal batch size
    """
    if device is None:
        device = get_gpu_manager().get_device()

    if device.type == "cpu":
        # For CPU, use reasonable default
        return 32

    # Move model to device
    model = model.to(device)
    model.eval()

    def try_batch_size(batch_size: int) -> bool:
        """Try running forward pass with given batch size."""
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=device)

            with torch.no_grad():
                _ = model(dummy_input)

            # Check memory usage
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory
                fraction = allocated / total
                return fraction <= max_memory_fraction

            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise
        finally:
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Binary search for optimal batch size
    left, right = min_batch_size, max_batch_size
    optimal = min_batch_size

    while left <= right:
        mid = (left + right) // 2

        if try_batch_size(mid):
            optimal = mid
            left = mid + 1
        else:
            right = mid - 1

    logger.info(
        "optimal_batch_size_found",
        batch_size=optimal,
        device=str(device),
        input_shape=input_shape,
    )

    return optimal


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Profile model performance.

    Args:
        model: PyTorch model
        input_shape: Input shape (seq_len, features)
        batch_size: Batch size for profiling
        num_iterations: Number of iterations to profile
        warmup_iterations: Number of warmup iterations
        device: Target device

    Returns:
        dict: Profiling results
    """
    if device is None:
        device = get_gpu_manager().get_device()

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Profile
    import time

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

    import numpy as np

    times_array = np.array(times)
    results = {
        "device": str(device),
        "batch_size": batch_size,
        "input_shape": input_shape,
        "num_iterations": num_iterations,
        "mean_time_ms": float(times_array.mean() * 1000),
        "std_time_ms": float(times_array.std() * 1000),
        "min_time_ms": float(times_array.min() * 1000),
        "max_time_ms": float(times_array.max() * 1000),
        "throughput_samples_per_sec": float(batch_size / times_array.mean()),
    }

    logger.info("model_profiling_complete", **results)

    return results


def get_device_info() -> dict[str, Any]:
    """
    Get comprehensive device information.

    Returns:
        dict: Device information
    """
    manager = get_gpu_manager()
    device = manager.get_device()

    info = {
        "device": str(device),
        "device_type": device.type,
        "supports_amp": manager.supports_amp(),
    }

    if device.type == "cuda":
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "num_gpus": torch.cuda.device_count(),
            }
        )

        gpu_info = manager.get_gpu_info()
        if gpu_info:
            info.update(
                {
                    "gpu_name": gpu_info.name,
                    "total_memory_gb": round(gpu_info.total_memory_mb / 1024, 2),
                    "compute_capability": gpu_info.compute_capability,
                }
            )

    elif device.type == "mps":
        info["mps_available"] = True

    return info


def enable_deterministic_mode() -> None:
    """
    Enable deterministic mode for reproducibility.

    Note: May reduce performance.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("deterministic_mode_enabled")


def disable_deterministic_mode() -> None:
    """Disable deterministic mode for better performance."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logger.info("deterministic_mode_disabled")
