"""GPU manager for optimized device selection and memory management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


class DeviceType(Enum):
    """Available device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    ROCM = "rocm"  # AMD ROCm


@dataclass
class GPUInfo:
    """Information about available GPU."""

    device_type: DeviceType
    device_index: int
    name: str
    total_memory_mb: float
    available_memory_mb: float
    compute_capability: tuple[int, int] | None = None
    is_available: bool = True


@dataclass
class GPUConfig:
    """GPU configuration settings."""

    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", "mps"
    allow_tf32: bool = True  # Enable TF32 for faster training on Ampere+
    cudnn_benchmark: bool = True  # Enable cuDNN auto-tuner
    cudnn_deterministic: bool = False  # Deterministic mode (slower)
    memory_fraction: float = 0.9  # Max GPU memory fraction to use
    empty_cache_interval: int = 100  # Empty cache every N batches
    enable_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "float16"  # "float16" or "bfloat16"


class GPUManager:
    """Manage GPU detection, selection, and memory optimization."""

    def __init__(self, config: GPUConfig | None = None) -> None:
        """
        Initialize GPU manager.

        Args:
            config: GPU configuration. If None, uses default config.
        """
        self.config = config or GPUConfig()
        self._device: torch.device | None = None
        self._gpu_info: GPUInfo | None = None
        self._batch_counter = 0

    def detect_best_device(self) -> torch.device:
        """
        Detect and return the best available device.

        Priority:
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple Silicon)
        3. CPU (fallback)

        Returns:
            torch.device: Best available device
        """
        if self.config.device != "auto":
            device_str = self.config.device
            device = torch.device(device_str)
            logger.info("gpu_device_specified", device=device_str)
            return device

        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            device = self._select_cuda_device()
            logger.info(
                "cuda_device_selected",
                device=str(device),
                gpu_count=torch.cuda.device_count(),
            )
            return device

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("mps_device_selected", device="mps")
            return device

        # Fallback to CPU
        device = torch.device("cpu")
        logger.info("cpu_device_selected", reason="no_gpu_available")
        return device

    def _select_cuda_device(self) -> torch.device:
        """
        Select best CUDA device based on available memory.

        Returns:
            torch.device: Selected CUDA device
        """
        if torch.cuda.device_count() == 1:
            return torch.device("cuda:0")

        # Select GPU with most available memory
        best_device = 0
        max_memory = 0.0

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                available = total_memory - allocated

                logger.debug(
                    "cuda_device_info",
                    device_id=i,
                    name=props.name,
                    total_memory_gb=round(total_memory, 2),
                    available_memory_gb=round(available, 2),
                )

                if available > max_memory:
                    max_memory = available
                    best_device = i

            except Exception as e:
                logger.warning(
                    "cuda_device_check_failed",
                    device_id=i,
                    error=str(e),
                )

        return torch.device(f"cuda:{best_device}")

    def setup_device(self) -> torch.device:
        """
        Setup and configure the best available device.

        Returns:
            torch.device: Configured device
        """
        if self._device is not None:
            return self._device

        self._device = self.detect_best_device()
        self._configure_device()
        self._gpu_info = self._collect_gpu_info()

        logger.info(
            "gpu_setup_complete",
            device=str(self._device),
            device_type=self._device.type,
            gpu_info=self._format_gpu_info(),
        )

        return self._device

    def _configure_device(self) -> None:
        """Configure device-specific optimizations."""
        if self._device is None:
            return

        if self._device.type == "cuda":
            self._configure_cuda()
        elif self._device.type == "mps":
            self._configure_mps()

    def _configure_cuda(self) -> None:
        """Configure CUDA-specific optimizations."""
        # Enable TF32 for Ampere+ GPUs (faster matmul)
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("cuda_tf32_enabled")

        # Enable cuDNN auto-tuner for faster convolutions
        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.debug("cudnn_benchmark_enabled")

        # Set deterministic mode if requested
        if self.config.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("cudnn_deterministic_enabled")

        # Set memory allocation strategy
        if self.config.memory_fraction < 1.0:
            try:
                # Note: This is a soft limit, PyTorch may exceed it if needed
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"max_split_size_mb=512,expandable_segments:True"
                )
                logger.debug(
                    "cuda_memory_config",
                    memory_fraction=self.config.memory_fraction,
                )
            except Exception as e:
                logger.warning("cuda_memory_config_failed", error=str(e))

    def _configure_mps(self) -> None:
        """Configure MPS-specific optimizations."""
        # MPS is relatively new, fewer configurations available
        logger.debug("mps_configured")

    def _collect_gpu_info(self) -> GPUInfo | None:
        """
        Collect information about the current GPU.

        Returns:
            GPUInfo | None: GPU information or None if not GPU device
        """
        if self._device is None or self._device.type == "cpu":
            return None

        try:
            if self._device.type == "cuda":
                return self._collect_cuda_info()
            elif self._device.type == "mps":
                return self._collect_mps_info()
        except Exception as e:
            logger.warning("gpu_info_collection_failed", error=str(e))

        return None

    def _collect_cuda_info(self) -> GPUInfo:
        """Collect CUDA GPU information."""
        device_index = self._device.index or 0
        props = torch.cuda.get_device_properties(device_index)
        total_memory = props.total_memory / (1024**2)  # MB
        allocated = torch.cuda.memory_allocated(device_index) / (1024**2)
        available = total_memory - allocated

        return GPUInfo(
            device_type=DeviceType.CUDA,
            device_index=device_index,
            name=props.name,
            total_memory_mb=total_memory,
            available_memory_mb=available,
            compute_capability=(props.major, props.minor),
            is_available=True,
        )

    def _collect_mps_info(self) -> GPUInfo:
        """Collect MPS GPU information."""
        # MPS doesn't expose detailed memory info yet
        return GPUInfo(
            device_type=DeviceType.MPS,
            device_index=0,
            name="Apple Silicon GPU",
            total_memory_mb=0.0,  # Not available
            available_memory_mb=0.0,  # Not available
            compute_capability=None,
            is_available=True,
        )

    def _format_gpu_info(self) -> dict[str, Any]:
        """Format GPU info for logging."""
        if self._gpu_info is None:
            return {"type": "cpu"}

        info = {
            "type": self._gpu_info.device_type.value,
            "name": self._gpu_info.name,
            "device_index": self._gpu_info.device_index,
        }

        if self._gpu_info.total_memory_mb > 0:
            info["total_memory_gb"] = round(self._gpu_info.total_memory_mb / 1024, 2)
            info["available_memory_gb"] = round(
                self._gpu_info.available_memory_mb / 1024, 2
            )

        if self._gpu_info.compute_capability is not None:
            info["compute_capability"] = (
                f"{self._gpu_info.compute_capability[0]}.{self._gpu_info.compute_capability[1]}"
            )

        return info

    def get_device(self) -> torch.device:
        """
        Get the configured device.

        Returns:
            torch.device: Configured device
        """
        if self._device is None:
            return self.setup_device()
        return self._device

    def get_gpu_info(self) -> GPUInfo | None:
        """
        Get GPU information.

        Returns:
            GPUInfo | None: GPU info or None if CPU device
        """
        if self._device is None:
            self.setup_device()
        return self._gpu_info

    def on_batch_end(self) -> None:
        """
        Hook to call at the end of each batch.

        Handles periodic cache clearing to prevent memory fragmentation.
        """
        self._batch_counter += 1

        if (
            self.config.empty_cache_interval > 0
            and self._batch_counter % self.config.empty_cache_interval == 0
        ):
            self.empty_cache()

    def empty_cache(self) -> None:
        """Empty GPU cache to free up memory."""
        if self._device is None:
            return

        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("cuda_cache_cleared")
        elif self._device.type == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                logger.debug("mps_cache_cleared")

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            dict: Memory statistics
        """
        if self._device is None or self._device.type == "cpu":
            return {"device": "cpu", "memory_info": "not_available"}

        stats: dict[str, Any] = {"device": str(self._device)}

        if self._device.type == "cuda":
            device_index = self._device.index or 0
            allocated = torch.cuda.memory_allocated(device_index) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_index) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(device_index) / (1024**2)

            stats.update(
                {
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                    "max_allocated_mb": round(max_allocated, 2),
                }
            )

        return stats

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device.index or 0)
            logger.debug("cuda_peak_memory_stats_reset")

    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self._device is None:
            return

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elif self._device.type == "mps":
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    def supports_amp(self) -> bool:
        """
        Check if device supports Automatic Mixed Precision.

        Returns:
            bool: True if AMP is supported
        """
        if self._device is None:
            self.setup_device()

        if not self.config.enable_amp:
            return False

        # CUDA supports AMP
        if self._device.type == "cuda":
            # Check compute capability for BF16 support
            if self.config.amp_dtype == "bfloat16":
                if self._gpu_info and self._gpu_info.compute_capability:
                    major, _ = self._gpu_info.compute_capability
                    return major >= 8  # Ampere+ for BF16
            return True  # FP16 supported on all CUDA devices

        # MPS supports AMP since PyTorch 2.0
        if self._device.type == "mps":
            return True

        return False

    def get_amp_dtype(self) -> torch.dtype | None:
        """
        Get appropriate AMP dtype for the device.

        Returns:
            torch.dtype | None: AMP dtype or None if not supported
        """
        if not self.supports_amp():
            return None

        if self.config.amp_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16


# Global GPU manager instance
_global_gpu_manager: GPUManager | None = None


def get_gpu_manager(config: GPUConfig | None = None) -> GPUManager:
    """
    Get or create global GPU manager instance.

    Args:
        config: GPU configuration. Only used on first call.

    Returns:
        GPUManager: Global GPU manager instance
    """
    global _global_gpu_manager

    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager(config)
        _global_gpu_manager.setup_device()

    return _global_gpu_manager


def reset_gpu_manager() -> None:
    """Reset global GPU manager instance."""
    global _global_gpu_manager
    _global_gpu_manager = None
