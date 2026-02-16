#!/usr/bin/env python3
"""Test script for GPU support functionality."""

import sys
from pathlib import Path

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

import torch
import numpy as np

from app.ml.gpu_manager import GPUManager, GPUConfig, get_gpu_manager
from app.ml.gpu_utils import (
    get_device_info,
    move_to_device,
    gpu_memory_guard,
    optimize_model_for_inference,
    profile_model,
)
from app.ml.model import SignTransformer


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def test_gpu_detection() -> None:
    """Test GPU detection and setup."""
    print_section("GPU Detection")

    # Test auto detection
    config = GPUConfig(device="auto")
    manager = GPUManager(config)
    device = manager.setup_device()

    print(f"✓ Device detected: {device}")
    print(f"✓ Device type: {device.type}")

    # Get GPU info
    gpu_info = manager.get_gpu_info()
    if gpu_info:
        print(f"\nGPU Information:")
        print(f"  Name: {gpu_info.name}")
        print(f"  Type: {gpu_info.device_type.value}")
        print(f"  Device Index: {gpu_info.device_index}")

        if gpu_info.total_memory_mb > 0:
            print(
                f"  Total Memory: {gpu_info.total_memory_mb / 1024:.2f} GB"
            )
            print(
                f"  Available Memory: {gpu_info.available_memory_mb / 1024:.2f} GB"
            )

        if gpu_info.compute_capability:
            print(
                f"  Compute Capability: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}"
            )
    else:
        print("\nCPU mode (no GPU available)")

    # Test AMP support
    supports_amp = manager.supports_amp()
    print(f"\n✓ AMP Support: {supports_amp}")
    if supports_amp:
        amp_dtype = manager.get_amp_dtype()
        print(f"  AMP dtype: {amp_dtype}")


def test_device_info() -> None:
    """Test device information retrieval."""
    print_section("Device Information")

    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def test_memory_management() -> None:
    """Test memory management utilities."""
    print_section("Memory Management")

    manager = get_gpu_manager()

    # Initial memory stats
    print("Initial Memory Stats:")
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test memory guard
    print("\nTesting memory guard...")
    with gpu_memory_guard(manager):
        # Allocate some tensors
        device = manager.get_device()
        tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(tensor, tensor.T)
        print(f"  Allocated tensor shape: {result.shape}")

    print("✓ Memory guard completed")

    # Final memory stats
    print("\nFinal Memory Stats:")
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_model_optimization() -> None:
    """Test model optimization for inference."""
    print_section("Model Optimization")

    # Create a small test model
    model = SignTransformer(
        num_features=469,
        num_classes=10,
        d_model=128,
        nhead=4,
        num_layers=2,
    )

    print(f"Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimize for inference
    manager = get_gpu_manager()
    device = manager.get_device()

    model = optimize_model_for_inference(model, device)
    print(f"✓ Model optimized and moved to {device}")

    # Test inference
    dummy_input = torch.randn(4, 64, 469, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Inference successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")


def test_move_to_device() -> None:
    """Test moving objects to device."""
    print_section("Move to Device")

    device = get_gpu_manager().get_device()

    # Test tensor
    tensor = torch.randn(10, 10)
    tensor_moved = move_to_device(tensor, device)
    print(f"✓ Tensor moved to {tensor_moved.device}")

    # Test dict
    data_dict = {
        "input": torch.randn(5, 5),
        "target": torch.randn(5),
    }
    dict_moved = move_to_device(data_dict, device)
    print(f"✓ Dict moved: input device={dict_moved['input'].device}")

    # Test list
    data_list = [torch.randn(3, 3) for _ in range(3)]
    list_moved = move_to_device(data_list, device)
    print(f"✓ List moved: first item device={list_moved[0].device}")


def test_model_profiling() -> None:
    """Test model profiling."""
    print_section("Model Profiling")

    # Create test model
    model = SignTransformer(
        num_features=469,
        num_classes=10,
        d_model=128,
        nhead=4,
        num_layers=2,
    )

    # Profile with different batch sizes
    for batch_size in [1, 4, 8]:
        print(f"\nProfiling batch_size={batch_size}...")
        results = profile_model(
            model=model,
            input_shape=(64, 469),
            batch_size=batch_size,
            num_iterations=50,
            warmup_iterations=5,
        )

        print(f"  Mean time: {results['mean_time_ms']:.2f} ms")
        print(f"  Std time: {results['std_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")


def test_cuda_specific() -> None:
    """Test CUDA-specific features."""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping CUDA-specific tests")
        return

    print_section("CUDA-Specific Tests")

    # Test multi-GPU detection
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  Name: {props.name}")
        print(f"  Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    # Test memory operations
    print("\nCUDA Memory Operations:")
    device = torch.device("cuda:0")

    # Allocate
    tensor = torch.randn(10000, 10000, device=device)
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"  Allocated: {allocated:.2f} MB")

    # Free
    del tensor
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"  After free: {allocated:.2f} MB")


def test_mps_specific() -> None:
    """Test MPS-specific features (Apple Silicon)."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("\nMPS not available, skipping MPS-specific tests")
        return

    print_section("MPS-Specific Tests")

    device = torch.device("mps")
    print(f"MPS device available: {device}")

    # Test basic operations
    tensor = torch.randn(1000, 1000, device=device)
    result = torch.matmul(tensor, tensor.T)
    print(f"✓ Matrix multiplication successful: {result.shape}")

    # Test model inference
    model = SignTransformer(
        num_features=469,
        num_classes=10,
        d_model=128,
        nhead=4,
        num_layers=2,
    ).to(device)

    dummy_input = torch.randn(4, 64, 469, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Model inference successful: {output.shape}")


def run_all_tests() -> None:
    """Run all GPU tests."""
    print("\n" + "=" * 70)
    print("  GPU SUPPORT TEST SUITE")
    print("=" * 70)

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Device Information", test_device_info),
        ("Memory Management", test_memory_management),
        ("Model Optimization", test_model_optimization),
        ("Move to Device", test_move_to_device),
        ("Model Profiling", test_model_profiling),
        ("CUDA-Specific", test_cuda_specific),
        ("MPS-Specific", test_mps_specific),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"  TEST SUMMARY")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
