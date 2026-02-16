"""Model export utilities for ONNX and other formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)

# Optional ONNX import
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnx_not_installed", msg="ONNX not installed, export disabled")


def export_to_onnx(
    model: nn.Module,
    save_path: str | Path,
    *,
    input_shape: tuple[int, int, int] = (1, 64, 469),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    verify: bool = True,
) -> bool:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        input_shape: Shape of input tensor (batch, seq_len, features)
        opset_version: ONNX opset version
        dynamic_axes: Optional dynamic axes specification
        verify: Whether to verify the exported model

    Returns:
        True if export successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("onnx_export_failed", reason="onnx_not_installed")
        return False

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Set model to evaluation mode (not arbitrary code execution)
        model.train(False)  # Explicitly set to eval mode

        # Create dummy input
        dummy_input = torch.randn(*input_shape, dtype=torch.float32)

        # Default dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                "landmarks": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch"}
            }

        logger.info(
            "onnx_export_starting",
            save_path=str(save_path),
            input_shape=input_shape,
            opset=opset_version
        )

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["landmarks"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        logger.info("onnx_export_complete", save_path=str(save_path))

        # Verify exported model
        if verify:
            is_valid = verify_onnx_model(save_path, dummy_input, model)
            if not is_valid:
                logger.warning("onnx_verification_failed", save_path=str(save_path))
                return False

        return True

    except Exception as e:
        logger.error("onnx_export_failed", error=str(e), save_path=str(save_path))
        return False


def verify_onnx_model(
    onnx_path: str | Path,
    test_input: torch.Tensor,
    original_model: nn.Module,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model produces same output as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        original_model: Original PyTorch model
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if models match within tolerance, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.warning("onnx_verification_skipped", reason="onnx_not_installed")
        return False

    try:
        # Check ONNX model validity
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.debug("onnx_model_valid", path=str(onnx_path))

        # Create ONNX Runtime session
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

        # Run ONNX inference
        onnx_input = test_input.numpy()
        onnx_outputs = session.run(None, {"landmarks": onnx_input})
        onnx_logits = onnx_outputs[0]

        # Run PyTorch inference
        original_model.train(False)
        with torch.no_grad():
            pytorch_logits = original_model(test_input).numpy()

        # Compare outputs
        import numpy as np
        matches = np.allclose(pytorch_logits, onnx_logits, rtol=rtol, atol=atol)

        if matches:
            max_diff = np.abs(pytorch_logits - onnx_logits).max()
            logger.info(
                "onnx_verification_passed",
                max_diff=float(max_diff),
                rtol=rtol,
                atol=atol
            )
        else:
            max_diff = np.abs(pytorch_logits - onnx_logits).max()
            logger.error(
                "onnx_verification_failed",
                max_diff=float(max_diff),
                rtol=rtol,
                atol=atol
            )

        return matches

    except Exception as e:
        logger.error("onnx_verification_error", error=str(e))
        return False


def optimize_onnx_model(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
) -> bool:
    """
    Optimize ONNX model for inference (constant folding, shape inference, etc.).

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save optimized model (defaults to overwrite input)

    Returns:
        True if optimization successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("onnx_optimization_failed", reason="onnx_not_installed")
        return False

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        model = onnx.load(str(onnx_path))

        # Run optimizations
        from onnx import optimizer

        # Basic optimizations
        passes = [
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]

        optimized_model = optimizer.optimize(model, passes)

        # Save optimized model
        onnx.save(optimized_model, str(output_path))

        logger.info(
            "onnx_optimization_complete",
            input_path=str(onnx_path),
            output_path=str(output_path)
        )

        return True

    except Exception as e:
        logger.error("onnx_optimization_failed", error=str(e))
        return False


def export_model_metadata(
    model: nn.Module,
    save_path: str | Path,
    class_labels: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Export model metadata to JSON file.

    Useful for storing class labels, model config, calibration parameters, etc.

    Args:
        model: PyTorch model
        save_path: Path to save metadata JSON
        class_labels: Optional list of class labels
        extra_metadata: Additional metadata to include
    """
    import json

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_class": model.__class__.__name__,
        "num_classes": getattr(model, "num_classes", None),
        "num_features": getattr(model, "num_features", None),
        "d_model": getattr(model, "d_model", None),
        "num_layers": getattr(model, "num_layers", None),
        "nhead": getattr(model, "nhead", None),
        "class_labels": class_labels or [],
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("model_metadata_exported", save_path=str(save_path))


def get_onnx_model_info(onnx_path: str | Path) -> dict[str, Any]:
    """
    Get information about an ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with model information
    """
    if not ONNX_AVAILABLE:
        return {"error": "ONNX not installed"}

    try:
        model = onnx.load(str(onnx_path))

        # Get input/output shapes
        inputs = []
        for inp in model.graph.input:
            shape = [
                dim.dim_value if dim.dim_value > 0 else str(dim.dim_param)
                for dim in inp.type.tensor_type.shape.dim
            ]
            inputs.append({
                "name": inp.name,
                "shape": shape,
            })

        outputs = []
        for out in model.graph.output:
            shape = [
                dim.dim_value if dim.dim_value > 0 else str(dim.dim_param)
                for dim in out.type.tensor_type.shape.dim
            ]
            outputs.append({
                "name": out.name,
                "shape": shape,
            })

        return {
            "opset_version": model.opset_import[0].version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "inputs": inputs,
            "outputs": outputs,
            "num_nodes": len(model.graph.node),
        }

    except Exception as e:
        return {"error": str(e)}
