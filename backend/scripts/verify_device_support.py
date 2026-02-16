#!/usr/bin/env python3
"""
Script de vérification du support multi-device pour TorchServe.
Teste CPU, MPS (Apple Silicon), et CUDA.
"""
import sys
import torch
import platform
from typing import Dict, List

def check_pytorch_install() -> Dict[str, any]:
    """Vérifie l'installation PyTorch."""
    return {
        "version": torch.__version__,
        "cuda_built": torch.cuda.is_available(),
        "mps_built": hasattr(torch.backends, 'mps') and torch.backends.mps.is_built(),
    }

def detect_devices() -> List[str]:
    """Détecte tous les devices disponibles."""
    devices = ["cpu"]  # CPU toujours disponible

    # Check CUDA
    if torch.cuda.is_available():
        devices.append("cuda")

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")

    return devices

def test_device_inference(device: str) -> Dict[str, any]:
    """Teste l'inférence sur un device."""
    try:
        # Créer un petit modèle test
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )

        # Déplacer sur device
        model = model.to(device)
        model.training = False

        # Test input
        x = torch.randn(4, 10).to(device)

        # Inférence
        import time
        start = time.time()
        with torch.no_grad():
            output = model(x)
        latency_ms = (time.time() - start) * 1000

        return {
            "success": True,
            "output_shape": list(output.shape),
            "latency_ms": round(latency_ms, 2),
            "device": str(output.device)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def check_onnx_runtime():
    """Vérifie le support ONNX Runtime."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return {
            "installed": True,
            "version": ort.__version__,
            "providers": providers,
            "cuda_support": "CUDAExecutionProvider" in providers
        }
    except ImportError:
        return {
            "installed": False,
            "error": "onnxruntime not installed"
        }

def print_section(title: str):
    """Print une section formatée."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(label: str, value: any, status: str = None):
    """Print un résultat formaté."""
    status_icon = {
        "success": "✅",
        "warning": "⚠️ ",
        "error": "❌",
        None: "ℹ️ "
    }.get(status, "")

    print(f"{status_icon} {label:30} {value}")

def main():
    print("🔍 TorchServe Multi-Device Support Verification")

    # 1. Système
    print_section("System Information")
    print_result("Platform", platform.platform())
    print_result("Architecture", platform.machine())
    print_result("Python", sys.version.split()[0])

    # 2. PyTorch
    print_section("PyTorch Installation")
    pytorch_info = check_pytorch_install()
    print_result("PyTorch Version", pytorch_info["version"])
    print_result("CUDA Built", pytorch_info["cuda_built"],
                 "success" if pytorch_info["cuda_built"] else "warning")
    print_result("MPS Built", pytorch_info["mps_built"],
                 "success" if pytorch_info["mps_built"] else "warning")

    # 3. Devices disponibles
    print_section("Available Devices")
    devices = detect_devices()
    print_result("Detected Devices", ", ".join(devices))

    # Détails par device
    if "cuda" in devices:
        print_result("CUDA Device", torch.cuda.get_device_name(0), "success")
        print_result("CUDA Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if "mps" in devices:
        print_result("MPS Device", "Apple Silicon GPU", "success")

    print_result("CPU Cores", torch.get_num_threads())

    # 4. Test d'inférence
    print_section("Inference Tests")
    for device in devices:
        print(f"\n🧪 Testing {device.upper()}...")
        result = test_device_inference(device)

        if result["success"]:
            print_result(f"{device.upper()} Inference",
                        f"{result['latency_ms']:.2f}ms", "success")
            print_result(f"{device.upper()} Output Shape",
                        result["output_shape"])
        else:
            print_result(f"{device.upper()} Inference",
                        result["error"], "error")

    # 5. ONNX Runtime
    print_section("ONNX Runtime")
    onnx_info = check_onnx_runtime()

    if onnx_info["installed"]:
        print_result("ONNX Runtime", onnx_info["version"], "success")
        print_result("CUDA Provider",
                    onnx_info["cuda_support"],
                    "success" if onnx_info["cuda_support"] else "warning")
        print_result("Available Providers",
                    ", ".join(onnx_info["providers"][:3]))
    else:
        print_result("ONNX Runtime", "Not installed", "warning")
        print("   Install with: pip install onnxruntime")

    # 6. Recommandations
    print_section("Recommendations")

    if "cuda" in devices:
        print("✅ CUDA detected → Use docker-compose.gpu.yml for production")
        print("   Expected latency: 5-15ms per inference")
    elif "mps" in devices:
        print("✅ MPS detected → Use docker-compose.arm64.yml for development")
        print("   Expected latency: 10-30ms per inference")
    else:
        print("⚠️  CPU only → Use docker-compose.cpu.yml")
        print("   Consider ONNX export for 2-5x speedup:")
        print("   python backend/app/ml/export.py --optimize")

    if not onnx_info["installed"]:
        print("\n💡 Install ONNX Runtime for better CPU performance:")
        print("   pip install onnxruntime  # CPU")
        if "cuda" in devices:
            print("   pip install onnxruntime-gpu  # GPU support")

    print("\n" + "="*60)
    print("✅ Verification complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
