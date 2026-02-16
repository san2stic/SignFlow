#!/bin/bash
set -e

echo "🚀 Starting TorchServe with device auto-detection..."

# Détection du device disponible
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    NUM_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "✅ CUDA detected: $NUM_GPU GPU(s) available"
elif python3 -c "import torch; exit(0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "✅ MPS (Apple Silicon GPU) detected"
else
    echo "⚠️  No GPU detected, using CPU"
fi

# Export du device pour les handlers
export TORCH_DEVICE="$DEVICE"

# Configuration TorchServe selon le device
if [ "$DEVICE" = "cuda" ]; then
    TS_NUMBER_OF_GPU=${TS_NUMBER_OF_GPU:-$NUM_GPU}
    echo "🔧 Configuring TorchServe with $TS_NUMBER_OF_GPU CUDA GPU(s)"
    exec torchserve \
        --start \
        --model-store /home/model-server/model-store \
        --ts-config /home/model-server/config/config.properties \
        --ncs
elif [ "$DEVICE" = "mps" ]; then
    echo "🔧 Configuring TorchServe with MPS (Apple Silicon GPU)"
    # MPS n'est pas supporté nativement par TorchServe, on utilise CPU avec note
    echo "⚠️  Note: TorchServe handlers will use MPS via PyTorch directly"
    exec torchserve \
        --start \
        --model-store /home/model-server/model-store \
        --ts-config /home/model-server/config/config.properties \
        --ncs
else
    echo "🔧 Configuring TorchServe with CPU"
    exec torchserve \
        --start \
        --model-store /home/model-server/model-store \
        --ts-config /home/model-server/config/config.properties \
        --ncs
fi
