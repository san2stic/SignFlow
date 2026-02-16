# TorchServe Multi-Device Setup

Configuration TorchServe pour supporter **CPU, MPS (Apple Silicon), et CUDA GPU** de manière transparente.

## 🚀 Démarrage Rapide

### Apple Silicon (M1/M2/M3) - MPS

```bash
# Build et démarrage
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up --build torchserve

# Le handler détectera automatiquement MPS
```

### x86_64 avec NVIDIA GPU - CUDA

```bash
# Prérequis: NVIDIA Container Toolkit installé
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build torchserve
```

### CPU uniquement (toutes plateformes)

```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up --build torchserve
```

## 📦 Architecture

```
backend/
├── Dockerfile.torchserve         # Build multi-platform
├── torchserve/
│   ├── start.sh                  # Détection device auto
│   ├── handlers/
│   │   └── sign_handler.py       # Handler multi-device
│   ├── model-store/              # Modèles .mar
│   └── config/
│       └── config.properties     # Config TorchServe
```

## 🔧 Détection Automatique du Device

Le script `start.sh` détecte automatiquement dans cet ordre :
1. **CUDA** : Si `torch.cuda.is_available()` → utilise GPU NVIDIA
2. **MPS** : Si `torch.backends.mps.is_available()` → utilise GPU Apple
3. **CPU** : Fallback si aucun GPU détecté

### Override manuel

```bash
# Forcer CPU
TORCH_DEVICE=cpu docker-compose up torchserve

# Forcer MPS
TORCH_DEVICE=mps docker-compose up torchserve

# Forcer CUDA
TORCH_DEVICE=cuda docker-compose up torchserve
```

## 📊 Formats de Modèles Supportés

### 1. PyTorch (.pt)
```bash
# TorchScript JIT
torch.jit.save(traced_model, "model.pt")

# Ou sauvegarde directe (nécessite architecture)
torch.save(model, "model.pt")
```

### 2. ONNX (.onnx)
```bash
# Export ONNX (recommandé pour performance CPU)
python backend/app/ml/export.py --model-path model.pt --output-path model.onnx
```

**Providers ONNX selon device :**
- **CUDA** : `CUDAExecutionProvider` → 2-3x plus rapide
- **CPU** : `CPUExecutionProvider` → baseline
- **MPS** : Pas de provider natif → fallback `CPUExecutionProvider`

## 🏗️ Créer un Model Archive (.mar)

```bash
# Depuis backend/
torch-model-archiver \
  --model-name signflow_model \
  --version 1.0 \
  --model-file app/ml/model.py \
  --serialized-file data/models/model.pt \
  --handler torchserve/handlers/sign_handler.py \
  --export-path torchserve/model-store \
  --extra-files "app/ml/feature_engineering.py,app/ml/model_configs.py" \
  --requirements-file requirements.txt
```

## 🧪 Test de l'Inférence

```bash
# Health check
curl http://localhost:8080/ping

# Lister les modèles
curl http://localhost:8081/models

# Inférence
curl -X POST http://localhost:8080/predictions/signflow_model \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [
      [[0.5, 0.5, 0.1], [0.6, 0.4, 0.15], ...],
      [[0.5, 0.6, 0.12], [0.61, 0.41, 0.16], ...]
    ]
  }'

# Réponse inclut le device utilisé
{
  "predictions": [
    {"label": "hello", "confidence": 0.95}
  ],
  "device": "mps"  # ou "cuda", "cpu"
}
```

## 📈 Métriques Prometheus

```bash
# Métriques d'inférence
curl http://localhost:8082/metrics

# Métriques clés :
# - ts_inference_latency_microseconds : latence par device
# - ts_queue_latency_microseconds : temps d'attente batch
# - ts_inference_requests_total : nombre de requêtes
```

## ⚙️ Configuration Performance

### Apple Silicon (MPS)
```yaml
# docker-compose.arm64.yml
deploy:
  resources:
    limits:
      memory: 6G    # MPS utilise unified memory
      cpus: '3.0'   # 3 cores suffisants
```

### NVIDIA GPU (CUDA)
```yaml
# docker-compose.gpu.yml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      devices:
        - driver: nvidia
          count: 1    # 1+ GPUs
```

### CPU uniquement
```yaml
# docker-compose.cpu.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

## 🐛 Troubleshooting

### "platform (linux/amd64) does not match (linux/arm64)"
→ Utilisez `docker-compose.arm64.yml` ou `docker-compose.cpu.yml`

### MPS détecté mais pas utilisé
```bash
# Vérifier support MPS dans container
docker exec signflow_torchserve python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

### CUDA non détecté
```bash
# Vérifier NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Vérifier dans container
docker exec signflow_torchserve python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

### Latence élevée sur CPU
→ Utilisez ONNX Runtime pour ~2-5x speedup :
```bash
python backend/app/ml/export.py --optimize
```

## 📚 Performance Attendue

| Device | Latence (ms) | Throughput (req/s) | Notes |
|--------|--------------|-------------------|-------|
| **CUDA GPU** | 5-15 ms | 200-500 | Optimal pour production |
| **Apple MPS** | 10-30 ms | 100-200 | Bon pour développement M1/M2 |
| **CPU (ONNX)** | 15-50 ms | 50-100 | Acceptable pour dev/test |
| **CPU (PyTorch)** | 40-120 ms | 20-40 | Fallback uniquement |

## 🔗 Intégration Backend FastAPI

Le backend détecte automatiquement si TorchServe est actif :

```python
# backend/app/ml/pipeline.py
if USE_TORCHSERVE:
    response = requests.post(
        f"{TORCHSERVE_URL}/predictions/signflow_model",
        json={"landmarks": landmarks}
    )
    return response.json()
else:
    # Fallback: PyTorch direct
    return pytorch_inference(landmarks)
```

## 📝 Prochaines Étapes

1. **Batching Asynchrone** : Activer `batch_size > 1` dans `config.properties`
2. **Model Versioning** : A/B testing avec `canary_percentage`
3. **Drift Detection** : Monitoring de distribution via Prometheus
4. **Auto-scaling** : Kubernetes HPA sur métriques latence

---

**Support** : MPS (PyTorch 2.0+), CUDA 12.1+, CPU (x86_64/ARM64)
