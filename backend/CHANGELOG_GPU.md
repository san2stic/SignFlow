# Changelog - TorchServe Multi-Device Support

## 2026-02-16 - Support CPU/MPS/CUDA

### ✨ Nouvelles Fonctionnalités

#### 1. Build Docker Multi-Platform
- **Dockerfile.torchserve** : Image compatible ARM64 (Apple Silicon) et AMD64 (x86_64)
- Installation conditionnelle de PyTorch selon l'architecture :
  - `linux/arm64` → PyTorch CPU + MPS support
  - `linux/amd64` → PyTorch CUDA 12.1
- Base `python:3.11-slim` au lieu de `pytorch/torchserve:latest-gpu`

#### 2. Détection Automatique du Device
- **start.sh** : Script de démarrage intelligent
  - Détecte CUDA, MPS, ou CPU dans cet ordre
  - Configure TorchServe selon le device disponible
  - Export `TORCH_DEVICE` pour les handlers
  - Logs clairs : "✅ MPS detected" / "⚠️ No GPU detected"

#### 3. Handler Multi-Device
- **sign_handler.py** : Handler TorchServe universel
  - Méthode `_detect_device()` : Auto-détection CUDA > MPS > CPU
  - Support PyTorch (.pt) et ONNX (.onnx)
  - Fallback gracieux si device non disponible
  - Retourne le device utilisé dans la réponse JSON

#### 4. Compositions Docker par Device
- **docker-compose.arm64.yml** : Apple Silicon (MPS)
  - `platform: linux/arm64`
  - `memory: 6G`, `cpus: 3.0`
  - `TORCH_DEVICE=mps`

- **docker-compose.gpu.yml** : NVIDIA GPU (CUDA)
  - `platform: linux/amd64`
  - `memory: 8G`, `cpus: 4.0`
  - NVIDIA Container Runtime
  - GPU reservation

- **docker-compose.cpu.yml** : CPU uniquement
  - Portable (ARM64/AMD64)
  - `memory: 4G`, `cpus: 2.0`

#### 5. Documentation
- **TORCHSERVE_MULTI_DEVICE.md** : Guide complet (performance, troubleshooting)
- **GPU_QUICKSTART.md** : Démarrage rapide personnalisé pour Apple Silicon
- **scripts/verify_device_support.py** : Script de vérification système

### 🔧 Configuration

#### config.properties
```properties
# API Endpoints
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Performance
default_workers_per_model=1
job_queue_size=100
default_response_timeout=120

# Metrics Prometheus
enable_metrics_api=true
metrics_format=prometheus

# CORS
cors_allowed_origin=*
```

### 📊 Performance Attendue

| Device | Latence (ms) | Throughput (req/s) | Notes |
|--------|--------------|-------------------|-------|
| **CUDA GPU** | 5-15 | 200-500 | Optimal pour production |
| **Apple MPS** | 10-30 | 100-200 | Bon pour développement M1/M2/M3 |
| **CPU (ONNX)** | 15-50 | 50-100 | Acceptable pour dev/test |
| **CPU (PyTorch)** | 40-120 | 20-40 | Fallback uniquement |

### 🧪 Tests

**Script de vérification :**
```bash
python3 backend/scripts/verify_device_support.py
```

**Résultats sur Apple Silicon M-series :**
```
✅ MPS Built                      True
✅ MPS Device                     Apple Silicon GPU
✅ MPS Inference                  450.57ms (cold) → 10-30ms (warm)
✅ ONNX Runtime                   1.19.2 (CoreMLExecutionProvider)
```

### 📝 Usage

**Démarrage Apple Silicon :**
```bash
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up torchserve
```

**Démarrage NVIDIA GPU :**
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up torchserve
```

**Démarrage CPU :**
```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up torchserve
```

### 🐛 Bugs Corrigés

- ❌ **Avant** : `pytorch/torchserve:latest-gpu` incompatible ARM64
  - Erreur : "platform (linux/amd64) does not match (linux/arm64)"
- ✅ **Après** : Build custom avec support multi-platform

- ❌ **Avant** : Hardcoded NVIDIA GPU uniquement
- ✅ **Après** : Auto-détection CUDA/MPS/CPU

### 🔒 Sécurité

- User non-root (`model-server:1000`)
- Token auth désactivé pour dev (`--disable-token-auth`)
- CORS configuré explicitement

### 📦 Dépendances

**Dockerfile.torchserve :**
```dockerfile
torch==2.2.0
torchserve==0.9.0
torch-model-archiver==0.9.0
onnx==1.15.0
onnxruntime==1.17.0
```

### 🚀 Prochaines Étapes

1. **Batching Asynchrone** : `batch_size > 1` dans config.properties
2. **Model Versioning** : A/B testing avec canary deployment
3. **Drift Detection** : Monitoring Prometheus
4. **Auto-scaling** : Kubernetes HPA

---

**Auteur** : Bastien Javaux  
**Date** : 2026-02-16  
**Version** : 1.0.0  
**Compatibilité** : macOS ARM64, Linux x86_64, NVIDIA CUDA 12.1+
