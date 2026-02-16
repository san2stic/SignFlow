# ⚠️ Limitation MPS (Apple Silicon) avec Docker

## Le Problème

**MPS (Metal Performance Shaders) n'est PAS accessible depuis les containers Docker sur macOS.**

### Pourquoi ?

1. **Architecture** : Docker sur macOS utilise une VM Linux (via HyperKit/QEMU)
2. **MPS requis** : MPS nécessite un accès direct au driver Metal qui est spécifique à macOS
3. **Container Linux** : Les containers tournent dans une VM Linux qui n'a pas accès à Metal

```
macOS (Metal/MPS natif)
  └── Docker VM (Linux)
       └── Container (❌ Pas d'accès MPS)
```

### Conséquence

Dans les logs TorchServe, vous verrez :
```
⚠️  No GPU detected, using CPU
```

**C'est NORMAL et attendu** pour un déploiement Docker sur Apple Silicon.

---

## Solutions

### Option 1: CPU dans Docker (Actuel - Production)

**Avantages:**
- ✅ Portable (fonctionne partout)
- ✅ Containerisé (isolation, reproductibilité)
- ✅ Scalable (Kubernetes, Docker Swarm)

**Performance:**
- CPU (PyTorch): 40-120ms par inférence
- CPU (ONNX): 15-50ms par inférence (2-5x speedup)

**Recommandation:** Utilisez ONNX pour de meilleures perfs CPU

```bash
# Export en ONNX pour optimisation CPU
python backend/app/ml/export.py \
  --model-path data/models/model.pt \
  --output-path data/models/model.onnx \
  --optimize
```

### Option 2: MPS Natif (Dev Local uniquement)

**Pour profiter de MPS**, lancez le backend **directement sur macOS** (sans Docker) :

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run backend nativement
TORCH_DEVICE=mps uvicorn app.main:app --reload
```

**Performance MPS native:**
- Cold start: ~450ms (compilation Metal shaders)
- Warm inference: 10-30ms
- Throughput: 100-200 req/s

**⚠️ Limites:**
- Pas de containerisation
- Configuration manuelle des dépendances
- Pas scalable (single machine)

### Option 3: CUDA GPU (Production Cloud)

Pour les déploiements production nécessitant GPU, utilisez des serveurs NVIDIA :

```bash
# AWS EC2 avec GPU, GCP Compute Engine, Azure...
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

**Performance CUDA:**
- Latence: 5-15ms
- Throughput: 200-500 req/s
- Batching: Jusqu'à 32 samples simultanés

---

## Benchmark Comparatif

| Environnement | Device | Latence | Throughput | Scalabilité | Coût |
|---------------|--------|---------|------------|-------------|------|
| **Docker CPU (PyTorch)** | CPU | 40-120ms | 20-40 req/s | ⭐⭐⭐⭐⭐ | $ |
| **Docker CPU (ONNX)** | CPU | 15-50ms | 50-100 req/s | ⭐⭐⭐⭐⭐ | $ |
| **macOS Native MPS** | MPS | 10-30ms | 100-200 req/s | ⭐ | $ |
| **Cloud CUDA GPU** | CUDA | 5-15ms | 200-500 req/s | ⭐⭐⭐⭐⭐ | $$$ |

---

## Recommandations par Use Case

### Développement Local (Apple Silicon)

```bash
# Option A: Docker CPU avec ONNX (recommandé)
docker compose up  # CPU dans container
python backend/app/ml/export.py --optimize  # Export ONNX

# Option B: Native MPS (max performance dev)
cd backend && TORCH_DEVICE=mps uvicorn app.main:app --reload
```

### Production

```bash
# Small/Medium scale: Docker CPU avec ONNX
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up

# Large scale: Cloud GPU avec TorchServe
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## FAQ

### Q: Peut-on "passer" MPS au container Docker ?
**R:** Non, impossible techniquement. Metal est lié au kernel macOS.

### Q: Et si j'utilise Docker Desktop avec "Use VirtIO"?
**R:** Ça améliore les I/O disque mais ne donne pas accès à Metal/MPS.

### Q: Colima ou OrbStack changent quelque chose ?
**R:** Non, même limitation. Ils utilisent aussi une VM Linux.

### Q: Pourquoi CUDA fonctionne dans Docker mais pas MPS ?
**R:** CUDA utilise `nvidia-docker` qui expose `/dev/nvidia*` au container. Metal n'a pas d'équivalent Linux.

### Q: Performance CPU acceptable en production ?
**R:** Oui ! Avec ONNX Runtime, 15-50ms est suffisant pour beaucoup d'applications. Si besoin <10ms, utilisez Cloud GPU.

---

## Conclusion

**TL;DR:**
- 🐳 **Docker = CPU uniquement** (Apple Silicon ou x86_64)
- 🍎 **MPS = Native macOS uniquement** (dev local)
- 🚀 **GPU production = Cloud NVIDIA** (CUDA)

Pour SignFlow :
- **Dev :** Docker CPU + ONNX (simple, portable)
- **Prod :** Cloud GPU si latence critique, sinon CPU + ONNX suffit

---

**Mis à jour :** 2026-02-16
**Version :** 1.0.0
