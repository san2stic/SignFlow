# 🚀 GPU Quickstart - SignFlow TorchServe

Guide de démarrage rapide pour TorchServe avec support **CPU, MPS (Apple Silicon), et CUDA GPU**.

## ✅ Votre Configuration Actuelle

**Système détecté :**
- **Platform** : Apple Silicon (ARM64)
- **GPU** : MPS (Apple Silicon GPU) ✅
- **PyTorch** : 2.8.0 avec support MPS
- **ONNX Runtime** : 1.19.2 avec CoreMLExecutionProvider

**Device recommandé** : `docker-compose.arm64.yml`

---

## 🏃 Démarrage en 3 Étapes

### 1. Build l'image TorchServe multi-device

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/SignFlow

# Build pour Apple Silicon avec support MPS
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml build torchserve
```

### 2. Démarrer TorchServe

```bash
# Démarrage avec auto-détection MPS
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up torchserve

# Vous devriez voir :
# ✅ MPS (Apple Silicon GPU) detected
# 🔧 Configuring TorchServe with MPS
```

### 3. Vérifier que ça fonctionne

```bash
# Health check
curl http://localhost:8080/ping

# Lister les modèles
curl http://localhost:8081/models

# Métriques
curl http://localhost:8082/metrics | grep ts_
```

---

## 📦 Créer et Déployer un Modèle

### Test d'inférence

```bash
# Test avec des landmarks fictifs
curl -X POST http://localhost:8080/predictions/signflow_baseline \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [
      [[0.5, 0.5, 0.1], [0.6, 0.4, 0.15]],
      [[0.5, 0.6, 0.12], [0.61, 0.41, 0.16]]
    ]
  }'
```

**Latence attendue sur Apple Silicon :** 10-30ms

---

**Plus de détails** : Voir `TORCHSERVE_MULTI_DEVICE.md`
