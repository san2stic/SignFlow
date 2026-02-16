# 🔧 Corrections Appliquées - 2026-02-16

## Problèmes Résolus

### 1. ✅ Port MLflow Conflict (Port 5000)
**Problème:** Port 5000 déjà utilisé par AirPlay Receiver sur macOS  
**Solution:** Changé MLflow de 5000 → 5001

```yaml
# docker-compose.yml
mlflow:
  ports:
    - "5001:5001"  # Avant: 5000:5000
```

### 2. ✅ TorchServe Permission Denied
**Problème:** `exec /home/model-server/start.sh: permission denied`  
**Causes:**
- `chmod +x` exécuté après `USER model-server`
- Volume mount écrasait le script avec permissions incorrectes

**Solution:**
- Déplacé `chmod +x` AVANT création utilisateur
- Supprimé volume mount du start.sh (gardé dans l'image)

```dockerfile
# Dockerfile.torchserve - AVANT création user
COPY torchserve/start.sh /home/model-server/start.sh
RUN chmod +x /home/model-server/start.sh
```

### 3. ✅ Frontend lucide-react Missing
**Problème:** `Failed to resolve import "lucide-react"`  
**Cause:** Volume anonyme `/app/node_modules` ne persistait pas

**Solution:** Volume nommé pour node_modules

```yaml
# docker-compose.yml
frontend:
  volumes:
    - ./frontend:/app
    - frontend_node_modules:/app/node_modules  # Nommé au lieu d'anonyme

volumes:
  frontend_node_modules:  # Déclaration
```

### 4. ✅ Java Package Name (Debian Trixie)
**Problème:** `Unable to locate package openjdk-17-jdk`  
**Cause:** python:3.11-slim basé sur Debian Trixie

**Solution:** Utiliser `default-jdk`

```dockerfile
RUN apt-get install -y default-jdk  # Au lieu de openjdk-17-jdk
```

---

## 📋 Commandes à Exécuter

### Option A: Script Automatique (Recommandé)

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/SignFlow

# Rebuild et redémarrer
./scripts/restart-services.sh --rebuild
```

### Option B: Commandes Manuelles

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/SignFlow

# 1. Arrêter les services
docker compose -f docker-compose.yml -f docker-compose.arm64.yml down

# 2. Supprimer les volumes pour fresh start (optionnel)
docker volume rm signflow_frontend_node_modules 2>/dev/null || true

# 3. Rebuild TorchServe avec les corrections
docker compose -f docker-compose.yml -f docker-compose.arm64.yml build torchserve

# 4. Redémarrer tous les services
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d

# 5. Vérifier les logs
docker logs -f signflow_torchserve  # Doit voir "✅ MPS detected"
docker logs -f signflow-frontend-1  # Doit installer lucide-react
```

---

## ✅ Vérification Post-Redémarrage

### 1. Services Démarrés

```bash
docker compose ps
# Tous les services doivent être "Up" ou "healthy"
```

### 2. Health Checks

```bash
# Backend
curl http://localhost:8000/health
# {"status":"healthy"}

# TorchServe
curl http://localhost:8080/ping
# {"status":"Healthy"}

# Frontend (dans le navigateur)
open http://localhost:3000

# MLflow
open http://localhost:5001
```

### 3. Logs TorchServe

```bash
docker logs signflow_torchserve 2>&1 | grep -E "MPS|device"
```

**Sortie attendue:**
```
✅ MPS (Apple Silicon GPU) detected
🔧 Configuring TorchServe with MPS
```

### 4. Frontend lucide-react

```bash
docker logs signflow-frontend-1 2>&1 | grep lucide
```

**Sortie attendue:**
```
added 1 package (lucide-react)
```

---

## 🐛 Troubleshooting

### TorchServe ne démarre toujours pas

```bash
# Vérifier les permissions du script
docker run --rm signflow-torchserve ls -la /home/model-server/start.sh
# Doit montrer: -rwxr-xr-x (x = exécutable)

# Rebuild force
docker compose build --no-cache torchserve
```

### Frontend toujours "lucide-react" missing

```bash
# Forcer npm install
docker compose exec frontend npm install

# Restart frontend
docker compose restart frontend
```

### Port 5001 aussi occupé

Changer dans `docker-compose.yml`:
```yaml
mlflow:
  ports:
    - "5002:5001"  # Utiliser 5002 côté hôte
```

---

## 📁 Fichiers Modifiés

1. `docker-compose.yml`
   - MLflow port: 5000 → 5001
   - Frontend: volume nommé pour node_modules
   - TorchServe: supprimé volume mount start.sh

2. `backend/Dockerfile.torchserve`
   - Java: `default-jdk` au lieu de `openjdk-17-jdk`
   - Permissions: `chmod +x` avant `USER model-server`

3. `scripts/restart-services.sh` (NOUVEAU)
   - Script automatique de redémarrage
   - Health checks intégrés
   - Support Apple Silicon auto

---

## 🎯 Résultat Attendu

Après ces corrections, vous devriez avoir :

- ✅ **7 services** opérationnels
- ✅ **TorchServe** avec MPS détecté
- ✅ **Frontend** avec toutes les dépendances
- ✅ **MLflow** sur port 5001
- ✅ **Aucune erreur** de permissions

**Temps de démarrage complet:** ~30-60 secondes

---

**Auteur:** Bastien Javaux  
**Date:** 2026-02-16  
**Version:** 1.0.0

### 5. ✅ TorchServe --disable-token-auth Flag
**Problème:** `torchserve: error: unrecognized arguments: --disable-token-auth`  
**Cause:** Flag n'existe pas dans TorchServe 0.9.0

**Solution:** Supprimé le flag de start.sh

```bash
# Avant
exec torchserve --start ... --disable-token-auth

# Après
exec torchserve --start ... --ncs  # Auth désactivé par défaut
```

---

## ⚠️ **IMPORTANT: Limitation MPS avec Docker**

### MPS (Apple Silicon GPU) N'EST PAS Accessible depuis Docker

**Raison Technique:**
- Docker sur macOS utilise une VM Linux
- MPS/Metal nécessite accès direct au kernel macOS
- VM Linux ne peut pas accéder au driver Metal

**Dans les logs TorchServe:**
```
⚠️  No GPU detected, using CPU
🔧 Configuring TorchServe with CPU
```

**✅ C'est NORMAL et ATTENDU pour Docker sur Apple Silicon**

### Solutions Alternatives

| Solution | Device | Latence | Use Case |
|----------|--------|---------|----------|
| **Docker CPU + ONNX** | CPU | 15-50ms | ✅ Production recommandée |
| **Native macOS** | MPS | 10-30ms | Dev local uniquement |
| **Cloud NVIDIA** | CUDA | 5-15ms | Production haute perf |

**Pour profiter de MPS**, lancez le backend **nativement** (sans Docker):

```bash
cd backend
pip install -r requirements.txt
TORCH_DEVICE=mps uvicorn app.main:app --reload
```

**📖 Détails complets :** Voir `backend/DOCKER_MPS_LIMITATION.md`

