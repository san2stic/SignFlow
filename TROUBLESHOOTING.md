# 🔧 Troubleshooting Guide - SignFlow

Guide de résolution des problèmes courants.

---

## TorchServe

### ❌ Problème : TorchServe redémarre en boucle (macOS)

**Symptômes :**
```
🚀 Starting TorchServe with device auto-detection...
⚠️  No GPU detected, using CPU
🔧 Configuring TorchServe with CPU
Removing orphan pid file.
[redémarre immédiatement]
```

**Cause :** `load_models=all` dans config.properties mais model-store vide

**Solution :**
```bash
# Option A: Désactiver auto-load (déjà fait)
# backend/torchserve/config/config.properties
# load_models=all → # load_models=all

# Option B: Ajouter un modèle dummy
cd backend/torchserve/model-store
touch .gitkeep

# Rebuild et redémarrer
docker compose down
docker compose -f docker-compose.yml -f docker-compose.arm64.yml build torchserve
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

**Vérification :**
```bash
docker logs signflow_torchserve
# Ne doit PAS voir de boucle, doit rester up
```

---

### ❌ Problème : "exec format error" (Ubuntu/x86_64)

**Symptômes :**
```
exec /bin/sh: exec format error
exit code: 255
```

**Cause :** Image buildée pour ARM64 mais machine est AMD64

**Solution :**

1. **Supprimez la ligne platform dans docker-compose.yml** (déjà fait)
2. **Rebuild sans forcer l'architecture :**

```bash
cd ~/SignFlow  # Votre chemin sur Ubuntu

# Clean des anciennes images
docker compose down
docker system prune -f

# Rebuild avec auto-détection architecture
docker compose build torchserve

# Ou utiliser docker-compose.cpu.yml
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build torchserve
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

**Vérification :**
```bash
# Vérifier l'architecture de l'image
docker inspect signflow-torchserve | grep Architecture
# Doit montrer "amd64" sur Ubuntu x86_64
```

---

### ❌ Problème : "No GPU detected" sur Apple Silicon

**Symptômes :**
```
⚠️  No GPU detected, using CPU
```

**C'est NORMAL ✅**

**Explication :** MPS (Apple Silicon GPU) n'est pas accessible depuis Docker (VM Linux). Voir `DOCKER_MPS_LIMITATION.md`.

**Performance attendue :**
- CPU PyTorch : 40-120ms
- CPU ONNX : 15-50ms (recommandé)

**Pour MPS natif (10-30ms) :**
```bash
cd backend
pip install -r requirements.txt
TORCH_DEVICE=mps uvicorn app.main:app --reload
```

---

## Frontend

### ❌ Problème : "Failed to resolve import lucide-react"

**Symptômes :**
```
Failed to resolve import "lucide-react" from "src/components/layout/Sidebar.tsx"
```

**Cause :** node_modules non installé ou volume monté incorrectement

**Solution :**

```bash
# Option A: Forcer npm install
docker compose exec frontend npm install
docker compose restart frontend

# Option B: Clean start
docker compose down
docker volume rm signflow_frontend_node_modules
docker compose up -d frontend

# Vérifier installation
docker compose exec frontend npm list lucide-react
```

---

### ❌ Problème : Frontend slow/HMR ne fonctionne pas

**Symptômes :**
- Hot Module Replacement lent
- Changements non détectés
- Build très lent

**Cause :** Volume mount + node_modules dans Docker

**Solution pour Dev :**

Lancez le frontend **nativement** sur votre machine :

```bash
cd frontend

# Install
npm install

# Dev server natif
npm run dev

# Ouvrir http://localhost:5173 (Vite port par défaut)
```

**Avantages :**
- ⚡ HMR instantané
- 🚀 Build 3-5x plus rapide
- 🔥 Meilleure expérience dev

---

## Backend

### ❌ Problème : Port 5000 déjà utilisé (MLflow)

**Symptômes :**
```
Error: ports are not available: listen tcp 0.0.0.0:5000: bind: address already in use
```

**Cause :** AirPlay Receiver utilise le port 5000 sur macOS

**Solution :** MLflow déjà configuré sur port 5001

```bash
# Vérifier
curl http://localhost:5001
# Ouvrir
open http://localhost:5001
```

**Si 5001 aussi occupé :**
```yaml
# docker-compose.yml
mlflow:
  ports:
    - "5002:5001"  # Utiliser 5002 sur l'hôte
```

---

### ❌ Problème : Database connection failed

**Symptômes :**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution :**

```bash
# Vérifier que PostgreSQL tourne
docker compose ps db
# Doit être "Up"

# Si down, redémarrer
docker compose up -d db

# Vérifier les logs
docker compose logs db

# Tester la connexion
docker compose exec backend python -c "
from app.database import engine
engine.connect()
print('✅ DB connected')
"
```

---

## Docker

### ❌ Problème : "Cannot connect to Docker daemon"

**Symptômes :**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution :**

```bash
# macOS: Vérifier Docker Desktop
open -a Docker

# Linux: Démarrer Docker
sudo systemctl start docker
sudo systemctl enable docker

# Vérifier
docker ps
```

---

### ❌ Problème : Out of disk space

**Symptômes :**
```
no space left on device
```

**Solution :**

```bash
# Nettoyer images inutilisées
docker system prune -a

# Nettoyer volumes inutilisés
docker volume prune

# Voir l'espace
docker system df

# Clean agressif (⚠️ supprime TOUT ce qui est arrêté)
docker system prune -a --volumes
```

---

### ❌ Problème : Build très lent

**Solution :**

```bash
# Activer BuildKit (cache layers)
export DOCKER_BUILDKIT=1

# Build avec cache
docker compose build

# Build sans cache (si problème)
docker compose build --no-cache torchserve
```

---

## Performance

### ⚡ Optimiser la latence CPU

**Pour production Docker CPU :**

```bash
# 1. Exporter le modèle en ONNX
cd backend
python app/ml/export.py \
  --model-path data/models/model.pt \
  --output-path data/models/model.onnx \
  --optimize

# 2. Créer .mar avec ONNX
torch-model-archiver \
  --model-name signflow_onnx \
  --version 1.0 \
  --serialized-file data/models/model.onnx \
  --handler torchserve/handlers/sign_handler.py \
  --export-path torchserve/model-store

# 3. Enregistrer dans TorchServe
curl -X POST "http://localhost:8081/models?url=signflow_onnx.mar"

# Gains attendus: 40-120ms → 15-50ms (2-5x speedup)
```

---

## Commandes Utiles

### Logs

```bash
# Tous les services
docker compose logs -f

# Service spécifique
docker logs -f signflow_torchserve
docker logs -f signflow-frontend-1
docker logs -f signflow-backend-1

# Dernières 50 lignes
docker logs --tail 50 signflow_torchserve
```

### État des services

```bash
# Liste des containers
docker compose ps

# Santé TorchServe
curl http://localhost:8080/ping

# Santé Backend
curl http://localhost:8000/health

# Métriques TorchServe
curl http://localhost:8082/metrics | grep ts_
```

### Redémarrage propre

```bash
# Tout arrêter
docker compose down

# Rebuild services spécifiques
docker compose build torchserve frontend backend

# Redémarrer
docker compose up -d

# Vérifier
docker compose ps
```

### Reset complet

```bash
# ⚠️ ATTENTION: Supprime TOUTES les données
docker compose down -v
docker system prune -a --volumes
docker compose up --build
```

---

## Obtenir de l'aide

**Logs à fournir :**
```bash
# Capturer tous les logs
docker compose logs > logs.txt

# Info système
docker version > system_info.txt
docker compose version >> system_info.txt
uname -a >> system_info.txt
```

**Fichiers à vérifier :**
1. `docker-compose.yml`
2. `backend/Dockerfile.torchserve`
3. `backend/torchserve/config/config.properties`
4. Logs des containers

---

**Mis à jour :** 2026-02-16
**Version :** 1.0.0
