# 🔍 Debug TorchServe Crash Loop

Guide pour identifier pourquoi TorchServe redémarre en boucle.

---

## 📊 Symptômes Actuels

```
🚀 Starting TorchServe with device auto-detection...
⚠️  No GPU detected, using CPU
🔧 Configuring TorchServe with CPU
Removing orphan pid file.
[redémarre immédiatement]
```

---

## 🔍 Étape 1: Voir les VRAIES Erreurs

J'ai ajouté `--foreground` au script de démarrage pour voir les logs complets.

### Commandes à exécuter :

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/SignFlow

# Arrêter
docker compose down

# Rebuild avec --foreground
docker compose build torchserve

# Démarrer EN PREMIER PLAN (voir les logs)
docker compose up torchserve
```

### Ce Que Vous Devriez Voir :

**✅ Si TorchServe démarre correctement :**
```
INFO - Model server started
INFO - Listening on port 8080
```

**❌ Si erreur, vous verrez un stack trace Python :**
```
ERROR - [stack trace détaillé]
Traceback (most recent call last):
  File ...
```

**→ Copiez l'erreur complète et je pourrai vous aider**

---

## 🧪 Étape 2: Test avec Docker Run Direct

Pour isoler le problème :

```bash
# Test TorchServe hors Docker Compose
docker run -it --rm \
  -p 8080:8080 \
  -v $(pwd)/backend/torchserve/model-store:/home/model-server/model-store \
  -v $(pwd)/backend/torchserve/config:/home/model-server/config \
  signflow-torchserve \
  /bin/bash

# Puis dans le container :
/home/model-server/start.sh
```

---

## 🔄 Étape 3: Démarrer SANS TorchServe

En attendant de résoudre le problème, vous pouvez démarrer le reste de l'app :

```bash
# Démarrer tout SAUF TorchServe
docker compose -f docker-compose.yml -f docker-compose.no-torchserve.yml up -d

# Vérifier
docker compose ps
# TorchServe sera "Up" mais inactif (alpine container)
# Backend, Frontend, DB, Redis, MLflow fonctionnent normalement
```

Le backend utilisera PyTorch direct au lieu de TorchServe.

---

## 🐛 Causes Possibles

### 1. Permissions Fichiers

```bash
# Vérifier permissions model-store
ls -la backend/torchserve/model-store/

# Doit être readable par user 1000
```

**Fix :**
```bash
chmod 755 backend/torchserve/model-store
```

### 2. Config Properties Invalide

```bash
# Tester la config
docker run -it --rm signflow-torchserve \
  cat /home/model-server/config/config.properties
```

**Vérifier :**
- Pas de caractères spéciaux
- Chemins corrects
- Syntaxe valide

### 3. Java Installation

```bash
# Vérifier Java dans l'image
docker run -it --rm signflow-torchserve java -version
```

**Attendu :**
```
openjdk version "17.x.x"
```

### 4. TorchServe Installation

```bash
# Vérifier TorchServe
docker run -it --rm signflow-torchserve torchserve --help
```

### 5. Port Déjà Utilisé

```bash
# Vérifier ports
lsof -i :8080
lsof -i :8081
lsof -i :8082

# Si occupés, changer dans docker-compose.yml :
ports:
  - "8090:8080"  # Utiliser 8090 au lieu de 8080
```

---

## 🔧 Solutions Alternatives

### Option A: TorchServe Image Officielle

Essayez l'image officielle au lieu de notre build :

```yaml
# docker-compose.yml
torchserve:
  image: pytorch/torchserve:latest-cpu  # Image officielle
  # Commentez 'build:'
  volumes:
    - ./backend/torchserve/model-store:/home/model-server/model-store
    - ./backend/torchserve/config:/home/model-server/config/config.properties
  ports:
    - "8080:8080"
    - "8081:8081"
    - "8082:8082"
```

```bash
docker compose up torchserve
```

### Option B: PyTorch Backend Direct (Sans TorchServe)

Le plus simple pour continuer le développement :

```bash
# Utiliser docker-compose.no-torchserve.yml
docker compose -f docker-compose.yml -f docker-compose.no-torchserve.yml up -d
```

Le backend FastAPI fera l'inférence directement avec PyTorch.

**Avantages :**
- ✅ Plus simple (pas de .mar à créer)
- ✅ Développement plus rapide
- ✅ Même performance pour dev local

**Inconvénient :**
- ❌ Pas de batching automatique
- ❌ Pas de métriques Prometheus natives

---

## 📋 Checklist Debug

Exécutez cette checklist et notez les résultats :

```bash
# 1. Container se build ?
docker compose build torchserve
# ✅ ou ❌

# 2. Start script exécutable ?
docker run --rm signflow-torchserve ls -la /home/model-server/start.sh
# Doit montrer -rwxr-xr-x

# 3. Java fonctionne ?
docker run --rm signflow-torchserve java -version
# Doit afficher version

# 4. TorchServe installé ?
docker run --rm signflow-torchserve torchserve --version
# Doit afficher version

# 5. Config valide ?
docker run --rm signflow-torchserve \
  cat /home/model-server/config/config.properties
# Doit afficher le fichier

# 6. Model store accessible ?
docker run --rm signflow-torchserve \
  ls -la /home/model-server/model-store/
# Doit lister .gitkeep

# 7. Foreground logs ?
docker compose up torchserve
# Noter l'erreur exacte
```

---

## 🆘 Si Rien Ne Marche

Utilisez la config sans TorchServe :

```bash
# 1. Arrêter tout
docker compose down

# 2. Démarrer sans TorchServe
docker compose -f docker-compose.yml -f docker-compose.no-torchserve.yml up -d

# 3. Vérifier que le reste fonctionne
curl http://localhost:8000/health  # Backend
curl http://localhost:3000         # Frontend (navigateur)
curl http://localhost:5001         # MLflow

# 4. Test inférence backend direct
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[[0.5, 0.5, 0.1]]]}'
```

---

## 📤 Partager pour Aide

Pour obtenir de l'aide, fournir :

```bash
# 1. Logs complets foreground
docker compose up torchserve > torchserve_logs.txt 2>&1
# Ctrl+C après quelques secondes

# 2. Info système
docker version > debug_info.txt
docker compose version >> debug_info.txt
uname -a >> debug_info.txt

# 3. Checklist résultats
# Coller les résultats de la checklist ci-dessus
```

---

**Mis à jour :** 2026-02-16
**Version :** 1.0.0
