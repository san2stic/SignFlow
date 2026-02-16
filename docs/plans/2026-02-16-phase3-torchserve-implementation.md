# Phase 3: Infrastructure de Serving Scalable avec TorchServe - Plan d'Implémentation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remplacer le pipeline d'inférence singleton par TorchServe avec batching GPU pour passer de 30 req/s à 200-500 req/s.

**Architecture:** TorchServe sidecar dans docker-compose avec handler custom pour preprocessing/inference/postprocessing. FastAPI garde l'état temporel (frame_buffer, state machine) et appelle TorchServe via HTTP async. Feature flag `USE_TORCHSERVE` permet migration sécurisée avec fallback automatique.

**Tech Stack:** TorchServe (GPU batching), httpx (client HTTP async), torch-model-archiver (packaging .mar), NVIDIA Docker (GPU support)

---

## Task 1: Configuration TorchServe Base

**Files:**
- Create: `backend/torchserve/config/config.properties`
- Create: `backend/torchserve/requirements.txt`
- Create: `backend/torchserve/README.md`
- Modify: `docker-compose.yml`

**Step 1: Créer structure TorchServe**

Run: `mkdir -p backend/torchserve/{config,model-store}`

**Step 2: Créer configuration TorchServe**

Créer `backend/torchserve/config/config.properties`:

```properties
# Inference API
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Workers
default_workers_per_model=2

# Batching
batch_size=16
max_batch_delay=50

# GPU
number_of_gpu=1
enable_envvars_config=true

# Logging
install_py_dep_per_model=true
enable_metrics_api=true
```

**Step 3: Créer requirements handler**

Créer `backend/torchserve/requirements.txt`:

```
torch>=2.0.0
numpy>=1.24.0
structlog>=23.1.0
```

**Step 4: Créer README**

Créer `backend/torchserve/README.md` avec documentation setup (voir Task 1 complet dans design doc)

**Step 5: Ajouter service docker-compose**

Dans `docker-compose.yml`, ajouter après service `mlflow`:

```yaml
  torchserve:
    image: pytorch/torchserve:latest-gpu
    container_name: signflow_torchserve
    ports:
      - "8080:8080"
      - "8081:8081"
      - "8082:8082"
    volumes:
      - ./backend/torchserve/model-store:/home/model-server/model-store
      - ./backend/torchserve/config:/home/model-server/config
    environment:
      - TS_NUMBER_OF_GPU=1
      - TS_INSTALL_PY_DEP_PER_MODEL=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1
```

**Step 6: Tester démarrage**

Run: `docker-compose up torchserve -d`
Expected: Service démarre

Run: `curl http://localhost:8080/ping`
Expected: `{"status": "Healthy"}`

**Step 7: Commit**

```bash
git add backend/torchserve/ docker-compose.yml
git commit -m "feat(phase3): add TorchServe service to docker-compose

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: TorchServe Client HTTP

**Files:**
- Create: `backend/app/ml/torchserve_client.py`
- Create: `backend/tests/test_ml/test_torchserve_client.py`
- Modify: `backend/pyproject.toml`

**Step 1: Ajouter dépendance httpx**

Dans `backend/pyproject.toml`, section dependencies:

```toml
httpx = "^0.27.0"
respx = "^0.21.0"  # dev dependency pour tests
```

Run: `cd backend && poetry install`

**Step 2: Écrire test client (failing)**

Créer `backend/tests/test_ml/test_torchserve_client.py` avec tests de base (voir Task 2 complet)

**Step 3: Run test**

Run: `cd backend && poetry run pytest tests/test_ml/test_torchserve_client.py -v`
Expected: FAIL - Module not found

**Step 4: Implémenter client**

Créer `backend/app/ml/torchserve_client.py` avec class `TorchServeClient` (voir Task 2 complet)

**Step 5: Run test**

Run: `cd backend && poetry run pytest tests/test_ml/test_torchserve_client.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backend/app/ml/torchserve_client.py backend/tests/test_ml/test_torchserve_client.py backend/pyproject.toml
git commit -m "feat(phase3): implement TorchServe HTTP client

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Feature Flag Configuration

**Files:**
- Modify: `backend/app/config.py`
- Modify: `docker-compose.yml`
- Create: `backend/tests/test_config.py`

**Step 1: Écrire test config (failing)**

Créer `backend/tests/test_config.py` (voir Task 3 complet)

**Step 2: Run test**

Run: `cd backend && poetry run pytest tests/test_config.py -v`
Expected: FAIL

**Step 3: Ajouter settings**

Dans `backend/app/config.py`, ajouter dans class Settings:

```python
# TorchServe (Phase 3)
use_torchserve: bool = Field(default=False, env="USE_TORCHSERVE")
torchserve_url: str = Field(default="http://torchserve:8080", env="TORCHSERVE_URL")
torchserve_timeout_ms: int = Field(default=2000, env="TORCHSERVE_TIMEOUT_MS")
```

**Step 4: Run test**

Run: `cd backend && poetry run pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Ajouter env vars docker-compose**

Dans `docker-compose.yml`, backend.environment:

```yaml
- USE_TORCHSERVE=false
- TORCHSERVE_URL=http://torchserve:8080
- TORCHSERVE_TIMEOUT_MS=2000
```

**Step 6: Commit**

```bash
git add backend/app/config.py backend/tests/test_config.py docker-compose.yml
git commit -m "feat(phase3): add TorchServe feature flag

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: TorchServe Handler

**Files:**
- Create: `backend/torchserve/handler.py`
- Create: `backend/torchserve/labels.json`
- Create: `backend/scripts/package_torchserve_model.sh`

**Step 1: Créer labels default**

Créer `backend/torchserve/labels.json`:

```json
["HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY"]
```

**Step 2: Implémenter handler**

Créer `backend/torchserve/handler.py` avec class `SignFlowHandler` (voir Task 4 complet, utilise `json.loads()` pas `json.parse()`)

**Step 3: Créer script packaging**

Créer `backend/scripts/package_torchserve_model.sh` (voir Task 4 complet)

Run: `chmod +x backend/scripts/package_torchserve_model.sh`

**Step 4: Commit**

```bash
git add backend/torchserve/handler.py backend/torchserve/labels.json backend/scripts/package_torchserve_model.sh
git commit -m "feat(phase3): implement TorchServe handler

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Intégration FastAPI

**Files:**
- Modify: `backend/app/ml/pipeline.py`
- Modify: `backend/app/api/translate.py`
- Create: `backend/tests/test_api/test_translate_torchserve.py`

**Step 1: Écrire test intégration (failing)**

Créer `backend/tests/test_api/test_translate_torchserve.py` (voir Task 5 complet)

**Step 2: Run test**

Run: `cd backend && poetry run pytest tests/test_api/test_translate_torchserve.py -v`
Expected: FAIL

**Step 3: Modifier pipeline.py**

Ajouter méthode `_infer_window_async()` avec support TorchServe + fallback (voir Task 5 complet)

**Step 4: Modifier translate.py**

Ajouter `get_torchserve_client()` et initialisation dans `translate_stream()` (voir Task 5 complet)

**Step 5: Run test**

Run: `cd backend && poetry run pytest tests/test_api/test_translate_torchserve.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backend/app/ml/pipeline.py backend/app/api/translate.py backend/tests/test_api/test_translate_torchserve.py
git commit -m "feat(phase3): integrate TorchServe in pipeline

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Benchmark Script

**Files:**
- Create: `backend/scripts/benchmark_torchserve.py`

**Step 1: Créer script**

Créer `backend/scripts/benchmark_torchserve.py` (voir Task 6 complet)

Run: `chmod +x backend/scripts/benchmark_torchserve.py`

**Step 2: Tester**

Run: `cd backend && poetry run python scripts/benchmark_torchserve.py --concurrent 2 --duration 10`
Expected: Résultats affichés

**Step 3: Commit**

```bash
git add backend/scripts/benchmark_torchserve.py
git commit -m "feat(phase3): add benchmark script

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Documentation

**Files:**
- Create: `docs/phase3-validation-checklist.md`
- Modify: `backend/README.md`

**Step 1: Créer checklist**

Créer `docs/phase3-validation-checklist.md` avec checklist complète (voir Task 7 complet)

**Step 2: Mettre à jour README**

Dans `backend/README.md`, ajouter section TorchServe (voir Task 7 complet)

**Step 3: Commit final**

```bash
git add docs/phase3-validation-checklist.md backend/README.md
git commit -m "docs(phase3): add validation checklist

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Validation Finale

Run checklist: `docs/phase3-validation-checklist.md`

Critères succès:
- ✅ Tous tests passent
- ✅ Throughput >200 req/s (TorchServe)
- ✅ Latency p95 <50ms
- ✅ Feature flag fonctionne
- ✅ Fallback automatique OK

Si tous validés → **Phase 3 COMPLETE**
