# Design Phase 3 : Infrastructure de Serving Scalable avec TorchServe

**Date** : 2026-02-16
**Auteur** : Claude (via brainstorming)
**Statut** : Approuvé pour implémentation

---

## Contexte

SignFlow Phase 1-2 ont établi les fondations ML (MLflow tracking, model configs, ONNX export, architecture GCN/Transformer). La Phase 3 vise à remplacer le pipeline d'inférence singleton par une infrastructure scalable capable de supporter :

- **Batching GPU** : Traiter plusieurs requêtes simultanément pour maximiser l'utilisation GPU
- **Scaling horizontal** : Capacité à ajouter des instances TorchServe selon la charge
- **Métriques natives** : Prometheus metrics prêtes pour Phase 4 (MLOps & Monitoring)
- **Migration sécurisée** : Feature flag permettant un rollback instantané

**État actuel** :
- Pipeline singleton global (`_global_pipeline`) chargé au démarrage
- Inférence séquentielle : 1 frame → 1 forward pass CPU/ONNX
- Pas de batching, pas de GPU, pas de scaling horizontal
- Throughput limité : ~30 req/s (1 connexion WebSocket à 30fps)

**Contraintes validées** :
- GPU NVIDIA disponible (CUDA)
- Déploiement TorchServe en sidecar (docker-compose)
- Migration avec feature flag `USE_TORCHSERVE` pour rollback sécurisé

---

## 1. Vue d'Ensemble de l'Architecture

### Architecture Globale

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│   Frontend  │ WS:8000 │   FastAPI    │ HTTP    │  TorchServe  │
│  (React)    │────────>│   Backend    │────────>│   (GPU)      │
│             │         │              │  :8080  │              │
└─────────────┘         └──────────────┘         └──────────────┘
                              │                         │
                              │ Feature Flag            │
                              ▼ USE_TORCHSERVE         │
                        ┌──────────────┐               │
                        │  Old Pipeline│               │
                        │  (fallback)  │               │
                        └──────────────┘               │
                                                        ▼
                                                  ┌──────────────┐
                                                  │  Prometheus  │
                                                  │   Metrics    │
                                                  └──────────────┘
```

### Composants Principaux

1. **TorchServe Service** (nouveau)
   - Serveur d'inférence PyTorch officiel avec GPU support
   - Dynamic batching natif (batch_size=16-32, max_batch_delay=50ms)
   - Métriques Prometheus exposées sur port 8082

2. **TorchServe Handler** (nouveau)
   - `backend/torchserve/handler.py`
   - Implémente preprocessing (landmarks → features enrichies), inference (batch GPU), postprocessing (logits → predictions)

3. **TorchServe Client** (nouveau)
   - `backend/app/ml/torchserve_client.py`
   - Wrapper HTTP async pour appeler TorchServe depuis FastAPI
   - Gestion timeouts, retry, fallback automatique

4. **Feature Flag** (nouveau)
   - Variable d'environnement `USE_TORCHSERVE=true/false`
   - Toggle dans `backend/app/config.py`

5. **Pipeline Existant** (conservé)
   - Reste en place comme fallback si TorchServe disabled ou en erreur
   - Sera retiré après 2 semaines de stabilité en production

### Flux de Données

1. Frontend envoie landmarks via WebSocket → `backend/app/api/translate.py:translate_stream()`
2. FastAPI lit `settings.use_torchserve` flag
3. **Si enabled** :
   - FastAPI accumule frames dans `frame_buffer` (state machine inchangée)
   - Détection `sign_ended` → appel `torchserve_client.predict(window)`
   - TorchServe fait batching automatique + GPU inference
   - Retour JSON `{"prediction": str, "confidence": float, "alternatives": [...]}`
4. **Si disabled** :
   - Utilise `pipeline._infer_window()` (comportement actuel)
5. Réponse envoyée au frontend via WebSocket

**Principe clé** : L'état temporel (frame_buffer, state machine IDLE→RECORDING→INFERRING) reste dans FastAPI. TorchServe est stateless et ne fait que l'inférence sur séquences complètes (64 frames).

---

## 2. TorchServe Handler Custom

### Fichier : `backend/torchserve/handler.py`

Le handler hérite de `BaseHandler` et implémente 4 méthodes obligatoires.

#### 2.1 `initialize(context)`

Chargement du modèle au démarrage de TorchServe.

**Responsabilités** :
- Charger modèle PyTorch ou ONNX depuis le .mar archive
- Charger labels depuis metadata JSON
- Configurer device (CUDA si disponible)

#### 2.2 `preprocess(data)`

Conversion landmarks bruts → features enrichies (réutilise code existant).

**Pipeline de transformation** :
1. Parse JSON landmarks: `{"hands": {...}, "pose": [...]}`
2. `normalize_landmarks()` → array [225]
3. `compute_enriched_features()` → array [469]
4. `temporal_resample()` → [64, 469] si nécessaire
5. Retourner tensor [batch, 64, 469]

**Point critique** : Le handler doit inclure `app/ml/features.py`, `app/ml/feature_engineering.py`, `app/ml/dataset.py` dans le `.mar` (via `--extra-files`). Pas de duplication de logique.

#### 2.3 `inference(data, *args, **kwargs)`

Batch forward pass sur GPU.

TorchServe accumule automatiquement jusqu'à `batch_size=16` requêtes ou timeout `max_batch_delay=50ms` avant d'appeler cette méthode.

#### 2.4 `postprocess(inference_output)`

Conversion logits → JSON avec top-K predictions.

**Transformation** :
1. Recevoir logits [batch, num_classes]
2. Appliquer softmax avec température
3. Top-K predictions (k=4)
4. Mapper indices → labels (class_names)
5. Retourner JSON: `{"prediction": str, "confidence": float, "alternatives": [...]}`

### Gestion de l'État Temporel

**Question** : TorchServe est stateless, comment gérer `frame_buffer` et la state machine ?

**Réponse** : L'état reste dans FastAPI (`pipeline.process_frame()`). TorchServe fait uniquement l'inférence sur séquences complètes.

**Flow** :
1. FastAPI accumule frames dans `frame_buffer` (inchangé)
2. Détection `sign_ended` (rest frames threshold) → FastAPI
3. FastAPI extrait window [64, 225], applique enrichment → [64, 469]
4. FastAPI envoie JSON à TorchServe : `{"window": [[...], [...], ...]}` (64 frames)
5. TorchServe fait preprocessing (déjà enrichi → skip), inference, postprocessing
6. Retour à FastAPI qui met à jour `sentence_tokens` et envoie au frontend

**Alternative simplifiée** : FastAPI envoie directement `enriched_window [64, 469]` en base64, handler fait juste `np.frombuffer()` + inference. Évite de redupliquer compute_enriched_features côté TorchServe.

---

## 3. Configuration TorchServe & Docker

### 3.1 Service Docker Compose

Ajouter dans `docker-compose.yml` :

```yaml
torchserve:
  image: pytorch/torchserve:latest-gpu
  ports:
    - "8080:8080"  # Inference API
    - "8081:8081"  # Management API
    - "8082:8082"  # Metrics API
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

### 3.2 Configuration TorchServe (`config.properties`)

```properties
# Inference
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Batching (clé de la performance)
default_workers_per_model=2
batch_size=16
max_batch_delay=50

# GPU
number_of_gpu=1
```

### 3.3 Model Archive (MAR)

Le modèle doit être packagé en `.mar` avec :

```bash
torch-model-archiver \
  --model-name signflow \
  --version 1.0 \
  --serialized-file model_v1.pt \
  --handler handler.py \
  --extra-files "app/ml/features.py,app/ml/feature_engineering.py,app/ml/dataset.py" \
  --export-path model-store/
```

### 3.4 Métriques Prometheus Exposées

TorchServe expose automatiquement (port 8082) :
- `ts_inference_requests_total` - Nombre total de requêtes
- `ts_inference_latency_microseconds` - Histogramme de latence
- `ts_queue_latency_microseconds` - Temps d'attente dans la queue de batching
- `GPUUtilization` - % utilisation GPU

---

## 4. Client TorchServe dans FastAPI

### 4.1 Nouveau Module `backend/app/ml/torchserve_client.py`

Responsabilités :
1. Wrapper HTTP pour appeler TorchServe
2. Gestion des timeouts et retry
3. Conversion landmarks → format TorchServe
4. Parsing des réponses TorchServe

### Interface Publique

```python
class TorchServeClient:
    def __init__(self, base_url: str = "http://torchserve:8080"):
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=2.0)

    async def predict(
        self,
        window: np.ndarray  # [64, 469]
    ) -> tuple[str, float, list[dict]]:
        """
        Envoie une séquence à TorchServe pour inférence.

        Returns:
            (prediction, confidence, alternatives)
        """
```

### 4.2 Modification de `translate.py`

Ajouter le feature flag :

```python
from app.config import get_settings

settings = get_settings()

if settings.use_torchserve:
    # Nouveau chemin TorchServe
    torchserve_client = TorchServeClient()
    prediction = await torchserve_client.predict(window)
else:
    # Ancien chemin (pipeline existant)
    prediction = pipeline._infer_window(window)
```

**Point clé** : L'état temporel (frame_buffer, state machine) reste dans `pipeline.process_frame()`. Seule la méthode `_infer_window()` est remplacée par l'appel TorchServe.

### Gestion des Erreurs

Si TorchServe est down ou timeout :
1. Log l'erreur avec structlog
2. **Fallback automatique** vers l'ancien pipeline (même si `USE_TORCHSERVE=true`)
3. Incrémente métrique `torchserve_errors_total`
4. Continue de servir les requêtes (pas de downtime)

### Settings à Ajouter

Dans `backend/app/config.py` :

```python
class Settings(BaseSettings):
    # Existing settings...

    # TorchServe
    use_torchserve: bool = False
    torchserve_url: str = "http://torchserve:8080"
    torchserve_timeout_ms: int = 2000
```

---

## 5. Stratégie de Migration & Tests

### Plan de Déploiement

**Phase 1 - Développement (Semaine 1-2)**
1. Implémenter TorchServe handler + client
2. Tests locaux avec `USE_TORCHSERVE=false` (aucun impact)
3. Créer le `.mar` avec un modèle de test
4. Valider docker-compose démarre TorchServe correctement

**Phase 2 - Validation (Semaine 2-3)**
1. Activer `USE_TORCHSERVE=true` en dev
2. Tests end-to-end : frontend → FastAPI → TorchServe → GPU
3. Benchmark latence/throughput (comparer avec baseline)
4. Vérifier métriques Prometheus

**Phase 3 - Production (Semaine 3-4)**
1. Déployer avec `USE_TORCHSERVE=false` (pas de changement)
2. Basculer `USE_TORCHSERVE=true` pour 10% du traffic
3. Monitorer 24-48h (latency, erreurs, GPU utilization)
4. Si stable → `USE_TORCHSERVE=true` pour 100%
5. Après 2 semaines stable → Retirer l'ancien code pipeline

### Tests Critiques

**1. Test de Charge**
- Simuler N connexions WebSocket concurrentes
- Mesurer throughput (req/s) et latency (p50, p95, p99)
- Comparer TorchServe vs ancien pipeline

**2. Test de Fallback**
- Tuer TorchServe pendant une session active
- Vérifier que FastAPI bascule automatiquement sur ancien pipeline
- Pas de crash, juste logs d'erreur

**3. Test de Rechargement Modèle**
- Utiliser Management API pour recharger un nouveau .mar
- Vérifier que les nouvelles requêtes utilisent le nouveau modèle
- Pas de downtime

### Métriques de Succès

**Avant (Baseline actuel)**
- Throughput : ~30 req/s (1 req/frame à 30fps)
- Latency p95 : ~50ms (CPU inference)
- GPU utilization : 0%

**Après (TorchServe avec batching GPU)**
- Throughput : **200-500 req/s** (batching 16-32)
- Latency p95 : **<30ms** (GPU + batching overhead compensé)
- GPU utilization : **40-70%**

### Rollback d'Urgence

Si problème critique en production :
1. `docker-compose exec backend bash`
2. `export USE_TORCHSERVE=false`
3. Restart backend : `docker-compose restart backend`
4. Retour immédiat à l'ancien pipeline (< 30 secondes)

---

## 6. Fichiers à Créer/Modifier

### Fichiers Nouveaux

1. `backend/torchserve/handler.py` (~200 lignes)
2. `backend/torchserve/config/config.properties` (~15 lignes)
3. `backend/torchserve/requirements.txt` (~5 lignes)
4. `backend/app/ml/torchserve_client.py` (~150 lignes)
5. `backend/scripts/benchmark_torchserve.py` (~100 lignes)
6. `backend/scripts/package_model.sh` (~20 lignes)

### Fichiers Modifiés

1. `docker-compose.yml` (+20 lignes)
2. `backend/app/config.py` (+15 lignes)
3. `backend/app/ml/pipeline.py` (~30 lignes modifiées)
4. `backend/app/api/translate.py` (+20 lignes)
5. `backend/pyproject.toml` (+2 lignes)

---

## 7. Risques & Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| **TorchServe crash** | Downtime inference | Fallback automatique vers pipeline local |
| **GPU OOM** | Erreurs batch | Tuner `batch_size` selon GPU RAM |
| **Latency augmente** | UX dégradée | Benchmark avant prod + rollback via feature flag |
| **Batching inefficace** | Pas de gain throughput | Tuner `max_batch_delay` |

---

## Conclusion

La Phase 3 transforme SignFlow d'un pipeline mono-instance CPU en infrastructure scalable GPU-ready avec :

✅ **Batching automatique** (16-32 requêtes simultanées)
✅ **GPU support** (10-50x throughput)
✅ **Métriques Prometheus** natives (prêt pour Phase 4)
✅ **Migration sécurisée** (feature flag + fallback)
✅ **Zero downtime** (rechargement modèle à chaud)

**Timeline** : 3-4 semaines
**Effort** : ~6 fichiers nouveaux, ~5 fichiers modifiés
**Gain attendu** : 10-20x throughput, latency -40%, scalabilité horizontale

Prêt pour implémentation via **writing-plans skill**.
