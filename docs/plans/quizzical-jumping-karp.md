# Plan d'Upgrade ML pour SignFlow

## Contexte

SignFlow est une plateforme de traduction de langue des signes en temps réel avec inférence ML. L'analyse approfondie du code révèle un système fonctionnel mais limité qui nécessite une modernisation significative pour atteindre les performances de production.

**État actuel :**
- **Modèle** : Transformer PyTorch (192-dim, 6 têtes, 4 couches, ~150k paramètres)
- **Features** : 469 dimensions hand-crafted (landmarks + vélocités + angles + formes de mains)
- **Extraction** : MediaPipe Holistic (backend + frontend)
- **Entraînement** : AdamW + warmup + cosine + label smoothing + mixup + EMA + SWA
- **Inférence** : Machine à états (IDLE→RECORDING→INFERRING), pipeline singleton global
- **Infrastructure** : FastAPI + SQLAlchemy + Redis + Celery optionnel
- **Frontend** : React 18 + TypeScript + Vite + Zustand + WebSocket streaming 30fps

**Lacunes identifiées :**
1. **Capacité du modèle insuffisante** : 150k params vs 1-10M pour SOTA
2. **Features hand-crafted** : Manquent les patterns spatiaux que capturerait un CNN/ViT
3. **Resampling temporel fixe** : 64 frames fixes perdent la dynamique de vitesse variable
4. **Pas de pretraining** : Pas d'apprentissage auto-supervisé ou curriculum learning
5. **Serving mono-instance** : Pas de scaling horizontal, pas de batching, pas d'optimisation GPU
6. **Pas de MLOps** : Pas de tracking d'expériences, pas de registry de modèles, pas de détection de drift
7. **Pas de distributed training** : Multi-GPU non supporté
8. **Pas de pipeline de déploiement** : Pas d'A/B testing, pas de canary, pas de shadow mode

## Stratégie d'Upgrade - Approche par Phases

### Phase 1 : Fondations & Quick Wins (3 semaines)

**Objectif** : Établir l'infrastructure MLOps de base et gains rapides de performance sans disruption.

#### 1.1 Infrastructure MLflow
- **Fichier** : `backend/app/ml/tracking.py` (NOUVEAU)
- **Implémentation** :
  - Classe `MLFlowTracker` pour logging automatique des runs d'entraînement
  - Tracking URI : `{settings.model_dir}/mlruns`
  - Log params, metrics par epoch, artifacts (modèles PT)
- **Intégration** : Modifier `backend/app/ml/trainer.py:SignTrainer.fit()` pour appeler le tracker
- **Docker** : Ajouter service MLFlow UI dans `docker-compose.yml` (port 5000)
- **Dépendances** : `mlflow>=2.10.0` dans `backend/pyproject.toml`
- **Bénéfices** : Comparaison centralisée d'expériences, versioning, fondation pour registry

#### 1.2 Augmentation de la capacité du modèle (conservative)
- **Fichier** : `backend/app/ml/model_configs.py` (NOUVEAU)
- **Configuration upgrade** :
  ```python
  d_model=384 (192→384, 2x)
  nhead=8 (6→8)
  num_layers=6 (4→6, +50%)
  dim_feedforward=1536 (768→1536, 2x)
  # ~1.5M params (10x augmentation)
  ```
- **Approche** :
  - Entraîner les deux modèles en parallèle (150k et 1.5M)
  - A/B test avec MLFlow pour comparaison
  - Backward compatible avec features 469-dim existantes
- **Gain attendu** : +5-10% accuracy sur validation
- **Compatible** : CPU/MPS (4-8GB RAM suffisants)

#### 1.3 Export ONNX pour optimisation inférence
- **Fichier** : `backend/app/ml/export.py` (NOUVEAU)
- **Implémentation** :
  - Fonction `export_to_onnx()` avec opset_version=17
  - Axes dynamiques pour batch et sequence
- **Modification** : `backend/app/ml/pipeline.py:SignFlowInferencePipeline`
  - Détection auto `.onnx` vs `.pt`
  - ONNXRuntime session si disponible
  - Fallback PyTorch si échec
- **Dépendances** : `onnx>=1.15.0`, `onnxruntime>=1.17.0`
- **Bénéfices** : 2-5x speedup CPU, empreinte mémoire réduite, fondation TensorRT

---

### Phase 2 : Architecture Avancée (5 semaines)

**Objectif** : Remplacer features hand-crafted par apprentissage spatial et modélisation temporelle variable.

#### 2.1 Modèle hybride spatial-temporel
- **Fichier** : `backend/app/ml/spatial_encoder.py` (NOUVEAU)
- **Architecture** :
  ```python
  SpatialLandmarkEncoder (Graph Convolution Network)
    - Input: [batch, seq, 225] (landmarks bruts)
    - Graph: Arêtes mains (os des doigts) + squelette pose
    - GCN: 2 couches (225→128→256 dims)
    - Output: [batch, seq, 256] (features spatiales apprises)

  SpatialTemporalTransformer
    - SpatialEncoder → TemporalTransformer
    - Remplace features 469-dim hand-crafted
  ```
- **Fichier modifié** : `backend/app/ml/model.py` - ajouter nouvelle classe
- **Dépendances** : `torch-geometric>=2.5.0`
- **Migration** : Entraînement parallèle ancien/nouveau, A/B test via MLFlow
- **Gain attendu** : +10-15% accuracy (patterns spatiaux anatomiques appris)

#### 2.2 Modélisation temporelle à longueur variable
- **Fichier** : `backend/app/ml/temporal_pooling.py` (NOUVEAU)
- **Remplacement** : `dataset.py:temporal_resample()` par TCN pooling
- **Architecture** :
  ```python
  TemporalConvPooling
    - Convolutions dilatées multi-échelles (dilation 1,2,4)
    - Préserve dynamique temporelle variable
    - AdaptiveAvgPool1d vers 64 tokens fixes
  ```
- **Modifications** :
  - `backend/app/ml/dataset.py:LandmarkDataset` garde longueurs originales
  - `backend/app/ml/pipeline.py` handle buffers de taille variable
- **Bénéfices** : Préserve timing (critique pour signes), pas d'artefacts d'interpolation

#### 2.3 Pretraining auto-supervisé
- **Fichier** : `backend/app/ml/pretraining.py` (NOUVEAU)
- **Stratégie** : Masked Landmark Modeling (type BERT)
  - Masquer 15% des landmarks aléatoirement
  - Prédire landmarks masqués via tête de reconstruction
  - Loss MSE sur positions masquées uniquement
- **Script** : `backend/scripts/pretrain_encoder.py` (NOUVEAU)
- **Données** : Vidéos non-labellisées (YouTube, WLASL raw)
- **Workflow** : Pretrain → Fine-tune sur dataset labellisé
- **Gain attendu** : +8-12% accuracy (surtout signes rares)

---

### Phase 3 : Infrastructure de Serving Scalable (4 semaines)

**Objectif** : Remplacer pipeline singleton par serveur de modèles production-grade.

#### 3.1 Déploiement TorchServe
- **Fichier** : `docker-compose.prod.yml` - ajouter service `torchserve`
  - Image: `pytorch/torchserve:latest-gpu`
  - Ports: 8080 (inference), 8081 (management), 8082 (metrics)
  - GPU: NVIDIA device binding
- **Handler** : `backend/torchserve/handler.py` (NOUVEAU)
  - Classe `SignFlowHandler` (BaseHandler)
  - Preprocess: landmarks → enriched features
  - Inference: batch forward pass
  - Postprocess: top-k predictions + confidences
- **Model Archive** : Script packaging `torch-model-archiver`
- **Migration** : Déploiement parallèle, canary 10% → 100%
- **Bénéfices** : Dynamic batching, GPU support (10-50x speedup), métriques Prometheus natives

#### 3.2 Batching asynchrone & optimisation GPU
- **Fichier** : `backend/app/ml/batch_processor.py` (NOUVEAU)
- **Architecture** :
  ```python
  BatchedInferenceQueue
    - Queue async avec asyncio.Future
    - Accumulation jusqu'à batch_size ou max_latency_ms
    - Traitement batch GPU
    - Distribution des résultats via futures
  ```
- **Modification** : `backend/app/api/translate.py`
  - Remplacer inférence séquentielle par `batch_processor.infer()`
  - Garde session state isolé par WebSocket
- **Paramètres** : batch_size=32, max_latency_ms=50
- **Bénéfices** : 5-10x throughput, meilleure utilisation GPU

---

### Phase 4 : MLOps & Monitoring (4 semaines)

**Objectif** : Observabilité complète et détection de dégradation.

#### 4.1 Model Registry & promotion workflow
- **Fichier** : `backend/app/ml/registry.py` (NOUVEAU)
- **Classe** : `ModelRegistry` (wrapper MLFlow Client)
  - `register_model(run_id, version_name)` : Enregistrement auto post-training
  - `promote_to_staging(version)` : Transition de stage
  - `promote_to_production(version)` : Mise en production + reload pipeline
- **Workflow** :
  1. Training → Auto-register MLFlow
  2. Eval test set → Log metrics
  3. Review UI MLFlow
  4. Promote Staging → Production
  5. DB update + pipeline reload
- **Intégration** : `backend/app/services/training_service.py:run_training_session()`

#### 4.2 Détection de drift
- **Fichier** : `backend/app/ml/monitoring.py` (NOUVEAU)
- **Classe** : `DriftDetector`
  - Charge distribution de référence (training set)
  - Buffer prédictions récentes (1000 dernières)
  - Test Kolmogorov-Smirnov sur distributions de confidence
  - Alert si p_value < 0.05
- **Intégration** : `backend/app/api/translate.py`
  - Log chaque prédiction
  - Check drift tous les 1000 frames
  - Send alert ops team si détecté
- **Dashboard Grafana** : Distribution confidence temps réel, alertes

#### 4.3 Métriques Prometheus
- **Fichier** : `backend/app/ml/metrics.py` (NOUVEAU)
- **Métriques** :
  - `inference_total` : Counter par model_version et sign
  - `inference_latency_seconds` : Histogram avec buckets [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
  - `prediction_confidence` : Gauge par sign
- **Intégration** : `backend/app/ml/pipeline.py:_infer_window()`
- **Grafana** : Dashboard throughput, latency heatmap, confidence trends

---

### Phase 5 : Déploiement & A/B Testing (4 semaines)

**Objectif** : Déploiement sécurisé avec rollback automatique.

#### 5.1 Pipeline canary
- **Fichier** : `backend/app/api/router_v2.py` (NOUVEAU)
- **Classe** : `ModelRouter`
  - `production_pipeline` : Modèle actuel
  - `canary_pipeline` : Modèle candidat
  - `canary_percentage` : 0-100% traffic split
- **Workflow** :
  1. Deploy canary 5% → Monitor 1h
  2. Check metrics (accuracy ≥ 98% baseline)
  3. Si OK : 50% → 1h → 100% (promote)
  4. Si NOK : Rollback automatique
- **Métriques surveillées** : Accuracy, latency p95, error rate
- **Triggers rollback** : Accuracy -2%, latency +50ms, error >1%

#### 5.2 Shadow mode testing
- **Fichier** : `backend/app/api/shadow_mode.py` (NOUVEAU)
- **Classe** : `ShadowModeEvaluator`
  - Inférence production (user voit)
  - Inférence shadow (silent, logged)
  - Comparaison predictions async
  - Log disagreements
- **Use case** : Test modèles expérimentaux sans risque utilisateur
- **Métriques** : Agreement rate, disagreement analysis

---

### Phase 6 : Techniques d'Entraînement Avancées (4 semaines)

**Objectif** : Efficacité et performance d'entraînement.

#### 6.1 Curriculum Learning
- **Fichier** : `backend/app/ml/curriculum.py` (NOUVEAU)
- **Classe** : `CurriculumSampler`
  - Stratégies : `length` (séquences courtes→longues) ou `confidence` (facile→difficile)
  - Calcul scores de difficulté
  - Subset progressif par epoch
- **Intégration** : `SignTrainer.fit()` utilise curriculum subset
- **Bénéfices** : 20-30% moins d'epochs, +3-5% accuracy finale, training plus stable

#### 6.2 Knowledge Distillation
- **Fichier** : `backend/app/ml/distillation.py` (NOUVEAU)
- **Classe** : `DistillationTrainer`
  - Teacher model (1.5M params, frozen)
  - Student model (300k params, trained)
  - Loss combinée : hard targets (labels) + soft targets (teacher logits)
  - Temperature=4.0, alpha=0.5
- **Use case** : Déploiement edge (mobile, Raspberry Pi)
- **Bénéfices** : 90-95% accuracy du grand modèle, 5-10x plus rapide

#### 6.3 Test-Time Augmentation améliorée
- **Fichier** : `backend/app/ml/tta.py` (NOUVEAU)
- **Augmentations** :
  - Mirror flip (gauche↔droite)
  - Temporal jitter (±5% vitesse)
  - Spatial noise (±0.5% positions)
- **Ensemble** : Moyenne des probabilités (5 vues)
- **Intégration** : `pipeline.py` utilise TTA si configuré
- **Gain** : +2-4% accuracy, robustesse au bruit de tracking

---

### Phase 7 : Data Pipeline (4 semaines optionnel)

**Objectif** : Versioning dataset et active learning.

#### 7.1 DVC pour versioning
- **Fichiers** : `.dvc/config`, `data/videos/*.dvc`
- **Setup** : Remote S3 pour stockage datasets
- **Workflow** :
  ```bash
  dvc add data/videos/train
  git add data/videos/train.dvc
  git commit -m "Dataset v1.0"
  dvc push
  ```
- **Bénéfices** : Reproductibilité, gestion fichiers lourds, lineage modèle↔data

#### 7.2 Active Learning
- **Fichier** : `backend/app/ml/active_learning.py` (NOUVEAU)
- **Stratégie** : Uncertainty Sampling
  - Entropy ou margin-based selection
  - Top-N samples les plus incertains
  - Queue pour annotation
- **Bénéfices** : 30-50% moins d'annotations pour même accuracy

---

## Fichiers Critiques à Modifier

### Phase 1-2 (Fondations & Architecture)
- `backend/app/ml/model.py` - Ajouter SpatialTemporalTransformer, augmenter capacité
- `backend/app/ml/trainer.py` - Intégrer MLFlow tracking, curriculum learning
- `backend/app/ml/dataset.py` - TCN pooling au lieu de temporal_resample
- `backend/pyproject.toml` - Dépendances: mlflow, torch-geometric, onnx, onnxruntime

### Phase 3 (Serving)
- `backend/app/ml/pipeline.py` - ONNX inference, batching queue
- `backend/app/api/translate.py` - Remplacer singleton par batch processor
- `docker-compose.prod.yml` - Service TorchServe GPU
- `backend/torchserve/handler.py` (NOUVEAU) - Handler custom

### Phase 4-5 (MLOps & Déploiement)
- `backend/app/ml/monitoring.py` (NOUVEAU) - Drift detection
- `backend/app/ml/metrics.py` (NOUVEAU) - Prometheus metrics
- `backend/app/api/router_v2.py` (NOUVEAU) - Canary routing

---

## Métriques de Succès

### Performance Modèle
- **Accuracy** : 75% (actuel) → 90%+ (cible)
- **Latence inférence** : 50ms → <20ms (p95)
- **Throughput** : 30 req/s → 200+ req/s

### Métriques Opérationnelles
- **Training time** : Heures → <30 minutes (dataset complet)
- **Deployment frequency** : Manuel → Quotidien automatisé
- **MTTR** : Heures → <5 minutes (auto-rollback)

### Maturité MLOps
- **Experiment tracking** : 0% → 100% (tous runs loggés)
- **Model versioning** : Manuel → Registry automatisé
- **Drift detection** : Aucune → Monitoring temps réel
- **A/B testing** : Aucun → Pipeline canary automatisé

---

## Ordre de Priorité Recommandé

**MUST-HAVE (High Impact, Faisable rapidement) :**
1. Phase 1 : MLFlow + Augmentation capacité modèle + ONNX export (3 semaines)
2. Phase 2.1 : Modèle spatial-temporel (3 semaines)
3. Phase 3.1 : TorchServe deployment (2 semaines)
4. Phase 5.1 : Pipeline canary (2 semaines)

**SHOULD-HAVE (Impact moyen-élevé) :**
5. Phase 2.2 : TCN temporal pooling (2 semaines)
6. Phase 3.2 : Batching GPU (1.5 semaines)
7. Phase 4 : MLOps complet (4 semaines)

**NICE-TO-HAVE (Optimisations avancées) :**
8. Phase 2.3 : Pretraining SSL (3 semaines)
9. Phase 6 : Techniques avancées (4 semaines)
10. Phase 7 : Data pipeline (4 semaines)

**Total minimum viable (phases 1-5 prioritaires) : ~16 semaines (4 mois)**
**Total complet : ~32 semaines (8 mois)**

---

## Vérification Post-Implémentation

### Phase 1
```bash
# Vérifier MLFlow tracking
mlflow ui --backend-store-uri backend/data/models/mlruns

# Comparer modèles 150k vs 1.5M
cd backend
python scripts/compare_models.py --baseline model_150k.pt --candidate model_1.5M.pt

# Test ONNX export
python -m app.ml.export --model-path data/models/model_v1.pt --output model.onnx
python -c "import onnxruntime; ort.InferenceSession('model.onnx')"
```

### Phase 2
```bash
# Test modèle spatial-temporal
pytest backend/tests/test_ml/test_spatial_encoder.py -v
pytest backend/tests/test_ml/test_temporal_pooling.py -v

# Comparer accuracy ancien vs nouveau
python scripts/evaluate_models.py --models model_baseline.pt,model_spatial.pt
```

### Phase 3
```bash
# Vérifier TorchServe
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/predictions/signflow -d @sample_request.json

# Test batching
python scripts/benchmark_throughput.py --mode batch --batch-size 32
```

### Phase 4
```bash
# Vérifier drift detection
python scripts/simulate_drift.py --reference-dist data/reference_distribution.npy

# Dashboard Grafana
docker-compose exec grafana cat /var/lib/grafana/dashboards/signflow.json

# Prometheus metrics
curl http://localhost:9090/api/v1/query?query=signflow_inference_total
```

### Phase 5
```bash
# Test canary deployment
python scripts/deploy_canary.py --model model_v2.pt --percentage 10
python scripts/monitor_canary.py --duration 3600

# Shadow mode comparison
python scripts/shadow_mode_report.py --production model_v1.pt --shadow model_v2.pt
```

---

## Risques & Mitigations

### Risques Techniques
1. **Régression accuracy** → Mitigation: Training parallèle, A/B test, rollback auto
2. **Augmentation latence** → Mitigation: ONNX export, GPU batching, SLO monitoring
3. **Complexité infrastructure** → Mitigation: Rollout incrémental, documentation extensive

### Risques Opérationnels
1. **Temps training long** → Mitigation: Distributed training DDP, curriculum learning
2. **Échec pipeline data** → Mitigation: DVC versioning, validation automatisée
3. **Downtime déploiement** → Mitigation: Blue-green, canary, health checks

---

## Notes d'Implémentation

- **Backward compatibility** : Maintenir ancien pipeline pendant migration (dual-stack)
- **Feature flags** : Utiliser settings pour activer/désactiver nouvelles features
- **Documentation** : Mettre à jour `MEMORY.md` à chaque phase complétée
- **Tests** : Ajouter tests unitaires pour chaque nouveau composant ML
- **Monitoring** : Dashboard Grafana dès Phase 4 pour observabilité continue
