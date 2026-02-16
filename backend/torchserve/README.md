# TorchServe Setup pour SignFlow

## Structure

- `config/config.properties` - Configuration TorchServe
- `model-store/` - Model archives (.mar)
- `handler.py` - Handler custom pour preprocessing/inference/postprocessing
- `labels.json` - Labels par défaut pour sortie top-k

## Commandes Utiles

### Créer model archive (recommandé)
```bash
cd backend
./scripts/package_torchserve_model.sh ./data/models/model_v1.pt
```

### Créer model archive (manuel)
```bash
cd backend
torch-model-archiver \
  --model-name signflow \
  --version 1.0 \
  --serialized-file ./data/models/model_v1.pt \
  --handler ./torchserve/handler.py \
  --extra-files ./torchserve/labels.json \
  --export-path ./torchserve/model-store \
  --force
```

### Enregistrer modèle via Management API
```bash
curl -X POST "http://localhost:8081/models?url=signflow.mar&initial_workers=2&batch_size=16&max_batch_delay=50"
```

### Vérifier santé
```bash
curl http://localhost:8080/ping
curl http://localhost:8081/models
```

### Métriques Prometheus
```bash
curl http://localhost:8082/metrics | grep ts_inference
```
