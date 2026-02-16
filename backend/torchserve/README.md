# TorchServe Setup pour SignFlow

## Structure

- `config/config.properties` - Configuration TorchServe
- `model-store/` - Model archives (.mar)
- `handler.py` - Handler custom pour preprocessing/inference/postprocessing

## Commandes Utiles

### Créer model archive
```bash
torch-model-archiver \
  --model-name signflow \
  --version 1.0 \
  --serialized-file ../data/models/model_v1.pt \
  --handler handler.py \
  --extra-files "../app/ml/features.py,../app/ml/feature_engineering.py,../app/ml/dataset.py" \
  --export-path model-store/
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
