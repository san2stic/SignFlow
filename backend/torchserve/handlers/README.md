# TorchServe Handlers

Handlers personnalisés pour TorchServe avec support multi-device.

## sign_handler.py

Handler principal pour SignFlow avec support **CPU / MPS / CUDA**.

### Fonctionnalités

- **Auto-détection device** : CUDA > MPS > CPU
- **Multi-format** : PyTorch (.pt) et ONNX (.onnx)
- **Graceful fallback** : Si device non disponible
- **Response metadata** : Inclut le device utilisé

### Input Format

```json
{
  "landmarks": [
    [[x1, y1, z1], [x2, y2, z2], ...],  // Frame 1
    [[x1, y1, z1], [x2, y2, z2], ...]   // Frame 2
  ]
}
```

- `landmarks` : Liste de frames, chaque frame est une liste de landmarks [x, y, z]
- Shape attendue : `[batch_size, num_frames, num_landmarks, 3]`

### Output Format

```json
{
  "predictions": [
    {"label": "hello", "confidence": 0.95},
    {"label": "world", "confidence": 0.03},
    ...
  ],
  "device": "mps"  // or "cuda", "cpu"
}
```

### Lifecycle

```python
1. initialize(context)
   ↓
   - _detect_device()       # Auto-detect CUDA/MPS/CPU
   - _load_pytorch_model()  # or _load_onnx_model()
   ↓
2. handle(data, context)
   ↓
   - preprocess()   # landmarks → tensor
   - inference()    # model forward pass
   - postprocess()  # logits → predictions
   ↓
3. Return results
```

### Device Selection Logic

```python
def _detect_device():
    # 1. Check env variable override
    if TORCH_DEVICE in ['cuda', 'mps', 'cpu']:
        return TORCH_DEVICE

    # 2. Auto-detect
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
```

### Environment Variables

- `TORCH_DEVICE` : Force device ('cuda', 'mps', 'cpu')
- `TS_NUMBER_OF_GPU` : Number of GPUs (0=CPU)

### Customization

Pour ajouter un mapping label ID → nom de signe :

```python
def __init__(self):
    self.class_mapping = {
        0: "hello",
        1: "goodbye",
        2: "thank you",
        ...
    }

def postprocess(self, inference_output):
    ...
    label = self.class_mapping.get(idx, f"sign_{idx}")
    ...
```

### Error Handling

- **No landmarks** : ValueError avec message clair
- **Inference error** : Log + JSON avec `{"error": "..."}`
- **Device unavailable** : Fallback CPU automatique

### Performance

| Device | Cold Start | Warm Inference |
|--------|-----------|----------------|
| CUDA   | ~200ms    | 5-15ms        |
| MPS    | ~450ms    | 10-30ms       |
| CPU    | ~100ms    | 40-120ms      |

### Testing

```bash
# Test avec curl
curl -X POST http://localhost:8080/predictions/signflow_model \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [
      [[0.5, 0.5, 0.1], [0.6, 0.4, 0.15]],
      [[0.5, 0.6, 0.12], [0.61, 0.41, 0.16]]
    ]
  }'
```

### Logging

```python
logger.info(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
logger.info(f"✅ MPS (Apple Silicon GPU) detected")
logger.info(f"⚠️  No GPU detected, using CPU")
logger.error(f"❌ Inference error: {e}", exc_info=True)
```

---

**Auteur** : Bastien Javaux  
**Version** : 1.0.0  
**Compatibilité** : TorchServe 0.9+, PyTorch 2.0+
