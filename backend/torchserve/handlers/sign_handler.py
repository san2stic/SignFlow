"""
TorchServe handler pour SignFlow avec support multi-device (CPU/MPS/CUDA).

Ce handler détecte automatiquement le device disponible et adapte l'inférence.
Compatible avec PyTorch 2.0+ et supporte ONNX Runtime.
"""
import os
import json
import logging
from typing import List, Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class SignLanguageHandler:
    """
    Handler TorchServe pour modèles de traduction de langue des signes.
    Supporte PyTorch (CPU/MPS/CUDA) et ONNX Runtime.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.use_onnx = False
        self.initialized = False
        self.manifest = None
        self.map_location = None

    def initialize(self, context):
        """
        Initialise le modèle et détecte le device optimal.

        Args:
            context: TorchServe context avec properties et manifest
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Détection du device optimal
        self.device = self._detect_device()
        logger.info(f"🔧 Initializing with device: {self.device}")

        # Map location pour charger le modèle
        self.map_location = self._get_map_location()

        # Chargement du manifest
        manifest_file = os.path.join(model_dir, "MAR-INF", "MANIFEST.json")
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                self.manifest = json.load(f)

        # Détection du format de modèle
        model_pt_path = os.path.join(model_dir, "model.pt")
        model_onnx_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(model_onnx_path):
            self._load_onnx_model(model_onnx_path)
        elif os.path.exists(model_pt_path):
            self._load_pytorch_model(model_pt_path)
        else:
            raise RuntimeError(f"No model found in {model_dir}")

        self.initialized = True
        logger.info(f"✅ Model initialized successfully on {self.device}")

    def _detect_device(self) -> str:
        """
        Détecte le device optimal disponible.
        Priorité: CUDA > MPS > CPU

        Returns:
            str: 'cuda', 'mps', ou 'cpu'
        """
        # Check env variable override
        env_device = os.getenv('TORCH_DEVICE', '').lower()
        if env_device in ['cuda', 'mps', 'cpu']:
            if env_device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif env_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif env_device == 'cpu':
                return 'cpu'

        # Auto-detection
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✅ MPS (Apple Silicon GPU) detected")
            return 'mps'
        else:
            logger.info("⚠️  No GPU detected, using CPU")
            return 'cpu'

    def _get_map_location(self):
        """Retourne le map_location pour torch.load selon le device."""
        if self.device == 'cuda':
            return lambda storage, loc: storage.cuda()
        elif self.device == 'mps':
            return lambda storage, loc: storage  # MPS handled by .to(device)
        else:
            return 'cpu'

    def _load_pytorch_model(self, model_path: str):
        """Charge un modèle PyTorch (.pt)."""
        logger.info(f"📦 Loading PyTorch model from {model_path}")
        self.use_onnx = False

        try:
            self.model = torch.jit.load(model_path, map_location=self.map_location)
            self.model = self.model.to(self.device)
        except Exception as e:
            # Fallback: try regular torch.load
            logger.warning(f"JIT load failed, trying torch.load: {e}")
            checkpoint = torch.load(model_path, map_location=self.map_location)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Requires model architecture definition
                raise RuntimeError("State dict requires model class definition")
            else:
                self.model = checkpoint
                self.model = self.model.to(self.device)

        self.model.training = False  # Set to inference mode

    def _load_onnx_model(self, model_path: str):
        """Charge un modèle ONNX avec le provider approprié."""
        logger.info(f"📦 Loading ONNX model from {model_path}")
        self.use_onnx = True

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")

        # Sélection du provider selon le device
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == 'mps':
            # MPS pas supporté par ONNX Runtime, fallback CPU
            logger.warning("⚠️  ONNX Runtime doesn't support MPS, using CPU")
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        logger.info(f"✅ ONNX model loaded with providers: {self.model.get_providers()}")

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Prétraite les données d'entrée.

        Args:
            data: Liste de requêtes avec 'body' contenant landmarks

        Returns:
            Tensor prétraité prêt pour l'inférence
        """
        inputs = []
        for row in data:
            body = row.get("body", {})
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)

            # Extract landmarks (format: List[List[List[float]]])
            landmarks = body.get("landmarks", [])
            if not landmarks:
                raise ValueError("No landmarks provided in request")

            # Convert to numpy array
            landmarks_array = np.array(landmarks, dtype=np.float32)
            inputs.append(landmarks_array)

        # Stack into batch
        batch = np.stack(inputs, axis=0)

        if not self.use_onnx:
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(batch)
            tensor = tensor.to(self.device)
            return tensor
        else:
            # ONNX expects numpy
            return batch

    def inference(self, data):
        """
        Effectue l'inférence.

        Args:
            data: Tensor ou numpy array prétraité

        Returns:
            Résultats d'inférence
        """
        with torch.no_grad():
            if self.use_onnx:
                # ONNX Runtime inference
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: data})
                return outputs[0]  # logits
            else:
                # PyTorch inference
                outputs = self.model(data)
                return outputs.cpu().numpy()

    def postprocess(self, inference_output) -> List[Dict[str, Any]]:
        """
        Post-traite les résultats d'inférence.

        Args:
            inference_output: Logits du modèle

        Returns:
            Liste de prédictions formatées
        """
        # Convert to probabilities
        probs = torch.softmax(torch.from_numpy(inference_output), dim=-1).numpy()

        # Get top predictions
        top_k = 5
        results = []

        for prob_dist in probs:
            top_indices = np.argsort(prob_dist)[-top_k:][::-1]
            predictions = []

            for idx in top_indices:
                predictions.append({
                    "label": str(idx),  # Would map to sign label with class_mapping
                    "confidence": float(prob_dist[idx])
                })

            results.append({
                "predictions": predictions,
                "device": self.device
            })

        return results

    def handle(self, data, context):
        """Point d'entrée principal pour les requêtes."""
        if not self.initialized:
            self.initialize(context)

        try:
            preprocessed = self.preprocess(data)
            inference_output = self.inference(preprocessed)
            return self.postprocess(inference_output)
        except Exception as e:
            logger.error(f"❌ Inference error: {e}", exc_info=True)
            return [{"error": str(e), "device": self.device}]


# TorchServe entry point
_service = SignLanguageHandler()


def handle(data, context):
    """Entry point pour TorchServe."""
    return _service.handle(data, context)
