"""TorchServe custom handler for SignFlow sequence inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler


class SignFlowHandler(BaseHandler):
    """Serve sequence classification models expecting [batch, seq, features] input."""

    def initialize(self, context) -> None:  # type: ignore[override]
        """Load model and optional labels once at startup."""
        properties = context.system_properties
        model_dir = Path(properties.get("model_dir"))
        manifest = context.manifest
        serialized_file = manifest["model"].get("serializedFile", "")
        model_path = model_dir / serialized_file

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        )

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        labels_path = model_dir / "labels.json"
        self.labels = self._load_labels(labels_path)
        self.initialized = True

    def preprocess(self, data: list[dict[str, Any]]) -> torch.Tensor:  # type: ignore[override]
        """Parse JSON request payloads into a float tensor [batch, seq, features]."""
        windows: list[np.ndarray] = []

        for row in data:
            payload = row.get("body")
            if payload is None:
                payload = row.get("data")

            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)

            if not isinstance(payload, dict):
                raise ValueError("Invalid payload type, expected JSON object")

            window = np.asarray(payload.get("window"), dtype=np.float32)
            if window.ndim != 2:
                raise ValueError("Expected `window` with shape [seq, features]")
            windows.append(window)

        if not windows:
            raise ValueError("Empty batch received")

        target_len = int(windows[0].shape[0])
        target_features = int(windows[0].shape[1])
        normalized = [
            self._match_window_shape(window, target_len=target_len, target_features=target_features)
            for window in windows
        ]

        batch = np.stack(normalized, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(batch).to(self.device)

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        """Run a forward pass over the batch."""
        with torch.inference_mode():
            return self.model(data)

    def postprocess(self, output: torch.Tensor) -> list[dict[str, Any]]:  # type: ignore[override]
        """Convert logits into top-k predictions payload."""
        probabilities = torch.softmax(output, dim=1).detach().cpu().numpy()
        responses: list[dict[str, Any]] = []

        for probs in probabilities:
            top_k = min(4, int(probs.shape[0]))
            top_indices = np.argsort(probs)[-top_k:][::-1]

            winner_idx = int(top_indices[0])
            winner_label = self._label_for_index(winner_idx)
            winner_conf = float(probs[winner_idx])

            alternatives: list[dict[str, float]] = []
            for idx in top_indices[1:]:
                label = self._label_for_index(int(idx))
                confidence = float(probs[int(idx)])
                alternatives.append(
                    {
                        "sign": label,
                        "confidence": round(confidence, 3),
                    }
                )

            responses.append(
                {
                    "prediction": winner_label,
                    "confidence": round(winner_conf, 3),
                    "alternatives": alternatives,
                }
            )

        return responses

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load either TorchScript model or eager module checkpoint."""
        if model_path.suffix == ".pt":
            try:
                return torch.jit.load(str(model_path), map_location=self.device)
            except Exception:
                pass

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint
        raise ValueError(
            "Unsupported model format. Provide a TorchScript module or eager nn.Module checkpoint."
        )

    def _load_labels(self, labels_path: Path) -> list[str]:
        """Load class labels file when available."""
        if not labels_path.exists():
            return []
        with labels_path.open("r", encoding="utf-8") as stream:
            labels = json.load(stream)
        return [str(label) for label in labels if str(label).strip()]

    def _label_for_index(self, index: int) -> str:
        """Return label at index with stable fallback."""
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return f"class_{index}"

    def _match_window_shape(
        self,
        window: np.ndarray,
        *,
        target_len: int,
        target_features: int,
    ) -> np.ndarray:
        """Pad/trim sequence shape to match first sample in batch."""
        result = window
        if result.shape[1] != target_features:
            if result.shape[1] > target_features:
                result = result[:, :target_features]
            else:
                pad_width = target_features - result.shape[1]
                result = np.pad(result, ((0, 0), (0, pad_width)), mode="constant")

        if result.shape[0] == target_len:
            return result
        if result.shape[0] > target_len:
            return result[:target_len, :]

        padding = np.zeros((target_len - result.shape[0], target_features), dtype=np.float32)
        return np.vstack([result, padding])
