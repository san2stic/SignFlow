"""Inference pipeline for real-time sign translation from landmarks."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import structlog
import torch

from app.ml.feature_alignment import align_numpy_features, resolve_model_feature_dim
from app.ml.features import (
    FrameLandmarks,
    normalize_landmarks,
    extract_features_v2,
    ENRICHED_FEATURE_DIM_V2,
)
from app.ml.model import SignTransformer
from app.ml.tta import TTAConfig, TTAGenerator
from app.ml.trainer import load_model_checkpoint
from app.ml.gpu_manager import get_gpu_manager

# Optional V2 model and segmentation imports (lazy to avoid circular deps)
try:
    from app.ml.model_v2 import SignTransformerV2
    _MODEL_V2_AVAILABLE = True
except Exception:
    _MODEL_V2_AVAILABLE = False

# Optional grammar translation (Phase 3 — requires only numpy, always attempted)
try:
    from app.ml.grammar.lsfb_translator import LSFBToFrenchTranslator
    _GRAMMAR_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GRAMMAR_AVAILABLE = False
    LSFBToFrenchTranslator = None  # type: ignore[assignment,misc]

# Optional conversation context (Phase 5)
try:
    from app.ml.conversation_context import ConversationContext
    _CONVERSATION_CONTEXT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _CONVERSATION_CONTEXT_AVAILABLE = False
    ConversationContext = None  # type: ignore[assignment,misc]

try:
    from app.ml.sign_segmentation import SignBoundaryDetector, motion_energy_fallback
    _SEGMENTATION_AVAILABLE = True
except Exception:
    _SEGMENTATION_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.debug("onnxruntime_not_available", msg="ONNXRuntime not installed")


class InferenceState(Enum):
    """States for the inference state machine."""

    IDLE = "idle"
    RECORDING = "recording"
    INFERRING = "inferring"


@dataclass
class Prediction:
    """Prediction payload returned by translation pipeline."""

    prediction: str
    confidence: float
    alternatives: list[dict[str, float]]
    sentence_buffer: str
    is_sentence_complete: bool
    decision_diagnostics: dict[str, object] | None = None
    # Grammar translation outputs (Phase 3 — optional, populated when grammar_translator is active)
    translated_sentence: str | None = None   # Phrase française avec grammaire LSFB appliquée
    grammar_tags: list[str] | None = None    # Tags BIO CRF (debug/visualisation frontend)
    translation_mode: str | None = None      # 'rules' | 'crf' | 'seq2seq'
    # Conversation context (Phase 5 — optional, populated when conversation_context is active)
    turn_id: int | None = None               # Identifiant du tour de parole courant
    is_new_turn: bool = False                # True si nouveau tour détecté
    conversation_context: dict | None = None # Résumé du contexte pour le frontend


class SignFlowInferencePipeline:
    """Sliding-window inference with confidence threshold and temporal smoothing."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_version: str | None = None,
        seq_len: int = 64,
        confidence_threshold: float = 0.55,
        smoothing_window: int = 5,
        min_hand_visibility: float = 0.15,
        min_prediction_margin: float = 0.05,
        min_motion_energy: float = 0.002,
        device: str = "cpu",
        max_buffer_frames: int = 180,
        rest_frames_threshold: int = 10,
        motion_start_threshold: float = 0.005,
        min_recording_frames: int = 15,
        pre_roll_frames: int = 4,
        frontend_confidence_floor: float = 0.35,
        inference_num_views: int = 1,
        inference_temperature: float = 1.0,
        max_view_disagreement: float = 0.35,
        tta_enable_mirror: bool = True,
        tta_enable_temporal_jitter: bool = True,
        tta_enable_spatial_noise: bool = True,
        tta_temporal_jitter_ratio: float = 0.05,
        tta_spatial_noise_std: float = 0.005,
        calibration_temperature: float | None = None,
        class_thresholds: dict[str, float] | None = None,
        feature_version: int = 1,
    ) -> None:
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to model checkpoint. If None, pipeline runs without model.
            seq_len: Target sequence length for model input (default: 64 frames)
            confidence_threshold: Minimum confidence to accept prediction (default: 0.7)
            smoothing_window: Temporal smoothing window size in frames.
            min_hand_visibility: Minimum rolling hand visibility required to infer.
            min_prediction_margin: Minimum top1-top2 probability margin before accepting.
            min_motion_energy: Minimum hand motion level for stable confidence.
            device: Device to run inference on ("cpu" or "cuda")
            max_buffer_frames: Maximum frames to buffer before forcing inference.
            rest_frames_threshold: Consecutive low-motion frames to detect sign end.
            motion_start_threshold: Motion energy threshold to start recording.
            min_recording_frames: Minimum frames recorded before allowing sign end.
            pre_roll_frames: Number of frames to keep before motion onset.
            frontend_confidence_floor: Confidence floor for frontend landmarks quality.
            inference_num_views: Number of temporal views for test-time ensembling.
            inference_temperature: Softmax temperature used for calibrated probabilities.
            max_view_disagreement: Max tolerated view disagreement before confidence penalty.
            tta_enable_mirror: Whether to include left/right mirrored views.
            tta_enable_temporal_jitter: Whether to include temporal speed perturbations.
            tta_enable_spatial_noise: Whether to include spatial landmark noise.
            tta_temporal_jitter_ratio: Max relative speed perturbation for temporal jitter.
            tta_spatial_noise_std: Gaussian std for spatial perturbation.
            feature_version: Feature extraction version. 1 = legacy (493 dims),
                2 = enriched V2 with NMM/handshape/signing-space (611 dims).
                Can also be auto-detected from model metadata via ``set_feature_version_from_model``.
        """
        self.seq_len = seq_len
        self.model_version = str(model_version or "unknown")
        self.confidence_threshold = confidence_threshold
        self.min_hand_visibility = min_hand_visibility
        self.min_prediction_margin = min_prediction_margin
        self.min_motion_energy = min_motion_energy
        self.pre_roll_frames = max(0, pre_roll_frames)
        self.frontend_confidence_floor = float(np.clip(frontend_confidence_floor, 0.0, 1.0))
        self.inference_num_views = max(1, int(inference_num_views))
        self.inference_temperature = float(max(0.1, inference_temperature))
        self.max_view_disagreement = float(max(1e-6, max_view_disagreement))
        self.tta_enable_mirror = bool(tta_enable_mirror)
        self.tta_enable_temporal_jitter = bool(tta_enable_temporal_jitter)
        self.tta_enable_spatial_noise = bool(tta_enable_spatial_noise)
        self.tta_temporal_jitter_ratio = float(np.clip(tta_temporal_jitter_ratio, 0.0, 0.3))
        self.tta_spatial_noise_std = float(max(0.0, tta_spatial_noise_std))
        self._tta_generator = TTAGenerator(
            TTAConfig(
                num_views=self.inference_num_views,
                enable_mirror=self.tta_enable_mirror,
                enable_temporal_jitter=self.tta_enable_temporal_jitter,
                enable_spatial_noise=self.tta_enable_spatial_noise,
                temporal_jitter_ratio=self.tta_temporal_jitter_ratio,
                spatial_noise_std=self.tta_spatial_noise_std,
            )
        )
        self.calibration_temperature = (
            float(max(0.1, calibration_temperature))
            if calibration_temperature is not None
            else None
        )
        self.class_thresholds = {
            str(key): float(np.clip(value, 0.0, 1.0))
            for key, value in (class_thresholds or {}).items()
        }

        # Setup device with GPU manager support
        if device == "auto":
            gpu_manager = get_gpu_manager()
            self.device = gpu_manager.get_device()
        else:
            self.device = torch.device(device)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max_buffer_frames)
        self.hand_visibility_history: deque[float] = deque(maxlen=seq_len)
        self.motion_history: deque[float] = deque(maxlen=seq_len)
        self.prediction_history: deque[tuple[str, float]] = deque(maxlen=max(3, smoothing_window))
        self.labels: list[str] = []
        self.sentence_tokens: list[str] = []
        self._current_motion_energy = 0.0
        self._decision_counters: dict[str, int] = {
            "total_inferences": 0,
            "accepted": 0,
            "rejected_total": 0,
            "rejected_by_confidence_threshold": 0,
            "rejected_by_adaptive_threshold": 0,
            "rejected_by_class_threshold": 0,
            "rejected_by_label_none": 0,
            "rejected_by_calibration": 0,
            "rejected_by_margin": 0,
            "rejected_by_motion": 0,
            "rejected_by_tta_disagreement": 0,
        }
        self._last_decision_trace: dict[str, object] = {
            "status": "idle",
            "reason": "not_started",
        }

        # State machine attributes
        self.max_buffer_frames = max_buffer_frames
        self.rest_frames_threshold = rest_frames_threshold
        self.motion_start_threshold = motion_start_threshold
        self.min_recording_frames = min_recording_frames
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self._latest_frontend_confidence = 1.0

        # Feature extraction version (1 = V1 legacy, 2 = V2 enriched)
        self.feature_version: int = int(feature_version) if feature_version in (1, 2) else 1
        # Track previous wrist positions for signing-space motion features (V2 only)
        self._prev_right_wrist: np.ndarray | None = None
        self._prev_left_wrist: np.ndarray | None = None

        if self.feature_version == 2:
            logger.info(
                "pipeline_feature_version_v2",
                enriched_dim=ENRICHED_FEATURE_DIM_V2,
                nmm_enabled=True,
                handshape_enabled=True,
                signing_space_enabled=True,
            )

        # Load model if path provided (V1 or V2 — resolved at load time)
        self.model: SignTransformer | SignTransformerV2 | None = None  # type: ignore[assignment]
        self.onnx_session: ort.InferenceSession | None = None
        self.use_onnx = False
        self._model_is_v2: bool = False

        # Optional segmentation model (BiLSTM boundary detector)
        self.segmentation_model: SignBoundaryDetector | None = None  # type: ignore[assignment]
        self._use_bilstm_segmentation: bool = False

        # Optional grammar translator (Phase 3 — LSFB → Français)
        self.grammar_translator: LSFBToFrenchTranslator | None = None  # type: ignore[assignment]

        # Conversation context (Phase 5 — maintains cross-turn conversational state)
        self.conversation_context: ConversationContext | None = None  # type: ignore[assignment]

        if model_path:
            self.load_model(model_path)

    def set_calibration(
        self,
        *,
        calibration_temperature: float | None = None,
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Configure optional calibration temperature and per-class thresholds."""
        if calibration_temperature is not None:
            self.calibration_temperature = float(max(0.1, calibration_temperature))
        if class_thresholds is not None:
            self.class_thresholds = {
                str(key): float(np.clip(value, 0.0, 1.0))
                for key, value in class_thresholds.items()
            }

    def load_model(self, model_path: str | Path) -> None:
        """
        Load model from checkpoint (supports both PyTorch .pt and ONNX .onnx).

        Args:
            model_path: Path to .pt or .onnx checkpoint file
        """
        model_path = Path(model_path)

        # Check if ONNX model
        if model_path.suffix == ".onnx":
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model from checkpoint (auto-detects V1/V2 architecture)."""
        try:
            logger.info("loading_pytorch_model", path=str(model_path))

            # Peek at checkpoint to decide architecture
            try:
                _ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
                _arch = _ckpt.get("architecture", "")
            except Exception:
                _arch = ""

            if _arch == "SignTransformerV2" and _MODEL_V2_AVAILABLE:
                self.model = SignTransformerV2.load_checkpoint(  # type: ignore[assignment]
                    model_path, device=str(self.device), strict=True
                )
                self._model_is_v2 = True
                # Auto-activate V2 feature pipeline
                if self.feature_version != 2:
                    self.feature_version = 2
                    logger.info("feature_version_auto_set_v2", reason="v2_checkpoint_detected")
            else:
                self.model = load_model_checkpoint(str(model_path), device=str(self.device))
                self._model_is_v2 = False

            self.model.to(self.device)
            self.model.set_inference_mode()
            self.use_onnx = False
            self.onnx_session = None
            logger.info(
                "pytorch_model_loaded_successfully",
                num_classes=self.model.num_classes,
                architecture="SignTransformerV2" if self._model_is_v2 else "SignTransformer",
                device=str(self.device),
            )
        except Exception as e:
            logger.error("failed_to_load_pytorch_model", path=str(model_path), error=str(e))
            self.model = None
            raise

    def load_segmentation_model(self, model_path: str | Path) -> None:
        """Load an optional SignBoundaryDetector checkpoint for BiLSTM segmentation.

        When loaded, replaces the heuristic motion_energy segmentation with the
        learned BiLSTM boundary detector.

        Args:
            model_path: Path to SignBoundaryDetector .pt checkpoint.
        """
        if not _SEGMENTATION_AVAILABLE:
            logger.warning(
                "segmentation_model_load_skipped",
                reason="sign_segmentation module not available",
            )
            return
        try:
            from app.ml.sign_segmentation import SignBoundaryDetector
            self.segmentation_model = SignBoundaryDetector.load_checkpoint(
                model_path, device=str(self.device)
            )
            self.segmentation_model.set_inference_mode()
            self._use_bilstm_segmentation = True
            logger.info("segmentation_model_loaded", path=str(model_path))
        except Exception as e:
            logger.error("segmentation_model_load_failed", path=str(model_path), error=str(e))
            self.segmentation_model = None
            self._use_bilstm_segmentation = False

    def enable_conversation_context(
        self,
        max_history: int = 20,
        turn_gap_seconds: float = 3.0,
    ) -> None:
        """Active le contexte conversationnel pour cette session.

        Args:
            max_history: Nombre maximum de tours gardés en mémoire.
            turn_gap_seconds: Durée de silence pour détecter un nouveau tour.
        """
        if not _CONVERSATION_CONTEXT_AVAILABLE or ConversationContext is None:
            logger.warning(
                "conversation_context_unavailable",
                reason="app.ml.conversation_context module could not be imported",
            )
            return
        try:
            self.conversation_context = ConversationContext(
                max_history=max_history,
                turn_gap_seconds=turn_gap_seconds,
            )
            logger.info(
                "conversation_context_enabled",
                max_history=max_history,
                turn_gap_seconds=turn_gap_seconds,
            )
        except Exception as exc:
            logger.error("conversation_context_init_failed", error=str(exc))
            self.conversation_context = None

    def enable_grammar_translation(
        self,
        mode: str = "rules",
        crf_path: str | None = None,
    ) -> None:
        """Active la traduction grammaticale LSFB → Français.

        Args:
            mode: Stratégie de traduction parmi ``'rules'``, ``'crf'``,
                ``'seq2seq'``.
            crf_path: Chemin optionnel vers un modèle CRF pré-entraîné
                (fichier pickle produit par ``LSFBSequenceTagger.save()``).

        Si le module de grammaire n'est pas disponible, un avertissement est
        loggé et la traduction reste désactivée.
        """
        if not _GRAMMAR_AVAILABLE:
            logger.warning(
                "grammar_translation_unavailable",
                reason="app.ml.grammar module could not be imported",
            )
            return
        try:
            self.grammar_translator = LSFBToFrenchTranslator(
                mode=mode,
                crf_model_path=crf_path,
            )
            logger.info(
                "grammar_translation_enabled",
                mode=self.grammar_translator.mode,
                crf_loaded=self.grammar_translator.sequence_tagger.crf is not None,
            )
        except Exception as exc:
            logger.error("grammar_translation_init_failed", error=str(exc))
            self.grammar_translator = None

    def _translate_sentence_buffer(
        self,
        token_confidences: list[float] | None = None,
    ) -> tuple[str | None, list[str] | None, str | None]:
        """Traduit les tokens courants du sentence_buffer via le traducteur grammatical.

        Args:
            token_confidences: Confidences individuelles par token (même longueur que
                ``sentence_tokens``). Si ``None``, une valeur par défaut de 0.7 est utilisée.

        Returns:
            Tuple (translated_sentence, grammar_tags, translation_mode) ou
            (None, None, None) si la traduction est désactivée ou échoue.
        """
        if self.grammar_translator is None or not self.sentence_tokens:
            return None, None, None
        try:
            if token_confidences is None or len(token_confidences) != len(self.sentence_tokens):
                confs = [0.7] * len(self.sentence_tokens)
            else:
                confs = list(token_confidences)
            token_dicts = [
                {"label": label, "confidence": conf}
                for label, conf in zip(self.sentence_tokens, confs)
            ]
            result = self.grammar_translator.translate_buffer(token_dicts)
            # Retourner aussi les tags depuis la dernière invocation du tagger
            last_tags: list[str] | None = None
            if hasattr(self.grammar_translator, "sequence_tagger"):
                tagger = self.grammar_translator.sequence_tagger
                if hasattr(tagger, "_last_tags"):
                    last_tags = list(tagger._last_tags)
            return result, last_tags, self.grammar_translator.mode
        except Exception as exc:
            logger.warning("grammar_translation_failed", error=str(exc))
            return None, None, None

    def _flush_complete_turn(
        self,
        translated_text: str,
        grammar_tags: list[str] | None,
        confidence: float,
    ) -> tuple[int | None, bool, dict | None]:
        """Enregistre un tour terminé dans le contexte conversationnel.

        Appelé après qu'une phrase complète a été traduite (``is_sentence_complete``
        ou seuil de tokens atteint).

        Args:
            translated_text: Texte français traduit.
            grammar_tags: Tags BIO du CRF.
            confidence: Confiance agrégée du tour.

        Returns:
            Tuple (turn_id, is_new_turn, context_summary).
        """
        if self.conversation_context is None:
            return None, False, None

        is_new = self.conversation_context.is_new_turn()

        # Résolution anaphorique avant enregistrement
        resolved_text = self.conversation_context.resolve_anaphora(translated_text)
        if resolved_text != translated_text:
            logger.debug(
                "anaphora_resolved_in_turn",
                original=translated_text[:60],
                resolved=resolved_text[:60],
            )

        turn = self.conversation_context.add_turn(
            text=resolved_text,
            raw_signs=list(self.sentence_tokens),
            grammar_tags=list(grammar_tags or []),
            confidence=confidence,
        )
        context_summary = self.conversation_context.get_context_summary()
        return turn.id, is_new, context_summary

    def _load_onnx_model(self, model_path: Path) -> None:
        """Load ONNX model for inference."""
        if not ONNX_AVAILABLE:
            logger.error(
                "onnx_load_failed",
                reason="onnxruntime_not_installed",
                path=str(model_path)
            )
            raise ImportError("onnxruntime is required for ONNX model inference")

        try:
            logger.info("loading_onnx_model", path=str(model_path))

            # Determine execution providers based on device
            if self.device.type == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Create ONNX Runtime session
            self.onnx_session = ort.InferenceSession(
                str(model_path),
                providers=providers
            )

            # Clear PyTorch model
            self.model = None
            self.use_onnx = True

            # Get model info
            input_shape = self.onnx_session.get_inputs()[0].shape
            output_shape = self.onnx_session.get_outputs()[0].shape

            logger.info(
                "onnx_model_loaded_successfully",
                path=str(model_path),
                input_shape=input_shape,
                output_shape=output_shape,
                providers=self.onnx_session.get_providers(),
            )

        except Exception as e:
            logger.error("failed_to_load_onnx_model", path=str(model_path), error=str(e))
            self.onnx_session = None
            raise

    def set_labels(self, labels: list[str]) -> None:
        """
        Configure active vocabulary labels for model outputs.

        Args:
            labels: List of sign labels (class names)
        """
        cleaned = [str(label) for label in labels if label]

        if self.model is None:
            self.labels = cleaned
            logger.debug("labels_set", num_labels=len(self.labels))
            return

        expected = int(self.model.num_classes)
        aligned = cleaned
        if len(aligned) != expected:
            logger.warning(
                "label_count_mismatch",
                expected=expected,
                provided=len(aligned),
            )
            aligned = aligned[:expected]
            while len(aligned) < expected:
                aligned.append(f"class_{len(aligned)}")

        self.labels = aligned
        logger.debug("labels_set", num_labels=len(self.labels))

    def set_feature_version_from_model(self, model_metadata: dict | None = None) -> None:
        """Auto-detect feature version from model metadata.

        Reads the ``feature_version`` key from model metadata dict (stored in
        ``ModelVersion.extra_metadata`` or checkpoint ``config``). Falls back to
        checking ``input_dim`` against known constants.

        Args:
            model_metadata: Optional dict with a ``feature_version`` key.
                If None, attempts to read from self.model's config when available.
        """
        if model_metadata and "feature_version" in model_metadata:
            v = int(model_metadata["feature_version"])
            if v in (1, 2):
                self.feature_version = v
                logger.info("feature_version_from_metadata", feature_version=v)
                return

        # Fallback: detect from model input_dim
        if self.model is not None:
            try:
                input_dim = int(self.model.input_dim)  # type: ignore[attr-defined]
                if input_dim == ENRICHED_FEATURE_DIM_V2:
                    self.feature_version = 2
                    logger.info("feature_version_autodetected", feature_version=2, input_dim=input_dim)
                    return
            except AttributeError:
                pass

        logger.debug("feature_version_unchanged", feature_version=self.feature_version)

    def spawn_session(self) -> SignFlowInferencePipeline:
        """Create an isolated per-session pipeline sharing only immutable runtime artifacts."""
        session_pipeline = SignFlowInferencePipeline(
            model_path=None,
            model_version=self.model_version,
            seq_len=self.seq_len,
            confidence_threshold=self.confidence_threshold,
            smoothing_window=int(self.prediction_history.maxlen or 5),
            min_hand_visibility=self.min_hand_visibility,
            min_prediction_margin=self.min_prediction_margin,
            min_motion_energy=self.min_motion_energy,
            device=str(self.device),
            max_buffer_frames=self.max_buffer_frames,
            rest_frames_threshold=self.rest_frames_threshold,
            motion_start_threshold=self.motion_start_threshold,
            min_recording_frames=self.min_recording_frames,
            pre_roll_frames=self.pre_roll_frames,
            frontend_confidence_floor=self.frontend_confidence_floor,
            inference_num_views=self.inference_num_views,
            inference_temperature=self.inference_temperature,
            max_view_disagreement=self.max_view_disagreement,
            tta_enable_mirror=self.tta_enable_mirror,
            tta_enable_temporal_jitter=self.tta_enable_temporal_jitter,
            tta_enable_spatial_noise=self.tta_enable_spatial_noise,
            tta_temporal_jitter_ratio=self.tta_temporal_jitter_ratio,
            tta_spatial_noise_std=self.tta_spatial_noise_std,
            calibration_temperature=self.calibration_temperature,
            class_thresholds=dict(self.class_thresholds),
            feature_version=self.feature_version,
        )
        session_pipeline.model = self.model
        session_pipeline.onnx_session = self.onnx_session
        session_pipeline.use_onnx = self.use_onnx
        session_pipeline.set_labels(list(self.labels))
        return session_pipeline

    def process_frame(self, payload: dict) -> Prediction:
        """Process a landmarks frame through the state machine."""
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        # Use metadata if available (from improved frontend detection)
        metadata = payload.get("metadata", {})
        frontend_confidence = self._resolve_frontend_confidence(metadata)
        self._latest_frontend_confidence = frontend_confidence

        # Log frontend confidence for monitoring (optional)
        if frontend_confidence < 0.3:
            logger.debug(
                "low_frontend_confidence",
                confidence=round(frontend_confidence, 3),
                left_visible=metadata.get("leftHandVisible", False),
                right_visible=metadata.get("rightHandVisible", False),
            )

        raw_hand_visibility = self._compute_hand_visibility(frame)
        hand_visibility = self._blend_hand_visibility(raw_hand_visibility, frontend_confidence)
        self.hand_visibility_history.append(hand_visibility)

        if self.feature_version == 2:
            features = extract_features_v2(
                payload,
                include_handshape=True,
                include_nmm=True,
                include_signing_space=True,
                prev_right_wrist=self._prev_right_wrist,
                prev_left_wrist=self._prev_left_wrist,
            )
            # Update wrist history for next-frame motion estimation
            import numpy as _np  # local alias to avoid shadowing
            _rh = payload.get("hands", {}).get("right", [])
            _lh = payload.get("hands", {}).get("left", [])
            if _rh:
                _rh_arr = _np.array(_rh[:3] if len(_rh) >= 3 else _rh, dtype=_np.float32).reshape(-1)
                self._prev_right_wrist = _rh_arr[:3] if _rh_arr.shape[0] >= 3 else None
            if _lh:
                _lh_arr = _np.array(_lh[:3] if len(_lh) >= 3 else _lh, dtype=_np.float32).reshape(-1)
                self._prev_left_wrist = _lh_arr[:3] if _lh_arr.shape[0] >= 3 else None
        else:
            features = normalize_landmarks(
                frame,
                include_face=False,
                include_face_expressions=True,
            )
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        # State machine
        if self.state == InferenceState.IDLE:
            if (self._current_motion_energy > self.motion_start_threshold
                    and hand_visibility > self.min_hand_visibility):
                if self.pre_roll_frames > 0:
                    context_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                    self.frame_buffer.clear()
                    self.frame_buffer.extend(context_frames)
                else:
                    self.frame_buffer.clear()
                    self.frame_buffer.append(features)
                self.state = InferenceState.RECORDING
                self._recording_frame_count = 1
                self._rest_frame_count = 0
            return self._idle_prediction()

        if self.state == InferenceState.RECORDING:
            self._recording_frame_count += 1

            is_resting = self._current_motion_energy < self.motion_start_threshold
            if is_resting:
                self._rest_frame_count += 1
            else:
                self._rest_frame_count = 0

            sign_ended = (
                (self._rest_frame_count >= self.rest_frames_threshold
                 and self._recording_frame_count >= self.min_recording_frames)
                or len(self.frame_buffer) >= self.max_buffer_frames
            )

            if sign_ended:
                return self._infer_complete_sign()

            return self._recording_prediction()

        # INFERRING state fallback
        self.state = InferenceState.IDLE
        return self._idle_prediction()

    async def process_frame_async(
        self,
        payload: dict,
        *,
        infer_window_async: Callable[
            [np.ndarray], Awaitable[tuple[str, float, list[dict[str, float]]]]
        ] | None = None,
    ) -> Prediction:
        """
        Async variant of process_frame.

        Args:
            payload: Frame landmarks payload
            infer_window_async: Optional async backend used at sign-end inference.
                If omitted, falls back to local model inference.
        """
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        metadata = payload.get("metadata", {})
        frontend_confidence = self._resolve_frontend_confidence(metadata)
        self._latest_frontend_confidence = frontend_confidence

        if frontend_confidence < 0.3:
            logger.debug(
                "low_frontend_confidence",
                confidence=round(frontend_confidence, 3),
                left_visible=metadata.get("leftHandVisible", False),
                right_visible=metadata.get("rightHandVisible", False),
            )

        raw_hand_visibility = self._compute_hand_visibility(frame)
        hand_visibility = self._blend_hand_visibility(raw_hand_visibility, frontend_confidence)
        self.hand_visibility_history.append(hand_visibility)

        features = normalize_landmarks(
            frame,
            include_face=False,
            include_face_expressions=True,
        )
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        if self.state == InferenceState.IDLE:
            if (
                self._current_motion_energy > self.motion_start_threshold
                and hand_visibility > self.min_hand_visibility
            ):
                if self.pre_roll_frames > 0:
                    context_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                    self.frame_buffer.clear()
                    self.frame_buffer.extend(context_frames)
                else:
                    self.frame_buffer.clear()
                    self.frame_buffer.append(features)
                self.state = InferenceState.RECORDING
                self._recording_frame_count = 1
                self._rest_frame_count = 0
            return self._idle_prediction()

        if self.state == InferenceState.RECORDING:
            self._recording_frame_count += 1

            is_resting = self._current_motion_energy < self.motion_start_threshold
            if is_resting:
                self._rest_frame_count += 1
            else:
                self._rest_frame_count = 0

            sign_ended = (
                (
                    self._rest_frame_count >= self.rest_frames_threshold
                    and self._recording_frame_count >= self.min_recording_frames
                )
                or len(self.frame_buffer) >= self.max_buffer_frames
            )

            if sign_ended:
                return await self._infer_complete_sign_async(
                    infer_window_async=infer_window_async
                )

            return self._recording_prediction()

        self.state = InferenceState.IDLE
        return self._idle_prediction()

    def _idle_prediction(self) -> Prediction:
        """Return a neutral prediction for the IDLE state."""
        return Prediction(
            prediction="NONE",
            confidence=0.0,
            alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _recording_prediction(self) -> Prediction:
        """Return a recording-in-progress prediction."""
        return Prediction(
            prediction="RECORDING",
            confidence=0.0,
            alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _infer_complete_sign(self) -> Prediction:
        """Run inference on the complete recorded sign."""
        enriched = self._prepare_enriched_window()
        if enriched is None:
            self.state = InferenceState.IDLE
            return self._idle_prediction()

        predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)
        return self._build_prediction_response(
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            alternatives=alternatives,
        )

    async def _infer_complete_sign_async(
        self,
        *,
        infer_window_async: Callable[
            [np.ndarray], Awaitable[tuple[str, float, list[dict[str, float]]]]
        ] | None = None,
    ) -> Prediction:
        """Run inference on the complete recorded sign using an optional async backend."""
        enriched = self._prepare_enriched_window()
        if enriched is None:
            self.state = InferenceState.IDLE
            return self._idle_prediction()

        if infer_window_async is None:
            predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)
        else:
            predicted_label, predicted_confidence, alternatives = await infer_window_async(enriched)

        return self._build_prediction_response(
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            alternatives=alternatives,
        )

    def _prepare_enriched_window(self) -> np.ndarray | None:
        """Build an enriched feature window from the buffered raw landmarks.

        When feature_version == 2, the buffer already contains V2 (611-dim) features
        extracted by extract_features_v2(). We only need to resample to seq_len.
        For V1, we additionally apply compute_enriched_features().
        """
        from app.ml.dataset import temporal_resample

        if not self.frame_buffer:
            return None

        buffer_array = np.stack(list(self.frame_buffer), axis=0)
        trimmed = self._trim_recording_window(buffer_array, trailing_rest=self._rest_frame_count)
        resampled = temporal_resample(trimmed, target_len=self.seq_len)

        if self.feature_version == 2:
            # V2 features already complete; no additional enrichment needed
            return resampled

        # V1 legacy path
        from app.ml.feature_engineering import compute_enriched_features
        return compute_enriched_features(resampled)

    def _infer_window_v2(self, window: np.ndarray) -> tuple[str, float, list[dict[str, float]]]:
        """Run inference using SignTransformerV2 on a V2 feature window.

        This mirrors _infer_window() but asserts the model is V2.
        Falls back to _infer_window() if model is not V2.

        Args:
            window: (seq_len, 611) V2 feature array

        Returns:
            (prediction_label, confidence, alternatives)
        """
        if not self._model_is_v2 or self.model is None:
            return self._infer_window(window)

        # V2 inference uses the same probability pipeline — model is already V2
        return self._infer_window(window)

    def detect_boundaries_bilstm(
        self,
        feature_sequence: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Detect sign boundaries using the BiLSTM segmentation model.

        Falls back to motion_energy heuristic if no segmentation model is loaded.

        Args:
            feature_sequence: (seq_len, 611) V2 feature array

        Returns:
            List of (start_frame, end_frame) tuples (inclusive, 0-indexed)
        """
        if self._use_bilstm_segmentation and self.segmentation_model is not None:
            try:
                return self.segmentation_model.detect_boundaries(
                    feature_sequence,
                    device=str(self.device),
                )
            except Exception as e:
                logger.warning(
                    "bilstm_segmentation_failed_fallback",
                    error=str(e),
                )

        # Fallback: heuristic motion energy
        if _SEGMENTATION_AVAILABLE:
            return motion_energy_fallback(
                feature_sequence,
                motion_start_threshold=self.motion_start_threshold,
                rest_frames_threshold=self.rest_frames_threshold,
                min_recording_frames=self.min_recording_frames,
            )

        # Simple built-in fallback
        return self._motion_energy_segmentation_simple(feature_sequence)

    def _motion_energy_segmentation_simple(
        self, feature_sequence: np.ndarray
    ) -> list[tuple[int, int]]:
        """In-line motion energy segmentation (no external dependency)."""
        seq_len = feature_sequence.shape[0]
        # Use velocity block if feature is V2, else use all features
        if feature_sequence.shape[1] > 462:
            vel = feature_sequence[:, 237:462]
        else:
            vel = feature_sequence
        me = np.linalg.norm(vel, axis=1) / max(vel.shape[1], 1)

        segments: list[tuple[int, int]] = []
        in_sign = False
        start = 0
        count = 0
        rest_count = 0
        for i in range(seq_len):
            if not in_sign:
                if me[i] > self.motion_start_threshold:
                    in_sign = True
                    start = i
                    count = 1
                    rest_count = 0
            else:
                count += 1
                if me[i] < self.motion_start_threshold:
                    rest_count += 1
                else:
                    rest_count = 0
                if rest_count >= self.rest_frames_threshold and count >= self.min_recording_frames:
                    segments.append((start, i - rest_count))
                    in_sign = False
        if in_sign and count >= self.min_recording_frames:
            segments.append((start, seq_len - 1))
        return segments

    def _normalize_alternatives(self, alternatives: list[dict[str, float]] | None) -> list[dict[str, float]]:
        """Normalize alternatives payload shape and numeric confidence."""
        normalized: list[dict[str, float]] = []
        for alternative in alternatives or []:
            if not isinstance(alternative, dict):
                continue
            sign = alternative.get("sign") or alternative.get("prediction")
            if not sign:
                continue
            try:
                confidence = float(alternative.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            normalized.append({"sign": str(sign), "confidence": round(confidence, 3)})
        return normalized

    def _build_prediction_response(
        self,
        *,
        predicted_label: str,
        predicted_confidence: float,
        alternatives: list[dict[str, float]] | None,
    ) -> Prediction:
        """Apply post-filters, reset recording state, and produce final response payload."""
        normalized_label = str(predicted_label or "NONE")
        normalized_confidence = float(predicted_confidence or 0.0)
        normalized_alternatives = self._normalize_alternatives(alternatives)

        adaptive_threshold = self._adaptive_threshold()
        label_threshold = self._threshold_for_label(normalized_label)
        quality_delta = max(0.0, float(adaptive_threshold) - float(self.confidence_threshold))
        if label_threshold > 0:
            # Per-class thresholds define the baseline; quality penalties still tighten it.
            effective_threshold = min(0.95, float(label_threshold) + quality_delta)
        else:
            effective_threshold = adaptive_threshold
        filtered_label, filtered_confidence, filter_trace = self._apply_prediction_filters(
            prediction=normalized_label,
            confidence=normalized_confidence,
            threshold=effective_threshold,
        )

        # --- DIAGNOSTIC LOG: expose threshold decisions and final outcome ---
        logger.info(
            "DIAG_build_prediction_response",
            prediction_before_filters=normalized_label,
            confidence_before_filters=round(normalized_confidence, 4),
            adaptive_threshold=round(float(adaptive_threshold), 4),
            effective_threshold=round(float(effective_threshold), 4),
            base_confidence_threshold=round(float(self.confidence_threshold), 4),
            filtered_label=filtered_label,
            filtered_confidence=round(filtered_confidence, 4),
            is_accepted=filtered_label != "NONE" and filtered_confidence >= effective_threshold,
            frame_buffer_len=len(self.frame_buffer),
            recording_frames=self._recording_frame_count,
            num_labels=len(self.labels),
        )
        # --- END DIAGNOSTIC ---

        decision_trace = {
            "prediction_before_filters": self._last_decision_trace,
            "adaptive_threshold": round(float(adaptive_threshold), 4),
            "class_threshold": round(float(label_threshold), 4),
            "quality_delta": round(float(quality_delta), 4),
            "effective_threshold": round(float(effective_threshold), 4),
            "filter_trace": filter_trace,
            "counters": dict(self._decision_counters),
        }

        self.frame_buffer.clear()
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self.state = InferenceState.IDLE

        is_accepted = filtered_label != "NONE" and filtered_confidence >= effective_threshold

        if is_accepted:
            if not self.sentence_tokens or self.sentence_tokens[-1] != filtered_label:
                self.sentence_tokens.append(filtered_label)
            # Update conversation context sign time
            if self.conversation_context is not None:
                self.conversation_context.touch_sign_time()

        sentence = " ".join(self.sentence_tokens)

        # Grammar translation (Phase 3) — applied when enabled
        translated_sentence: str | None = None
        grammar_tags: list[str] | None = None
        translation_mode: str | None = None
        if self.grammar_translator is not None and self.sentence_tokens:
            translated_sentence, grammar_tags, translation_mode = (
                self._translate_sentence_buffer(
                    token_confidences=[filtered_confidence] * len(self.sentence_tokens)
                )
            )

        # Conversation context (Phase 5) — enrich prediction with turn info
        turn_id: int | None = None
        is_new_turn: bool = False
        context_summary: dict | None = None
        if (
            self.conversation_context is not None
            and translated_sentence is not None
            and self.sentence_tokens
        ):
            # Flush turn when sentence has enough tokens (≥ 3) — heuristic for sentence-complete
            # The API layer will also flush on is_sentence_complete events
            if len(self.sentence_tokens) >= 3:
                turn_id, is_new_turn, context_summary = self._flush_complete_turn(
                    translated_text=translated_sentence,
                    grammar_tags=grammar_tags,
                    confidence=filtered_confidence,
                )

        return Prediction(
            prediction=filtered_label,
            confidence=filtered_confidence,
            alternatives=normalized_alternatives,
            sentence_buffer=sentence,
            is_sentence_complete=False,
            decision_diagnostics=decision_trace,
            translated_sentence=translated_sentence,
            grammar_tags=grammar_tags,
            translation_mode=translation_mode,
            turn_id=turn_id,
            is_new_turn=is_new_turn,
            conversation_context=context_summary,
        )

    def reset(self) -> None:
        """Reset temporal state for a new translation session."""
        self.frame_buffer.clear()
        self.hand_visibility_history.clear()
        self.motion_history.clear()
        self.prediction_history.clear()
        self.sentence_tokens.clear()
        self._current_motion_energy = 0.0
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self._latest_frontend_confidence = 1.0
        for key in list(self._decision_counters.keys()):
            self._decision_counters[key] = 0
        self._last_decision_trace = {
            "status": "idle",
            "reason": "reset",
        }
        # Reset conversation context if present (keeps the context object, just clears state)
        if self.conversation_context is not None:
            self.conversation_context.clear()

    def snapshot_decision_diagnostics(self) -> dict[str, object]:
        """Expose serializable decision diagnostics for audit and tests."""
        return {
            "counters": dict(self._decision_counters),
            "last_decision_trace": dict(self._last_decision_trace),
            "state": self.state.value,
            "buffer_frames": int(len(self.frame_buffer)),
        }

    def _infer_window(self, window: np.ndarray) -> tuple[str, float, list[dict[str, float]]]:
        """
        Run PyTorch model inference over one sequence window.

        Args:
            window: Numpy array of shape [seq_len, num_features]

        Returns:
            Tuple of (prediction_label, confidence, alternatives)
        """
        # If no model loaded, return NONE
        if self.model is None or len(self.labels) == 0:
            return "NONE", 0.0, []

        try:
            probs, view_disagreement = self._infer_probabilities(window)
            self._decision_counters["total_inferences"] += 1

            # --- DIAGNOSTIC LOG: expose class distribution to identify "always bonjour" bug ---
            num_model_classes = len(probs)
            logger.info(
                "DIAG_infer_window_probs",
                num_model_classes=num_model_classes,
                num_labels=len(self.labels),
                labels=list(self.labels),
                all_probs={
                    self.labels[i] if i < len(self.labels) else f"class_{i}": round(float(probs[i]), 4)
                    for i in range(len(probs))
                },
                top1_label=self.labels[int(np.argmax(probs))] if len(probs) > 0 and len(self.labels) > 0 else "?",
                top1_prob=round(float(np.max(probs)), 4) if len(probs) > 0 else 0.0,
                prob_entropy=round(float(-np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)))), 4) if len(probs) > 0 else 0.0,
            )
            # --- END DIAGNOSTIC ---

            # Get top predictions
            top_k = min(4, len(probs))  # Get top 4 predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]

            if len(self.labels) != len(probs):
                logger.warning(
                    "runtime_label_mismatch",
                    model_classes=len(probs),
                    labels=len(self.labels),
                )
                runtime_labels = self.labels[: len(probs)]
                while len(runtime_labels) < len(probs):
                    runtime_labels.append(f"class_{len(runtime_labels)}")
            else:
                runtime_labels = self.labels

            # Main prediction
            pred_idx = int(top_indices[0])
            raw_confidence = float(probs[pred_idx])

            # Map index to label
            if pred_idx < len(runtime_labels):
                prediction = runtime_labels[pred_idx]
            else:
                prediction = "NONE"
                raw_confidence = 0.0

            # Alternatives (top 3 excluding main prediction)
            alternatives = []
            for idx in top_indices[1:top_k]:
                if idx < len(runtime_labels):
                    alt_label = runtime_labels[idx]
                    if alt_label in {"NONE", "[NONE]"}:
                        continue
                    alt_conf = float(probs[idx])
                    alternatives.append({
                        "sign": alt_label,
                        "confidence": round(alt_conf, 3)
                    })

            calibrated_confidence = self._calibrate_confidence(
                probs=probs,
                top_indices=top_indices,
                raw_confidence=raw_confidence,
                disagreement=view_disagreement,
            )

            # If confidence margin is too low, treat prediction as NONE.
            if calibrated_confidence <= 0.0:
                self._decision_counters["rejected_by_calibration"] += 1
                self._decision_counters["rejected_total"] += 1
                self._last_decision_trace = {
                    "status": "rejected",
                    "stage": "calibration",
                    "reason": "calibrated_confidence_non_positive",
                    "raw_confidence": round(raw_confidence, 4),
                    "calibrated_confidence": round(calibrated_confidence, 4),
                    "view_disagreement": round(float(view_disagreement), 6),
                }
                return "NONE", 0.0, alternatives

            if prediction in {"NONE", "[NONE]"}:
                self._decision_counters["rejected_by_label_none"] += 1
                self._decision_counters["rejected_total"] += 1
                self._last_decision_trace = {
                    "status": "rejected",
                    "stage": "labeling",
                    "reason": "model_predicted_none_class",
                    "calibrated_confidence": round(calibrated_confidence, 4),
                }
                return "NONE", round(calibrated_confidence, 3), alternatives

            self._last_decision_trace = {
                "status": "candidate",
                "stage": "post_inference",
                "prediction": prediction,
                "raw_confidence": round(raw_confidence, 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
                "view_disagreement": round(float(view_disagreement), 6),
                "top2_margin": round(
                    float(raw_confidence - (float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0)),
                    4,
                ),
            }

            return prediction, round(calibrated_confidence, 3), alternatives

        except Exception as e:
            logger.error("inference_failed", error=str(e), exc_info=True)
            return "NONE", 0.0, []

    def _infer_probabilities(self, window: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on one or multiple temporal views and return mean probabilities."""
        if self.model is None and self.onnx_session is None:
            return np.array([], dtype=np.float32), 0.0

        views = self._build_inference_views(window)

        # ONNX inference path
        if self.use_onnx and self.onnx_session is not None:
            return self._infer_probabilities_onnx(views)

        # PyTorch inference path
        return self._infer_probabilities_pytorch(views)

    def _resolve_runtime_feature_dim(self) -> int | None:
        """Resolve expected model input width for compatibility across checkpoints."""
        feature_dim = resolve_model_feature_dim(self.model)
        if feature_dim is not None:
            return feature_dim

        if self.use_onnx and self.onnx_session is not None:
            try:
                input_shape = self.onnx_session.get_inputs()[0].shape
                if len(input_shape) >= 3:
                    candidate = input_shape[2]
                    if isinstance(candidate, (int, np.integer)) and int(candidate) > 0:
                        return int(candidate)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _infer_probabilities_pytorch(self, views: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Run PyTorch inference on multiple views."""
        stacked_probs: list[torch.Tensor] = []
        target_dim = self._resolve_runtime_feature_dim()

        with torch.no_grad():
            for view in views:
                aligned_view = align_numpy_features(view, target_dim)
                tensor = torch.from_numpy(aligned_view).float().unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                softmax_temp = self.calibration_temperature or self.inference_temperature
                probs = torch.softmax(logits / softmax_temp, dim=1)
                stacked_probs.append(probs.squeeze(0))

        if not stacked_probs:
            return np.array([], dtype=np.float32), 0.0

        probs_stack = torch.stack(stacked_probs, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-8)
        disagreement = float(probs_stack.std(dim=0, unbiased=False).mean().item())
        return mean_probs.cpu().numpy(), disagreement

    def _infer_probabilities_onnx(self, views: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Run ONNX Runtime inference on multiple views."""
        stacked_probs: list[np.ndarray] = []
        target_dim = self._resolve_runtime_feature_dim()

        for view in views:
            aligned_view = align_numpy_features(view, target_dim)
            # Prepare input: [1, seq_len, features]
            onnx_input = aligned_view.astype(np.float32).reshape(
                1,
                aligned_view.shape[0],
                aligned_view.shape[1],
            )

            # Run ONNX inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            logits = self.onnx_session.run([output_name], {input_name: onnx_input})[0]

            # Apply softmax with temperature
            softmax_temp = self.calibration_temperature or self.inference_temperature
            logits_scaled = logits / softmax_temp

            # Compute softmax manually
            exp_logits = np.exp(logits_scaled - logits_scaled.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            stacked_probs.append(probs.squeeze(0))

        if not stacked_probs:
            return np.array([], dtype=np.float32), 0.0

        # Stack and average probabilities
        probs_array = np.stack(stacked_probs, axis=0)
        mean_probs = probs_array.mean(axis=0)
        mean_probs = mean_probs / max(mean_probs.sum(), 1e-8)
        disagreement = float(probs_array.std(axis=0, ddof=0).mean())

        return mean_probs, disagreement

    def _build_inference_views(self, window: np.ndarray) -> list[np.ndarray]:
        """Create lightweight temporal views for test-time ensembling."""
        if window.ndim != 2:
            return [window.astype(np.float32, copy=False)]
        return self._tta_generator.generate(window)

    def _smooth(self) -> tuple[str, float]:
        """Apply majority-vote smoothing with confidence stabilization."""
        if not self.prediction_history:
            return "NONE", 0.0

        score_by_label: dict[str, list[float]] = {}
        for label, score in self.prediction_history:
            score_by_label.setdefault(label, []).append(score)

        best_label = max(
            score_by_label,
            key=lambda key: (len(score_by_label[key]), np.mean(score_by_label[key])),
        )
        best_scores = score_by_label[best_label]
        vote_ratio = len(best_scores) / len(self.prediction_history)
        mean_confidence = float(np.mean(best_scores))

        # Keep confidence interpretable while rewarding temporal agreement.
        stabilized_confidence = mean_confidence * (0.6 + 0.4 * vote_ratio)
        return best_label, round(stabilized_confidence, 3)

    def _mean_hand_visibility(self) -> float:
        """Return rolling mean hand visibility over the current frame window."""
        if not self.hand_visibility_history:
            return 0.0
        return float(np.mean(self.hand_visibility_history))

    def _mean_motion_energy(self) -> float:
        """Return rolling hand motion magnitude over the active frame window."""
        if not self.motion_history:
            return 0.0
        return float(np.mean(self.motion_history))

    def _compute_hand_visibility(self, frame: FrameLandmarks) -> float:
        """Estimate hand landmark visibility score in [0, 1] for one frame."""
        expected_points_per_hand = 21

        def visible_points(hand: list[list[float]]) -> int:
            if not hand:
                return 0
            count = 0
            for point in hand[:expected_points_per_hand]:
                if len(point) >= 3 and any(abs(float(coord)) > 1e-6 for coord in point[:3]):
                    count += 1
            return count

        total_visible = visible_points(frame.left_hand) + visible_points(frame.right_hand)
        total_expected = expected_points_per_hand * 2
        return min(1.0, total_visible / total_expected)

    def _compute_motion_energy(self) -> float:
        """Estimate hand motion energy between recent frames."""
        if len(self.frame_buffer) < 2:
            return 0.0

        previous = self.frame_buffer[-2]
        current = self.frame_buffer[-1]
        hand_previous = previous[:126]  # left + right hands
        hand_current = current[:126]
        motion = np.mean(np.abs(hand_current - hand_previous))
        return float(motion)

    def _adaptive_threshold(self) -> float:
        """
        Raise threshold when tracking quality drops.

        This suppresses false positives for low-visibility / low-motion windows.
        """
        threshold = self.confidence_threshold
        if self._mean_hand_visibility() < max(0.3, self.min_hand_visibility + 0.05):
            threshold += 0.03
        if self._mean_motion_energy() < self.min_motion_energy:
            threshold += 0.04
        if self._latest_frontend_confidence < self.frontend_confidence_floor:
            threshold += 0.05
        elif self._latest_frontend_confidence < 0.55:
            threshold += 0.02
        return float(min(0.92, threshold))

    def _threshold_for_label(self, label: str) -> float:
        """Return class-specific confidence threshold when available."""
        if not label:
            return 0.0
        if label in self.class_thresholds:
            return float(self.class_thresholds[label])
        if label == "NONE" and "[NONE]" in self.class_thresholds:
            return float(self.class_thresholds["[NONE]"])
        if label == "[NONE]" and "NONE" in self.class_thresholds:
            return float(self.class_thresholds["NONE"])
        return 0.0

    def _apply_prediction_filters(
        self,
        *,
        prediction: str,
        confidence: float,
        threshold: float,
    ) -> tuple[str, float, dict[str, object]]:
        """Filter raw predictions with temporal consensus + adaptive threshold."""
        if prediction == "NONE" or confidence <= 0.0:
            self._decision_counters["rejected_by_calibration"] += 1
            self._decision_counters["rejected_total"] += 1
            trace = {
                "status": "rejected",
                "stage": "filters",
                "reason": "prediction_none_or_non_positive_confidence",
                "confidence": round(float(confidence), 4),
                "threshold": round(float(threshold), 4),
            }
            self._last_decision_trace = trace
            return "NONE", 0.0, trace

        if self.prediction_history and self.prediction_history[-1][0] != prediction:
            self.prediction_history.clear()

        self.prediction_history.append((prediction, confidence))
        smoothed_label, smoothed_confidence = self._smooth()

        # Keep fast response when a stable label repeats.
        if smoothed_label == prediction:
            confidence = max(confidence, smoothed_confidence)

        if confidence < threshold:
            self._decision_counters["rejected_by_confidence_threshold"] += 1
            self._decision_counters["rejected_total"] += 1
            if threshold > self.confidence_threshold:
                self._decision_counters["rejected_by_adaptive_threshold"] += 1
            class_threshold = self._threshold_for_label(prediction)
            if class_threshold > 0 and threshold >= class_threshold:
                self._decision_counters["rejected_by_class_threshold"] += 1
            trace = {
                "status": "rejected",
                "stage": "filters",
                "reason": "below_effective_threshold",
                "prediction": prediction,
                "confidence": round(float(confidence), 4),
                "smoothed_confidence": round(float(smoothed_confidence), 4),
                "threshold": round(float(threshold), 4),
            }
            self._last_decision_trace = trace
            return "NONE", 0.0, trace

        self._decision_counters["accepted"] += 1
        trace = {
            "status": "accepted",
            "stage": "filters",
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "threshold": round(float(threshold), 4),
            "smoothed_label": smoothed_label,
            "smoothed_confidence": round(float(smoothed_confidence), 4),
        }
        self._last_decision_trace = trace

        return prediction, round(confidence, 3), trace

    def _trim_recording_window(self, window: np.ndarray, trailing_rest: int) -> np.ndarray:
        """Trim pre/post idle frames before resampling."""
        if window.ndim != 2 or window.shape[0] == 0:
            return window

        trimmed = window
        if trailing_rest > 0:
            min_keep = max(2, self.min_recording_frames // 2)
            if trimmed.shape[0] > (trailing_rest + min_keep):
                trimmed = trimmed[:-trailing_rest]

        hand_energy = np.mean(np.abs(trimmed[:, :126]), axis=1)
        active_indices = np.flatnonzero(hand_energy > 1e-4)
        if active_indices.size == 0:
            return trimmed

        start = int(max(0, active_indices[0] - 1))
        end = int(min(trimmed.shape[0], active_indices[-1] + 2))
        if end <= start:
            return trimmed
        return trimmed[start:end]

    @staticmethod
    def _resolve_frontend_confidence(metadata: dict) -> float:
        """Parse frontend confidence metadata into a stable [0, 1] range."""
        if not isinstance(metadata, dict):
            return 1.0
        value = metadata.get("averageConfidence")
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 1.0
        if np.isnan(confidence):
            return 1.0
        return float(np.clip(confidence, 0.0, 1.0))

    def _blend_hand_visibility(self, hand_visibility: float, frontend_confidence: float) -> float:
        """Blend backend-computed hand visibility with frontend tracking quality."""
        if frontend_confidence >= self.frontend_confidence_floor:
            return hand_visibility
        penalty = (self.frontend_confidence_floor - frontend_confidence) * 0.8
        return float(max(0.0, hand_visibility - penalty))

    def _calibrate_confidence(
        self,
        *,
        probs: np.ndarray,
        top_indices: np.ndarray,
        raw_confidence: float,
        disagreement: float = 0.0,
    ) -> float:
        """Calibrate confidence with margin and distribution certainty."""
        if raw_confidence <= 0.0:
            return 0.0

        top2_confidence = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
        margin = max(0.0, raw_confidence - top2_confidence)

        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(max(len(probs), 2)))
        certainty = 1.0 - min(1.0, entropy / max_entropy)

        # Motion-aware calibration for open-set rejection in static/noisy windows.
        motion_factor = np.clip(
            self._current_motion_energy / max(self.min_motion_energy, 1e-6), 0.0, 1.0
        )
        calibrated = (
            (0.50 * raw_confidence)
            + (0.30 * margin)
            + (0.12 * certainty)
            + (0.08 * motion_factor)
        )
        if margin < self.min_prediction_margin:
            self._decision_counters["rejected_by_margin"] += 1
            calibrated *= margin / self.min_prediction_margin
        if self._current_motion_energy < self.min_motion_energy:
            self._decision_counters["rejected_by_motion"] += 1
            calibrated *= 0.85

        if self.inference_num_views > 1:
            disagreement_ratio = np.clip(disagreement / self.max_view_disagreement, 0.0, 1.0)
            calibrated *= 1.0 - (0.35 * disagreement_ratio)
            if disagreement > self.max_view_disagreement:
                self._decision_counters["rejected_by_tta_disagreement"] += 1
                calibrated *= 0.8

        return float(np.clip(calibrated, 0.0, 1.0))
