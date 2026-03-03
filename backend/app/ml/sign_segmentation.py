"""Sign boundary detection using BiLSTM sequence labeling.

Replaces the threshold-based motion_energy segmentation in pipeline.py with a
learned BiLSTM sequence tagger using BIEO labels.

Label scheme (per frame):
    B (0) — Beginning of a sign
    I (1) — Inside/continuation of a sign
    E (2) — End of a sign
    O (3) — Outside / transition (neutral / inter-sign)

The model uses V2 features as input. Key sub-vectors exploited:
    - Coordinate velocities  [237:462] — motion dynamics
    - NMM features           [561:593] — detect grammatical pauses
    - Handshape              [477:561] — discriminate signs vs. neutral

Integration with pipeline.py:
    - ``detect_boundaries(feature_sequence)`` returns (start, end) frame pairs.
    - Falls back to ``motion_energy_fallback()`` when no trained model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from app.ml.features import ENRICHED_FEATURE_DIM_V2

logger = structlog.get_logger(__name__)

# Optional torchcrf for CRF head
try:
    from torchcrf import CRF
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False
    logger.debug(
        "torchcrf_not_available",
        msg="torchcrf not installed — CRF head disabled; install with: pip install pytorch-crf",
    )

# ── Label constants ──────────────────────────────────────────────────────────
LABEL_B = 0   # Beginning
LABEL_I = 1   # Inside
LABEL_E = 2   # End
LABEL_O = 3   # Outside / neutral
NUM_BIEO_LABELS = 4
BIEO_NAMES = ["B", "I", "E", "O"]


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class SegmentationConfig:
    """Configuration for SignBoundaryDetector."""

    feature_dim: int = ENRICHED_FEATURE_DIM_V2   # 611
    input_selection: str = "velocity+nmm+handshape"
    # When input_selection=="all", full feature_dim is used as input.
    # When "velocity+nmm+handshape", a compact 225+32+84=341 subset is used.
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_crf: bool = False                # requires torchcrf
    min_sign_frames: int = 8             # minimum frames to form a valid sign segment
    merge_gap_frames: int = 3            # merge sign segments closer than N frames
    boundary_threshold: float = 0.5     # prob threshold for B/I/E vs O
    architecture: str = "SignBoundaryDetector"

    def to_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "input_selection": self.input_selection,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_crf": self.use_crf,
            "min_sign_frames": self.min_sign_frames,
            "merge_gap_frames": self.merge_gap_frames,
            "boundary_threshold": self.boundary_threshold,
            "architecture": self.architecture,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Input feature selector ────────────────────────────────────────────────────

def _build_segmentation_input(x: Tensor, selection: str) -> Tensor:
    """Select and concatenate relevant sub-vectors from V2 feature tensor.

    Args:
        x: (batch, seq, 611) V2 features
        selection: One of "all", "velocity+nmm+handshape", "velocity", "custom"

    Returns:
        Reduced tensor (batch, seq, selected_dim)
    """
    if selection == "all":
        return x

    parts = []
    if "velocity" in selection:
        # Coordinate velocities [237:462] = 225 dims (dominant motion signal)
        parts.append(x[..., 237:462])
        # Also include pose coords [126:225] = 99 dims (location information)
        parts.append(x[..., 126:225])

    if "nmm" in selection:
        # NMM features [561:593] = 32 dims (grammatical pauses, mouth shape)
        parts.append(x[..., 561:593])
        # Facial expression [225:237] = 12 dims complement NMM
        parts.append(x[..., 225:237])

    if "handshape" in selection:
        # Handshape [477:561] = 84 dims (discriminate sign vs. neutral)
        parts.append(x[..., 477:561])

    if "inter_dist" in selection or selection == "velocity+nmm+handshape":
        # Inter-hand distances [462:467] = 5 dims
        parts.append(x[..., 462:467])

    if not parts:
        return x

    return torch.cat(parts, dim=-1)


def _get_input_dim(selection: str, feature_dim: int = ENRICHED_FEATURE_DIM_V2) -> int:
    """Return the input dimension produced by _build_segmentation_input."""
    if selection == "all":
        return feature_dim
    dim = 0
    if "velocity" in selection:
        dim += 225 + 99  # coord_vel + pose
    if "nmm" in selection:
        dim += 32 + 12   # NMM + facial_expr
    if "handshape" in selection:
        dim += 84
    if "inter_dist" in selection or selection == "velocity+nmm+handshape":
        dim += 5
    return dim if dim > 0 else feature_dim


# ── Main model ───────────────────────────────────────────────────────────────


class SignBoundaryDetector(nn.Module):
    """BiLSTM for per-frame BIEO boundary labeling.

    Accepts V2 feature sequences (batch, seq_len, feature_dim) and produces
    per-frame logits (batch, seq_len, 4) for the BIEO label set.

    The model can optionally use a CRF layer to enforce valid label sequences
    (e.g., no B→B without I/E between them).

    Example usage::

        detector = SignBoundaryDetector()
        logits = detector(x)               # (batch, seq, 4)
        labels = detector.decode(x)        # list[(batch,)] int labels
        segments = detector.detect_boundaries(feature_seq)  # [(start, end), ...]
    """

    def __init__(
        self,
        feature_dim: int = ENRICHED_FEATURE_DIM_V2,   # 611
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_crf: bool = False,
        input_selection: str = "velocity+nmm+handshape",
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_crf = use_crf and _CRF_AVAILABLE
        self.input_selection = input_selection

        resolved_input_dim = _get_input_dim(input_selection, feature_dim)
        self._resolved_input_dim = resolved_input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(resolved_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # BiLSTM encoder
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: 2 * hidden_dim (bidirectional) → num labels
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_BIEO_LABELS),
        )

        # Optional CRF layer
        if self.use_crf:
            self.crf = CRF(NUM_BIEO_LABELS, batch_first=True)
        else:
            self.crf = None

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """Compute per-frame BIEO logits.

        Args:
            x: (batch, seq_len, feature_dim) V2 feature tensor
            lengths: (batch,) actual sequence lengths for masking (optional)

        Returns:
            logits: (batch, seq_len, 4) unnormalized label scores
        """
        # Select relevant sub-features
        selected = _build_segmentation_input(x, self.input_selection)

        # Project to hidden dim
        h = self.input_proj(selected)   # (batch, seq, hidden_dim)

        # Pack for variable length if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(h)   # (batch, seq, 2*hidden_dim)

        logits = self.output_proj(lstm_out)  # (batch, seq, 4)
        return logits

    def decode(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> list[list[int]]:
        """Decode BIEO labels from logits.

        Uses CRF Viterbi if available, otherwise argmax.

        Args:
            x: (batch, seq_len, feature_dim)
            lengths: (batch,) actual lengths (optional)

        Returns:
            List of label sequences, one per batch item.
        """
        logits = self.forward(x, lengths=lengths)  # (B, T, 4)

        if self.use_crf and self.crf is not None:
            # Build mask from lengths or all-ones
            if lengths is not None:
                mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            else:
                mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
            return self.crf.decode(logits, mask=mask)
        else:
            preds = logits.argmax(dim=-1)  # (B, T)
            return preds.tolist()

    def crf_loss(
        self,
        x: Tensor,
        tags: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute CRF negative log-likelihood loss (only when use_crf=True).

        Args:
            x: (batch, seq, feature_dim)
            tags: (batch, seq) ground-truth BIEO labels (int)
            lengths: (batch,) actual lengths (optional)

        Returns:
            Scalar loss tensor.
        """
        if not self.use_crf or self.crf is None:
            # Fall back to cross-entropy
            logits = self.forward(x, lengths=lengths)
            return F.cross_entropy(
                logits.view(-1, NUM_BIEO_LABELS),
                tags.view(-1).long(),
                ignore_index=-100,
            )
        logits = self.forward(x, lengths=lengths)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        else:
            mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        # CRF expects (batch, seq, num_tags)
        return -self.crf(logits, tags.long(), mask=mask, reduction="mean")

    # ── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def detect_boundaries(
        self,
        feature_sequence: np.ndarray | Tensor,
        *,
        min_sign_frames: int = 8,
        merge_gap_frames: int = 3,
        boundary_threshold: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> list[tuple[int, int]]:
        """Detect sign segment boundaries in a feature sequence.

        Operates on a single sequence (no batch dimension needed).

        Args:
            feature_sequence: (seq_len, feature_dim) array or tensor
            min_sign_frames: Minimum frames for a valid sign segment
            merge_gap_frames: Merge adjacent segments if gap ≤ this value
            boundary_threshold: Probability threshold for sign frames (B/I/E)
            device: Inference device

        Returns:
            List of (start_frame, end_frame) tuples (inclusive, 0-indexed)
        """
        self.eval()

        if isinstance(feature_sequence, np.ndarray):
            tensor = torch.from_numpy(feature_sequence).float()
        else:
            tensor = feature_sequence.float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # (1, seq, feat)

        tensor = tensor.to(device)
        logits = self.forward(tensor)   # (1, seq, 4)
        probs = F.softmax(logits, dim=-1)  # (1, seq, 4)

        # Frame is "in a sign" if prob(O) < 1 - boundary_threshold
        sign_probs = 1.0 - probs[0, :, LABEL_O]  # (seq,)
        is_sign = (sign_probs >= boundary_threshold).cpu().numpy()

        return self._extract_segments(
            is_sign,
            min_sign_frames=min_sign_frames,
            merge_gap_frames=merge_gap_frames,
        )

    @staticmethod
    def _extract_segments(
        is_sign: np.ndarray,
        min_sign_frames: int = 8,
        merge_gap_frames: int = 3,
    ) -> list[tuple[int, int]]:
        """Convert binary frame mask to (start, end) segment list.

        Args:
            is_sign: Boolean array (seq_len,)
            min_sign_frames: Drop segments shorter than this
            merge_gap_frames: Merge segments with gap ≤ this

        Returns:
            Sorted list of (start, end) tuples (inclusive)
        """
        if not np.any(is_sign):
            return []

        # Collect raw segments
        segments: list[tuple[int, int]] = []
        in_sign = False
        start = 0
        for i, flag in enumerate(is_sign):
            if flag and not in_sign:
                start = i
                in_sign = True
            elif not flag and in_sign:
                segments.append((start, i - 1))
                in_sign = False
        if in_sign:
            segments.append((start, len(is_sign) - 1))

        if not segments:
            return []

        # Merge nearby segments
        merged: list[tuple[int, int]] = [segments[0]]
        for seg_start, seg_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if seg_start - prev_end <= merge_gap_frames:
                merged[-1] = (prev_start, seg_end)
            else:
                merged.append((seg_start, seg_end))

        # Filter by minimum length
        return [(s, e) for s, e in merged if (e - s + 1) >= min_sign_frames]

    # ── Streaming mode ──────────────────────────────────────────────────────

    def streaming_detect(
        self,
        feature_window: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> list[tuple[int, int]]:
        """Detect boundaries in a sliding window for streaming inference.

        Thin wrapper around detect_boundaries for use in pipeline.py.

        Args:
            feature_window: (window_len, feature_dim) recent frames
            device: Inference device

        Returns:
            (start, end) pairs within the window
        """
        return self.detect_boundaries(feature_window, device=device)

    # ── Checkpoint I/O ──────────────────────────────────────────────────────

    def to_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.0,  # not stored explicitly — store as 0 for inference
            use_crf=self.use_crf,
            input_selection=self.input_selection,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.to_config().to_dict(),
                "architecture": "SignBoundaryDetector",
            },
            str(path),
        )
        logger.info("segmentation_checkpoint_saved", path=str(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "SignBoundaryDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        config = SegmentationConfig.from_dict(ckpt.get("config", {}))
        model = cls(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_crf=config.use_crf,
            input_selection=config.input_selection,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        logger.info("segmentation_checkpoint_loaded", path=str(path))
        return model

    def set_inference_mode(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False


# ── Motion energy fallback ───────────────────────────────────────────────────


def motion_energy_fallback(
    feature_sequence: np.ndarray,
    *,
    motion_start_threshold: float = 0.005,
    rest_frames_threshold: int = 10,
    min_recording_frames: int = 15,
) -> list[tuple[int, int]]:
    """Fallback segmentation using heuristic motion energy on velocity features.

    Used when no trained SignBoundaryDetector is available.

    Args:
        feature_sequence: (seq_len, 611) V2 feature array
        motion_start_threshold: Motion energy to start a sign
        rest_frames_threshold: Consecutive rest frames to end a sign
        min_recording_frames: Minimum frames for a valid sign

    Returns:
        List of (start, end) tuples
    """
    seq_len = feature_sequence.shape[0]

    # Extract velocity block [237:462] for motion energy
    if feature_sequence.shape[1] > 462:
        vel_block = feature_sequence[:, 237:462]
    else:
        vel_block = feature_sequence

    # Per-frame L2 motion energy
    motion_energy = np.linalg.norm(vel_block, axis=1) / max(vel_block.shape[1], 1)

    segments: list[tuple[int, int]] = []
    in_sign = False
    sign_start = 0
    frame_count = 0
    rest_count = 0

    for i in range(seq_len):
        me = float(motion_energy[i])
        if not in_sign:
            if me > motion_start_threshold:
                in_sign = True
                sign_start = i
                frame_count = 1
                rest_count = 0
        else:
            frame_count += 1
            if me < motion_start_threshold:
                rest_count += 1
            else:
                rest_count = 0
            if rest_count >= rest_frames_threshold and frame_count >= min_recording_frames:
                segments.append((sign_start, i - rest_count))
                in_sign = False

    if in_sign and frame_count >= min_recording_frames:
        segments.append((sign_start, seq_len - 1))

    return segments


# ── Synthetic BIEO corpus generator ─────────────────────────────────────────


def generate_bieo_corpus(
    n_sequences: int = 150,
    feature_dim: int = 457,
    min_seq_len: int = 20,
    max_seq_len: int = 80,
    min_sign_frames: int = 8,
    max_sign_frames: int = 25,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Génère des séquences de features (T, feature_dim) avec labels BIEO frame-par-frame.

    Comportement :
    - Chaque séquence est constituée de blocs "signe" (frames B I...I E) et de gaps "repos" (O)
    - Un signe occupe min_sign_frames..max_sign_frames frames consécutives
    - Un gap entre signes : 3..15 frames O
    - Features : np.float32, gaussian noise pour O, mean += 0.5 sur dims 0..15 pour B/I/E
    - Labels : list[int] alignés frame-à-frame (B=0, I=1, E=2, O=3)
    - Longueur variable ; pad avec O si nécessaire pour atteindre min_seq_len

    Args:
        n_sequences: Nombre de séquences à générer.
        feature_dim: Dimension des features (après input_selection).
        min_seq_len: Longueur minimale d'une séquence (padding O si besoin).
        max_seq_len: Longueur maximale d'une séquence.
        min_sign_frames: Nombre minimal de frames par signe.
        max_sign_frames: Nombre maximal de frames par signe.
        seed: Graine aléatoire.

    Returns:
        Tuple (sequences, labels) — listes de longueur n_sequences.
        - sequences : chaque élément est un np.ndarray float32 de shape (T_i, feature_dim).
        - labels    : chaque élément est une list[int] de longueur T_i.
    """
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    all_labels: list[list[int]] = []

    for _ in range(n_sequences):
        target_len = int(rng.integers(min_seq_len, max_seq_len + 1))
        frames: list[np.ndarray] = []
        labels: list[int] = []

        # Commence toujours par un petit gap de repos
        gap = int(rng.integers(2, 6))
        for _ in range(gap):
            frames.append(rng.standard_normal(feature_dim).astype(np.float32))
            labels.append(LABEL_O)

        while len(frames) < target_len:
            # Signe
            sign_len = int(rng.integers(min_sign_frames, max_sign_frames + 1))
            if len(frames) + sign_len > target_len + max_sign_frames:
                break  # pas assez de place pour un signe entier, on arrête

            for frame_idx in range(sign_len):
                feat = rng.standard_normal(feature_dim).astype(np.float32)
                # Signal distinctif sur les 16 premières dimensions
                feat[:16] += 0.5
                frames.append(feat)
                if sign_len == 1:
                    # Signe d'une seule frame : B = E → on met B
                    labels.append(LABEL_B)
                elif frame_idx == 0:
                    labels.append(LABEL_B)
                elif frame_idx == sign_len - 1:
                    labels.append(LABEL_E)
                else:
                    labels.append(LABEL_I)

            # Gap de repos entre signes
            gap = int(rng.integers(3, 16))
            for _ in range(gap):
                if len(frames) >= target_len + max_sign_frames:
                    break
                frames.append(rng.standard_normal(feature_dim).astype(np.float32))
                labels.append(LABEL_O)

        # Tronquer ou compléter jusqu'à target_len
        if len(frames) > target_len:
            frames = frames[:target_len]
            labels = labels[:target_len]

        # Padding avec O pour atteindre min_seq_len
        while len(frames) < min_seq_len:
            frames.append(rng.standard_normal(feature_dim).astype(np.float32))
            labels.append(LABEL_O)

        sequences.append(np.stack(frames, axis=0))
        all_labels.append(labels)

    return sequences, all_labels


# ── Segmentation trainer ─────────────────────────────────────────────────────


class SegmentationTrainer:
    """Gère l'entraînement supervisé du SignBoundaryDetector avec labels BIEO."""

    # Poids par défaut : sur-pondérer B et E (rares) vs O
    _DEFAULT_CLASS_WEIGHTS = [4.0, 2.0, 4.0, 1.0]

    def __init__(
        self,
        model: "SignBoundaryDetector",
        config: SegmentationConfig,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[list[float]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        weights = class_weights if class_weights is not None else self._DEFAULT_CLASS_WEIGHTS
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        self._ce_weight = weight_tensor

    # ── Collate helper ──────────────────────────────────────────────────────

    def _collate(
        self,
        sequences: list[np.ndarray],
        labels: list[list[int]],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Collate variable-length sequences into a padded batch.

        Returns:
            padded_x : (B, T_max, feature_dim) float32 tensor on self.device
            padded_y : (B, T_max) long tensor on self.device (padding = -100)
            lengths  : (B,) long tensor on self.device
        """
        tensors_x = [
            torch.from_numpy(seq).float() for seq in sequences
        ]
        tensors_y = [
            torch.tensor(lbl, dtype=torch.long) for lbl in labels
        ]
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

        # pad_sequence expects list of (T_i, feat)
        padded_x = nn.utils.rnn.pad_sequence(tensors_x, batch_first=True, padding_value=0.0)
        padded_y = nn.utils.rnn.pad_sequence(tensors_y, batch_first=True, padding_value=-100)

        return (
            padded_x.to(self.device),
            padded_y.to(self.device),
            lengths.to(self.device),
        )

    # ── Epoch ───────────────────────────────────────────────────────────────

    def train_epoch(
        self,
        sequences: list[np.ndarray],
        labels: list[list[int]],
        batch_size: int = 16,
    ) -> dict[str, float]:
        """Une époque d'entraînement.

        Args:
            sequences: Liste de tableaux (T_i, feature_dim) float32.
            labels: Liste de listes d'entiers BIEO.
            batch_size: Taille des mini-batchs.

        Returns:
            Dictionnaire {"loss": float, "lr": float}.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        indices = list(range(len(sequences)))
        # Mélangeons avec numpy pour reproductibilité
        import random as _random
        _random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            batch_seqs = [sequences[i] for i in batch_idx]
            batch_lbls = [labels[i] for i in batch_idx]

            padded_x, padded_y, lengths = self._collate(batch_seqs, batch_lbls)

            self.optimizer.zero_grad()

            if self.model.use_crf and self.model.crf is not None:
                loss = self.model.crf_loss(padded_x, padded_y, lengths=lengths)
            else:
                logits = self.model(padded_x, lengths=lengths)  # (B, T, 4)
                loss = F.cross_entropy(
                    logits.view(-1, NUM_BIEO_LABELS),
                    padded_y.view(-1),
                    weight=self._ce_weight,
                    ignore_index=-100,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": avg_loss, "lr": current_lr}

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        sequences: list[np.ndarray],
        labels: list[list[int]],
        batch_size: int = 16,
    ) -> dict[str, float]:
        """Évalue le modèle sur un ensemble de séquences.

        Métriques retournées :
        - loss         : perte moyenne
        - accuracy     : exactitude globale
        - f1_B, f1_E   : F1 par classe de frontière
        - f1_boundary  : mean(f1_B, f1_E) — métrique principale
        - iou_mean     : IoU moyen des segments détectés vs vrais

        Returns:
            Dictionnaire de métriques.
        """
        from sklearn.metrics import f1_score, accuracy_score  # type: ignore

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[int] = []
        all_true: list[int] = []

        with torch.no_grad():
            for batch_start in range(0, len(sequences), batch_size):
                batch_seqs = sequences[batch_start: batch_start + batch_size]
                batch_lbls = labels[batch_start: batch_start + batch_size]

                padded_x, padded_y, lengths = self._collate(batch_seqs, batch_lbls)

                if self.model.use_crf and self.model.crf is not None:
                    loss = self.model.crf_loss(padded_x, padded_y, lengths=lengths)
                else:
                    logits = self.model(padded_x, lengths=lengths)
                    loss = F.cross_entropy(
                        logits.view(-1, NUM_BIEO_LABELS),
                        padded_y.view(-1),
                        weight=self._ce_weight,
                        ignore_index=-100,
                    )

                total_loss += loss.item()
                n_batches += 1

                # Collect predictions (sans padding)
                if self.model.use_crf and self.model.crf is not None:
                    decoded = self.model.decode(padded_x, lengths=lengths)
                    for i, preds_seq in enumerate(decoded):
                        true_len = int(lengths[i].item())
                        all_preds.extend(preds_seq[:true_len])
                        all_true.extend(batch_lbls[i][:true_len])
                else:
                    logits = self.model(padded_x, lengths=lengths)
                    preds_batch = logits.argmax(dim=-1)  # (B, T)
                    for i in range(len(batch_seqs)):
                        true_len = int(lengths[i].item())
                        all_preds.extend(preds_batch[i, :true_len].cpu().tolist())
                        all_true.extend(batch_lbls[i][:true_len])

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = float(accuracy_score(all_true, all_preds))

        # F1 par classe (zero_division=0 évite les warnings sur classes absentes)
        f1_per_class = f1_score(
            all_true,
            all_preds,
            labels=[LABEL_B, LABEL_I, LABEL_E, LABEL_O],
            average=None,
            zero_division=0,
        )
        f1_B = float(f1_per_class[0])
        f1_E = float(f1_per_class[2])
        f1_boundary = (f1_B + f1_E) / 2.0

        # IoU moyen des segments
        iou_mean = self._compute_iou_mean(sequences, labels)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_boundary": f1_boundary,
            "f1_B": f1_B,
            "f1_E": f1_E,
            "iou_mean": iou_mean,
        }

    def _compute_iou_mean(
        self,
        sequences: list[np.ndarray],
        labels: list[list[int]],
    ) -> float:
        """Calcule l'IoU moyen entre les segments prédits et les segments vrais.

        L'IoU est calculé par paire greedy (segment prédit le plus proche du segment vrai).
        """
        ious: list[float] = []

        for seq, lbl in zip(sequences, labels):
            # Segments vrais extraits des labels
            true_segs = self._labels_to_segments(lbl)
            # Segments prédits via detect_boundaries
            pred_segs = self.model.detect_boundaries(
                seq,
                min_sign_frames=self.config.min_sign_frames,
                merge_gap_frames=self.config.merge_gap_frames,
                boundary_threshold=self.config.boundary_threshold,
                device=self.device,
            )

            if not true_segs and not pred_segs:
                ious.append(1.0)
                continue
            if not true_segs or not pred_segs:
                ious.append(0.0)
                continue

            # Greedy matching : pour chaque segment vrai, trouver le prédit le plus proche
            matched_pred = set()
            for ts, te in true_segs:
                best_iou = 0.0
                best_j = -1
                for j, (ps, pe) in enumerate(pred_segs):
                    if j in matched_pred:
                        continue
                    inter = max(0, min(te, pe) - max(ts, ps) + 1)
                    union = max(te, pe) - min(ts, ps) + 1
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0:
                    matched_pred.add(best_j)
                ious.append(best_iou)

        return float(np.mean(ious)) if ious else 0.0

    @staticmethod
    def _labels_to_segments(labels: list[int]) -> list[tuple[int, int]]:
        """Convertit une liste de labels BIEO en liste de segments (start, end)."""
        segments: list[tuple[int, int]] = []
        in_sign = False
        start = 0
        for i, lbl in enumerate(labels):
            if lbl in (LABEL_B, LABEL_I, LABEL_E) and not in_sign:
                start = i
                in_sign = True
            elif lbl == LABEL_O and in_sign:
                segments.append((start, i - 1))
                in_sign = False
        if in_sign:
            segments.append((start, len(labels) - 1))
        return segments

    # ── Full training loop ──────────────────────────────────────────────────

    def fit(
        self,
        train_sequences: list[np.ndarray],
        train_labels: list[list[int]],
        val_sequences: list[np.ndarray],
        val_labels: list[list[int]],
        *,
        epochs: int = 30,
        batch_size: int = 16,
        patience: int = 5,
        checkpoint_path: Optional[str] = None,
    ) -> list[dict]:
        """Boucle d'entraînement complète avec early stopping.

        Args:
            train_sequences: Séquences d'entraînement.
            train_labels: Labels BIEO correspondants.
            val_sequences: Séquences de validation.
            val_labels: Labels BIEO de validation.
            epochs: Nombre maximal d'époques.
            batch_size: Taille des mini-batchs.
            patience: Nombre d'époques sans amélioration avant arrêt.
            checkpoint_path: Chemin de sauvegarde du meilleur modèle (optionnel).

        Returns:
            Liste des métriques par époque :
            [{"epoch", "train_loss", "val_loss", "val_f1_boundary", ...}]
        """
        history: list[dict] = []
        best_f1 = -1.0
        no_improvement = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_sequences, train_labels, batch_size=batch_size)
            val_metrics = self.evaluate(val_sequences, val_labels, batch_size=batch_size)

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "lr": train_metrics["lr"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(entry)

            logger.info(
                "segmentation_train_epoch",
                epoch=epoch,
                train_loss=f"{train_metrics['loss']:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                val_f1_boundary=f"{val_metrics['f1_boundary']:.4f}",
                val_iou=f"{val_metrics['iou_mean']:.4f}",
            )

            # Early stopping et checkpoint
            if val_metrics["f1_boundary"] > best_f1:
                best_f1 = val_metrics["f1_boundary"]
                no_improvement = 0
                if checkpoint_path is not None:
                    self.model.save_checkpoint(checkpoint_path)
                    logger.info(
                        "segmentation_checkpoint_updated",
                        epoch=epoch,
                        val_f1_boundary=f"{best_f1:.4f}",
                    )
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(
                        "segmentation_early_stopping",
                        epoch=epoch,
                        patience=patience,
                        best_f1_boundary=f"{best_f1:.4f}",
                    )
                    break

        return history
