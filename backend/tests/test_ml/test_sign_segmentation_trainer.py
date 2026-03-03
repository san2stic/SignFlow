"""Tests unitaires pour les nouvelles fonctionnalités de backend/app/ml/sign_segmentation.py.

Couvre :
- generate_bieo_corpus()
- SegmentationTrainer.train_epoch(), evaluate(), fit()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.sign_segmentation import (
    LABEL_B,
    LABEL_I,
    LABEL_E,
    LABEL_O,
    NUM_BIEO_LABELS,
    SignBoundaryDetector,
    SegmentationConfig,
    SegmentationTrainer,
    generate_bieo_corpus,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_DIM = 457  # feature_dim par défaut de generate_bieo_corpus


@pytest.fixture()
def small_corpus() -> tuple[list[np.ndarray], list[list[int]]]:
    """Corpus réduit pour les tests rapides (n=20)."""
    return generate_bieo_corpus(
        n_sequences=20,
        feature_dim=FEATURE_DIM,
        min_seq_len=20,
        max_seq_len=40,
        min_sign_frames=5,
        max_sign_frames=12,
        seed=42,
    )


@pytest.fixture()
def small_model() -> SignBoundaryDetector:
    """Modèle BiLSTM minimal pour les tests."""
    return SignBoundaryDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        use_crf=False,
        input_selection="velocity+nmm+handshape",
    )


@pytest.fixture()
def trainer(small_model: SignBoundaryDetector) -> SegmentationTrainer:
    """Trainer minimal sur CPU."""
    config = SegmentationConfig(
        feature_dim=FEATURE_DIM,
        hidden_dim=16,
        min_sign_frames=5,
        merge_gap_frames=2,
    )
    return SegmentationTrainer(
        model=small_model,
        config=config,
        lr=1e-3,
        device="cpu",
    )


# ── TestGenerateBieoCorpus ────────────────────────────────────────────────────


class TestGenerateBieoCorpus:
    def test_corpus_length(self, small_corpus: tuple) -> None:
        """n_sequences arrays doivent être retournés."""
        seqs, labels = small_corpus
        assert len(seqs) == 20, f"Attendu 20 séquences, obtenu {len(seqs)}"
        assert len(labels) == 20

    def test_array_shapes(self, small_corpus: tuple) -> None:
        """Chaque array doit être (T_i, FEATURE_DIM) float32 avec T_i >= min_seq_len."""
        seqs, _ = small_corpus
        for i, seq in enumerate(seqs):
            assert isinstance(seq, np.ndarray), f"seq[{i}] n'est pas un np.ndarray"
            assert seq.ndim == 2, f"seq[{i}].ndim={seq.ndim} ≠ 2"
            assert seq.shape[1] == FEATURE_DIM, (
                f"seq[{i}].shape[1]={seq.shape[1]} ≠ {FEATURE_DIM}"
            )
            assert seq.dtype == np.float32, f"seq[{i}].dtype={seq.dtype} ≠ float32"
            assert seq.shape[0] >= 20, (
                f"seq[{i}] longueur {seq.shape[0]} < min_seq_len=20"
            )

    def test_label_alignment(self, small_corpus: tuple) -> None:
        """len(labels[i]) == len(sequences[i]) pour tout i."""
        seqs, labels = small_corpus
        for i, (seq, lbl) in enumerate(zip(seqs, labels)):
            assert len(lbl) == seq.shape[0], (
                f"Séquence {i} : {len(lbl)} labels ≠ {seq.shape[0]} frames"
            )

    def test_labels_are_bieo(self, small_corpus: tuple) -> None:
        """Tous les labels doivent être dans {0, 1, 2, 3} (BIEO)."""
        _, labels = small_corpus
        valid = {LABEL_B, LABEL_I, LABEL_E, LABEL_O}
        for i, lbl_seq in enumerate(labels):
            for frame_idx, lbl in enumerate(lbl_seq):
                assert lbl in valid, (
                    f"Label invalide {lbl} en seq[{i}][{frame_idx}]"
                )

    def test_reproducibility(self) -> None:
        """Même seed → mêmes premières features de la première séquence."""
        seqs_a, _ = generate_bieo_corpus(n_sequences=5, seed=0)
        seqs_b, _ = generate_bieo_corpus(n_sequences=5, seed=0)
        np.testing.assert_array_equal(
            seqs_a[0],
            seqs_b[0],
            err_msg="La reproductibilité est rompue pour generate_bieo_corpus",
        )

    def test_different_seeds(self) -> None:
        """Seeds différentes doivent produire des séquences différentes."""
        seqs_a, _ = generate_bieo_corpus(n_sequences=5, seed=1)
        seqs_b, _ = generate_bieo_corpus(n_sequences=5, seed=2)
        # Les premières frames ne doivent pas être identiques
        assert not np.array_equal(seqs_a[0], seqs_b[0]), (
            "Deux seeds différentes donnent la même séquence"
        )

    def test_contains_all_label_types(self) -> None:
        """Un corpus suffisamment grand doit contenir B, I, E et O."""
        _, labels = generate_bieo_corpus(n_sequences=30, seed=99)
        all_labels = {lbl for seq in labels for lbl in seq}
        for expected in (LABEL_B, LABEL_I, LABEL_E, LABEL_O):
            assert expected in all_labels, (
                f"Label {expected} manquant dans le corpus généré"
            )


# ── TestSegmentationTrainer ───────────────────────────────────────────────────


class TestSegmentationTrainer:
    def test_train_epoch_returns_loss(
        self,
        trainer: SegmentationTrainer,
        small_corpus: tuple,
    ) -> None:
        """train_epoch() doit retourner un dict avec la clé 'loss' float."""
        seqs, labels = small_corpus
        metrics = trainer.train_epoch(seqs, labels, batch_size=4)
        assert "loss" in metrics, "Clé 'loss' manquante dans train_epoch()"
        loss = metrics["loss"]
        assert isinstance(loss, float), f"loss doit être un float, got {type(loss)}"
        assert loss >= 0.0, f"loss ne doit pas être négatif, got {loss}"
        assert not np.isnan(loss), "loss ne doit pas être NaN"

    def test_evaluate_returns_metrics(
        self,
        trainer: SegmentationTrainer,
        small_corpus: tuple,
    ) -> None:
        """evaluate() doit retourner un dict avec 'loss', 'f1_boundary', 'iou_mean'."""
        seqs, labels = small_corpus
        metrics = trainer.evaluate(seqs, labels, batch_size=4)
        for key in ("loss", "f1_boundary", "iou_mean"):
            assert key in metrics, f"Clé manquante dans evaluate() : '{key}'"
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["f1_boundary"], float)
        assert isinstance(metrics["iou_mean"], float)

    def test_evaluate_f1_boundary_range(
        self,
        trainer: SegmentationTrainer,
        small_corpus: tuple,
    ) -> None:
        """f1_boundary doit être dans [0.0, 1.0]."""
        seqs, labels = small_corpus
        metrics = trainer.evaluate(seqs, labels, batch_size=4)
        assert 0.0 <= metrics["f1_boundary"] <= 1.0, (
            f"f1_boundary hors [0, 1] : {metrics['f1_boundary']}"
        )

    def test_fit_early_stopping(
        self,
        small_corpus: tuple,
    ) -> None:
        """fit() doit s'arrêter avant epochs max si patience est dépassée."""
        seqs, labels = small_corpus
        # Split train/val 80/20
        n_train = 16
        train_seqs, train_lbls = seqs[:n_train], labels[:n_train]
        val_seqs, val_lbls = seqs[n_train:], labels[n_train:]

        model = SignBoundaryDetector(
            feature_dim=FEATURE_DIM,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        config = SegmentationConfig(feature_dim=FEATURE_DIM, hidden_dim=16)
        trainer_local = SegmentationTrainer(model=model, config=config, lr=1e-3, device="cpu")

        history = trainer_local.fit(
            train_seqs,
            train_lbls,
            val_seqs,
            val_lbls,
            epochs=20,
            batch_size=4,
            patience=2,  # patience très courte → arrêt rapide
        )
        # Avec patience=2 et epochs=20, on ne doit pas faire toutes les 20 époques
        assert len(history) <= 20, "fit() doit s'arrêter avant ou à epoch_max"

    def test_fit_returns_history(
        self,
        small_corpus: tuple,
    ) -> None:
        """fit() doit retourner une liste de dicts avec 'epoch', 'train_loss', 'val_f1_boundary'."""
        seqs, labels = small_corpus
        n_train = 16
        train_seqs, train_lbls = seqs[:n_train], labels[:n_train]
        val_seqs, val_lbls = seqs[n_train:], labels[n_train:]

        model = SignBoundaryDetector(
            feature_dim=FEATURE_DIM,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        config = SegmentationConfig(feature_dim=FEATURE_DIM, hidden_dim=16)
        trainer_local = SegmentationTrainer(model=model, config=config, lr=1e-3, device="cpu")

        history = trainer_local.fit(
            train_seqs,
            train_lbls,
            val_seqs,
            val_lbls,
            epochs=3,
            batch_size=4,
            patience=5,
        )
        assert isinstance(history, list), "fit() doit retourner une liste"
        assert len(history) > 0, "L'historique ne doit pas être vide"
        for entry in history:
            for key in ("epoch", "train_loss", "val_f1_boundary"):
                assert key in entry, f"Clé manquante dans l'historique : '{key}'"

    def test_checkpoint_save_load(
        self,
        small_corpus: tuple,
        tmp_path: Path,
    ) -> None:
        """Après fit avec checkpoint_path, le fichier doit exister et être chargeable."""
        seqs, labels = small_corpus
        n_train = 16
        train_seqs, train_lbls = seqs[:n_train], labels[:n_train]
        val_seqs, val_lbls = seqs[n_train:], labels[n_train:]

        model = SignBoundaryDetector(
            feature_dim=FEATURE_DIM,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        config = SegmentationConfig(feature_dim=FEATURE_DIM, hidden_dim=16)
        trainer_local = SegmentationTrainer(model=model, config=config, lr=1e-3, device="cpu")

        ckpt_path = str(tmp_path / "seg_test.pt")
        trainer_local.fit(
            train_seqs,
            train_lbls,
            val_seqs,
            val_lbls,
            epochs=3,
            batch_size=4,
            patience=5,
            checkpoint_path=ckpt_path,
        )

        assert Path(ckpt_path).exists(), f"Le checkpoint {ckpt_path} n'a pas été créé"

        # Vérifier que le checkpoint peut être rechargé
        loaded_model = SignBoundaryDetector.load_checkpoint(ckpt_path)
        assert isinstance(loaded_model, SignBoundaryDetector), (
            "load_checkpoint() doit retourner un SignBoundaryDetector"
        )
        # Vérifier que le modèle reconstruit peut faire des inférences
        loaded_model.eval()
        dummy_input = torch.zeros(1, 30, FEATURE_DIM)
        with torch.no_grad():
            logits = loaded_model(dummy_input)
        assert logits.shape == (1, 30, NUM_BIEO_LABELS), (
            f"Logits shape inattendue : {logits.shape}"
        )
