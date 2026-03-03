#!/usr/bin/env python3
"""
Script standalone pour entraîner le BiLSTM de segmentation SignBoundaryDetector.

Usage :
    python backend/scripts/train_segmentation.py
    python backend/scripts/train_segmentation.py --n-sequences 300 --epochs 50 --use-crf
    python backend/scripts/train_segmentation.py \\
        --output backend/data/models/sign_segmentation.pt \\
        --lr 5e-4 --hidden-dim 256 --patience 8

Le script :
    1. Génère un corpus synthétique BIEO via generate_bieo_corpus().
    2. Effectue un split train/val selon --val-ratio.
    3. Instancie SegmentationConfig + SignBoundaryDetector.
    4. Instancie SegmentationTrainer.
    5. Lance trainer.fit() avec checkpoint_path.
    6. Affiche les métriques finales (val f1_boundary, iou_mean).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ajout du répertoire backend au sys.path pour permettre les imports app.*
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_segmentation")


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Entraîne le BiLSTM de segmentation SignBoundaryDetector sur corpus BIEO synthétique.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=150,
        dest="n_sequences",
        help="Nombre de séquences dans le corpus synthétique.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Nombre maximum d'époques d'entraînement.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate pour AdamW.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        dest="hidden_dim",
        help="Dimension cachée du BiLSTM.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        dest="num_layers",
        help="Nombre de couches BiLSTM.",
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        dest="use_crf",
        help="Activer la tête CRF (nécessite pytorch-crf installé).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/models/sign_segmentation.pt",
        help="Chemin de sauvegarde du checkpoint (.pt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        dest="val_ratio",
        help="Fraction des données réservée à la validation (0 < val_ratio < 1).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Nombre d'époques sans amélioration avant early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Taille des mini-batchs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Imports tardifs (après ajout au sys.path)
    # ------------------------------------------------------------------
    from app.ml.sign_segmentation import (  # noqa: E402
        SegmentationConfig,
        SegmentationTrainer,
        SignBoundaryDetector,
        generate_bieo_corpus,
    )

    # ------------------------------------------------------------------
    # 1. Génération du corpus synthétique
    # ------------------------------------------------------------------
    logger.info(
        "Génération du corpus synthétique : %d séquences (seed=%d)",
        args.n_sequences,
        args.seed,
    )
    sequences, labels = generate_bieo_corpus(
        n_sequences=args.n_sequences,
        seed=args.seed,
    )
    logger.info("Corpus généré : %d séquences.", len(sequences))

    # ------------------------------------------------------------------
    # 2. Split train / val
    # ------------------------------------------------------------------
    import math

    n_val = max(1, math.floor(len(sequences) * args.val_ratio))
    n_train = len(sequences) - n_val

    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]

    logger.info("Split : %d train / %d val", n_train, n_val)

    # ------------------------------------------------------------------
    # 3. Instanciation du modèle
    # ------------------------------------------------------------------
    config = SegmentationConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_crf=args.use_crf,
    )
    model = SignBoundaryDetector(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        use_crf=config.use_crf,
        input_selection=config.input_selection,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Modèle : hidden_dim=%d, num_layers=%d, use_crf=%s — %d paramètres entraînables",
        args.hidden_dim,
        args.num_layers,
        args.use_crf,
        n_params,
    )

    # ------------------------------------------------------------------
    # 4. Instanciation du trainer
    # ------------------------------------------------------------------
    trainer = SegmentationTrainer(
        model=model,
        config=config,
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # 5. Entraînement
    # ------------------------------------------------------------------
    output_path = str(args.output)
    logger.info(
        "Entraînement : %d époques max, patience=%d, checkpoint → %s",
        args.epochs,
        args.patience,
        output_path,
    )

    history = trainer.fit(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_path=output_path,
    )

    # ------------------------------------------------------------------
    # 6. Métriques finales
    # ------------------------------------------------------------------
    if history:
        best_entry = max(history, key=lambda e: e.get("val_f1_boundary", 0.0))
        logger.info(
            "=== Résultats finaux ===\n"
            "  Meilleure époque  : %d / %d\n"
            "  val_f1_boundary   : %.4f\n"
            "  val_iou_mean      : %.4f\n"
            "  val_f1_B          : %.4f\n"
            "  val_f1_E          : %.4f\n"
            "  val_accuracy      : %.4f\n"
            "  val_loss          : %.4f",
            best_entry["epoch"],
            len(history),
            best_entry.get("val_f1_boundary", 0.0),
            best_entry.get("val_iou_mean", 0.0),
            best_entry.get("val_f1_B", 0.0),
            best_entry.get("val_f1_E", 0.0),
            best_entry.get("val_accuracy", 0.0),
            best_entry.get("val_loss", 0.0),
        )
        print(
            f"\n✓ Entraînement terminé.\n"
            f"  val_f1_boundary = {best_entry.get('val_f1_boundary', 0.0):.4f}\n"
            f"  iou_mean        = {best_entry.get('val_iou_mean', 0.0):.4f}\n"
            f"  Checkpoint      → {output_path}"
        )
    else:
        logger.warning("Aucune époque d'entraînement complétée.")


if __name__ == "__main__":
    main()
