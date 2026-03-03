#!/usr/bin/env python3
"""
Script standalone pour entraîner le modèle seq2seq LSFB → Français.

Usage :
    python backend/scripts/train_seq2seq.py
    python backend/scripts/train_seq2seq.py --n-pairs 1000 --epochs 100 --beam-size 3
    python backend/scripts/train_seq2seq.py \\
        --output backend/data/models/seq2seq_lsfb.pkl \\
        --lr 5e-4 --hidden-dim 256 --patience 10

Le script :
    1. Génère un corpus synthétique via generate_seq2seq_corpus().
    2. Effectue un split train/val selon --val-ratio.
    3. Construit les vocabulaires src et tgt depuis le corpus train.
    4. Instancie Seq2SeqTranslator + Seq2SeqTrainer.
    5. Lance trainer.fit() avec early stopping.
    6. Affiche les métriques finales (val_loss, val_bleu1).
    7. Traduit 3 exemples de démonstration du jeu de test.
"""

from __future__ import annotations

import argparse
import logging
import math
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
logger = logging.getLogger("train_seq2seq")


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne le modèle seq2seq encoder-decoder LSFB → Français "
            "sur un corpus synthétique."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=500,
        dest="n_pairs",
        help="Nombre de paires synthétiques à générer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre maximum d'époques d'entraînement.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate pour Adam.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        dest="embed_dim",
        help="Dimension des embeddings (source et cible).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        dest="hidden_dim",
        help="Dimension cachée du BiLSTM encodeur / LSTM décodeur.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/models/seq2seq_lsfb.pkl",
        help="Chemin de sauvegarde du modèle entraîné (.pkl).",
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
        default=7,
        help="Nombre d'époques sans amélioration avant early stopping.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        dest="beam_size",
        help="Taille du faisceau pour le décodage (1 = greedy).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Taille des mini-batchs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device PyTorch ('cpu', 'cuda', 'mps').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Imports tardifs (après ajout au sys.path)
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch n'est pas installé. "
            "Veuillez l'installer : pip install torch"
        )
        sys.exit(1)

    from app.ml.grammar.lsfb_translator import (  # noqa: E402
        Seq2SeqTrainer,
        Seq2SeqTranslator,
        Seq2SeqVocab,
        generate_seq2seq_corpus,
    )

    # ------------------------------------------------------------------
    # 1. Génération du corpus synthétique
    # ------------------------------------------------------------------
    logger.info(
        "Génération du corpus synthétique : %d paires (seed=%d)",
        args.n_pairs,
        args.seed,
    )
    all_pairs = generate_seq2seq_corpus(n_pairs=args.n_pairs, seed=args.seed)
    logger.info("Corpus généré : %d paires.", len(all_pairs))

    # ------------------------------------------------------------------
    # 2. Split train / val
    # ------------------------------------------------------------------
    n_val = max(1, math.floor(len(all_pairs) * args.val_ratio))
    n_train = len(all_pairs) - n_val

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    logger.info("Split : %d train / %d val", n_train, n_val)

    # ------------------------------------------------------------------
    # 3. Construction des vocabulaires
    # ------------------------------------------------------------------
    src_vocab = Seq2SeqVocab()
    tgt_vocab = Seq2SeqVocab()

    src_sequences = [src for src, _ in train_pairs]
    tgt_sequences = [tgt for _, tgt in train_pairs]

    src_vocab.build_from_sequences(src_sequences)
    tgt_vocab.build_from_sequences(tgt_sequences)

    logger.info(
        "Vocabulaires construits : src=%d tokens, tgt=%d tokens",
        len(src_vocab),
        len(tgt_vocab),
    )

    # ------------------------------------------------------------------
    # 4. Instanciation du modèle
    # ------------------------------------------------------------------
    model = Seq2SeqTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.3,
        max_decode_len=30,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Modèle Seq2SeqTranslator :\n"
        "  embed_dim   = %d\n"
        "  hidden_dim  = %d\n"
        "  src_vocab   = %d\n"
        "  tgt_vocab   = %d\n"
        "  Paramètres  = %d",
        args.embed_dim,
        args.hidden_dim,
        len(src_vocab),
        len(tgt_vocab),
        n_params,
    )

    # ------------------------------------------------------------------
    # 5. Instanciation du trainer
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        lr=args.lr,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 6. Entraînement
    # ------------------------------------------------------------------
    output_path = str(args.output)
    logger.info(
        "Entraînement : %d époques max, patience=%d, batch_size=%d, device=%s",
        args.epochs,
        args.patience,
        args.batch_size,
        args.device,
    )

    history = trainer.fit(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # ------------------------------------------------------------------
    # 7. Sauvegarde
    # ------------------------------------------------------------------
    trainer.save(output_path)

    # ------------------------------------------------------------------
    # 8. Métriques finales
    # ------------------------------------------------------------------
    if history:
        best_entry = min(history, key=lambda e: e.get("val_loss", float("inf")))
        final_entry = history[-1]

        logger.info(
            "=== Résultats finaux ===\n"
            "  Époques totales   : %d\n"
            "  Meilleure époque  : %d\n"
            "  Meilleur val_loss : %.4f\n"
            "  val_bleu1         : %.4f\n"
            "  Dernier val_loss  : %.4f\n"
            "  Dernier val_bleu1 : %.4f",
            len(history),
            best_entry["epoch"],
            best_entry.get("val_loss", 0.0),
            best_entry.get("val_bleu1", 0.0),
            final_entry.get("val_loss", 0.0),
            final_entry.get("val_bleu1", 0.0),
        )
        print(
            f"\n✓ Entraînement terminé ({len(history)} époques).\n"
            f"  Meilleure époque  = {best_entry['epoch']}\n"
            f"  val_loss (best)   = {best_entry.get('val_loss', 0.0):.4f}\n"
            f"  val_bleu1 (best)  = {best_entry.get('val_bleu1', 0.0):.4f}\n"
            f"  Modèle sauvegardé → {output_path}"
        )
    else:
        logger.warning("Aucune époque d'entraînement complétée.")

    # ------------------------------------------------------------------
    # 9. Démonstration : traduire 3 exemples du corpus de validation
    # ------------------------------------------------------------------
    print("\n=== Démonstration : 3 exemples de traduction ===")

    model.eval()
    demo_pairs = val_pairs[:3] if len(val_pairs) >= 3 else val_pairs

    for i, (src_tokens, tgt_tokens) in enumerate(demo_pairs, 1):
        src_indices = src_vocab.encode(src_tokens)
        src_tensor = torch.tensor([src_indices], dtype=torch.long)
        src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)

        # Déplacer sur le device du modèle
        device = next(model.parameters()).device
        src_tensor = src_tensor.to(device)
        src_lengths = src_lengths.to(device)

        pred_indices = model.translate(
            src_tensor, src_lengths, beam_size=args.beam_size
        )
        pred_tokens = tgt_vocab.decode(pred_indices)

        src_str = " ".join(src_tokens)
        ref_str = " ".join(tgt_tokens)
        pred_str = " ".join(pred_tokens)

        print(
            f"\n  [{i}] Source LSFB : {src_str}\n"
            f"       Référence   : {ref_str}\n"
            f"       Prédiction  : {pred_str}"
        )

    print()


if __name__ == "__main__":
    main()
