#!/usr/bin/env python3
"""
Script standalone pour entraîner le CRF grammatical LSFB.

Usage (corpus synthétique) :
    python backend/scripts/train_crf.py

Usage (corpus JSON externe) :
    python backend/scripts/train_crf.py --corpus path/to/corpus.json

Format du corpus JSON :
    [
        {
            "tokens": [
                {
                    "label": "JE",
                    "confidence": 0.95,
                    "start_frame": 0,
                    "end_frame": 10,
                    "nmm_features": [0.1, 0.2, ...],   // 32 floats
                    "signing_space": [0.0, 0.1, ...],   // 18 floats
                    "grammar_role": "subject"
                },
                ...
            ],
            "labels": ["B-SUBJ", "B-VERB", "B-OBJ"]
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ajout du répertoire racine au sys.path pour permettre les imports app.*
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_crf")


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Entraîne le CRF grammatical LSFB (LSFBSequenceTagger).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=(
            "Chemin vers un corpus JSON annoté "
            "[{tokens: [...], labels: [...]}]. "
            "Si absent, un corpus synthétique est généré."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/models/lsfb_crf.pkl",
        help="Chemin de sortie du modèle CRF (.pkl).",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.1,
        help="Régularisation L1 du CRF (lbfgs).",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.05,
        help="Régularisation L2 du CRF (lbfgs).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        dest="max_iter",
        help="Nombre maximum d'itérations lbfgs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire (corpus synthétique + split train/test).",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=500,
        dest="n_sequences",
        help="Nombre de séquences synthétiques à générer (ignoré si --corpus fourni).",
    )
    return parser.parse_args()


def load_corpus_from_json(
    path: str,
) -> tuple[list, list[list[str]]]:
    """Charge un corpus depuis un fichier JSON.

    Args:
        path: Chemin vers le fichier JSON.

    Returns:
        Tuple (sequences, labels).

    Raises:
        SystemExit: En cas d'erreur de chargement.
    """
    from app.ml.grammar.lsfb_rules import GrammarRole, SignToken

    json_path = Path(path)
    if not json_path.exists():
        logger.error("Corpus file not found: %s", json_path)
        sys.exit(1)

    logger.info("Loading corpus from %s", json_path)
    with json_path.open("r", encoding="utf-8") as fh:
        corpus_data = json.load(fh)

    sequences: list[list[SignToken]] = []
    labels: list[list[str]] = []

    for idx, entry in enumerate(corpus_data):
        try:
            raw_tokens = entry["tokens"]
            raw_labels = entry["labels"]

            if len(raw_tokens) != len(raw_labels):
                logger.warning(
                    "Sequence %d: token/label length mismatch (%d vs %d), skipping.",
                    idx, len(raw_tokens), len(raw_labels),
                )
                continue

            token_list: list[SignToken] = []
            for rt in raw_tokens:
                nmm = np.array(rt.get("nmm_features", [0.0] * 32), dtype=np.float32)
                if len(nmm) < 32:
                    nmm = np.pad(nmm, (0, 32 - len(nmm)))
                space = np.array(rt.get("signing_space", [0.0] * 18), dtype=np.float32)
                if len(space) < 18:
                    space = np.pad(space, (0, 18 - len(space)))

                role_str = rt.get("grammar_role", "unknown")
                try:
                    role = GrammarRole(role_str)
                except ValueError:
                    role = GrammarRole.UNKNOWN

                token_list.append(SignToken(
                    label=str(rt.get("label", "")),
                    confidence=float(rt.get("confidence", 1.0)),
                    start_frame=int(rt.get("start_frame", 0)),
                    end_frame=int(rt.get("end_frame", 0)),
                    nmm_features=nmm,
                    signing_space=space,
                    grammar_role=role,
                ))

            sequences.append(token_list)
            labels.append(list(raw_labels))

        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Sequence %d: parse error (%s), skipping.", idx, exc)
            continue

    if not sequences:
        logger.error("No valid sequences found in corpus file: %s", path)
        sys.exit(1)

    logger.info("Loaded %d sequences from corpus.", len(sequences))
    return sequences, labels


def train_test_split(
    sequences: list,
    labels: list[list[str]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list[list[str]], list, list[list[str]]]:
    """Split déterministe 80/20 train/test.

    Args:
        sequences: Liste de séquences.
        labels: Labels correspondants.
        test_ratio: Fraction réservée au test (défaut 0.2).
        seed: Graine numpy.

    Returns:
        Tuple (train_seqs, train_labels, test_seqs, test_labels).
    """
    rng = np.random.default_rng(seed)
    n = len(sequences)
    indices = np.arange(n)
    rng.shuffle(indices)

    split_idx = max(1, int(n * (1 - test_ratio)))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_seqs, train_labels, test_seqs, test_labels


def main() -> None:
    """Point d'entrée principal du script d'entraînement."""
    args = parse_args()

    # ----------------------------------------------------------------
    # Import conditionnel de sklearn-crfsuite
    # ----------------------------------------------------------------
    try:
        import sklearn_crfsuite  # noqa: F401  type: ignore[import]
    except ImportError:
        logger.error(
            "sklearn-crfsuite is not installed. "
            "Please run: pip install sklearn-crfsuite"
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # Import des modules SignFlow
    # ----------------------------------------------------------------
    try:
        from app.ml.grammar.lsfb_crf import (
            LSFBSequenceTagger,
            generate_synthetic_corpus,
        )
    except ImportError as exc:
        logger.error(
            "Failed to import SignFlow modules. "
            "Make sure you run this script from the repo root or that "
            "backend/ is in PYTHONPATH. Error: %s",
            exc,
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # Chargement ou génération du corpus
    # ----------------------------------------------------------------
    if args.corpus is not None:
        logger.info("Loading corpus from: %s", args.corpus)
        sequences, labels = load_corpus_from_json(args.corpus)
    else:
        logger.info(
            "No corpus provided — generating synthetic corpus "
            "(%d sequences, seed=%d).",
            args.n_sequences,
            args.seed,
        )
        sequences, labels = generate_synthetic_corpus(
            n_sequences=args.n_sequences,
            seed=args.seed,
        )
        logger.info("Synthetic corpus generated: %d sequences.", len(sequences))

    # ----------------------------------------------------------------
    # Split train / test
    # ----------------------------------------------------------------
    train_seqs, train_labels, test_seqs, test_labels = train_test_split(
        sequences, labels, test_ratio=0.2, seed=args.seed
    )
    logger.info(
        "Split: %d train / %d test sequences.",
        len(train_seqs),
        len(test_seqs),
    )

    # ----------------------------------------------------------------
    # Entraînement
    # ----------------------------------------------------------------
    logger.info(
        "Training CRF (c1=%.4f, c2=%.4f, max_iter=%d)...",
        args.c1,
        args.c2,
        args.max_iter,
    )
    tagger = LSFBSequenceTagger(
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
    )
    tagger.fit(train_seqs, train_labels)
    logger.info("CRF training complete.")

    # ----------------------------------------------------------------
    # Évaluation sur le test set
    # ----------------------------------------------------------------
    if test_seqs:
        logger.info("Evaluating on %d test sequences...", len(test_seqs))
        metrics = tagger.evaluate(test_seqs, test_labels)

        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Weighted F1        : {metrics['weighted_f1']:.4f}")
        print(f"  Weighted Precision : {metrics['weighted_precision']:.4f}")
        print(f"  Weighted Recall    : {metrics['weighted_recall']:.4f}")
        print("=" * 60)
        print("\n--- Per-class report ---")
        print(metrics["report_str"])
    else:
        logger.warning("No test sequences available, skipping evaluation.")

    # ----------------------------------------------------------------
    # Sauvegarde du modèle
    # ----------------------------------------------------------------
    output_path = Path(args.output)
    logger.info("Saving model to: %s", output_path)
    tagger.save(output_path)
    logger.info("Model saved successfully.")

    print(f"\n✓ Model saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
