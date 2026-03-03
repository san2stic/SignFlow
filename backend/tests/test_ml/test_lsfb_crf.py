"""Tests unitaires pour backend/app/ml/grammar/lsfb_crf.py.

Couvre :
- generate_synthetic_corpus()
- _token_features() — signing_space, bigrammes de rôle, is_first_in_domain
- LSFBSequenceTagger.fit() / predict() / evaluate()
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.grammar.lsfb_crf import (
    GRAMMAR_TAGS,
    _CRF_AVAILABLE,
    _token_features,
    generate_synthetic_corpus,
    LSFBSequenceTagger,
)
from app.ml.grammar.lsfb_rules import GrammarRole, SignToken


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_token(
    label: str = "MANGER",
    confidence: float = 0.9,
    role: GrammarRole = GrammarRole.VERB,
    nmm: np.ndarray | None = None,
    signing_space: np.ndarray | None = None,
) -> SignToken:
    """Crée un SignToken minimal pour les tests."""
    return SignToken(
        label=label,
        confidence=confidence,
        nmm_features=nmm if nmm is not None else np.zeros(32, dtype=np.float32),
        signing_space=signing_space if signing_space is not None else np.zeros(18, dtype=np.float32),
        grammar_role=role,
    )


def _make_small_corpus(n: int = 20, seed: int = 0) -> tuple[list, list]:
    """Génère un corpus réduit pour les tests rapides."""
    return generate_synthetic_corpus(n_sequences=n, min_len=3, max_len=6, seed=seed)


# ── TestGenerateSyntheticCorpus ────────────────────────────────────────────────


class TestGenerateSyntheticCorpus:
    def test_corpus_length(self) -> None:
        """Vérifie que le nombre de séquences retournées est correct."""
        seqs, labels = generate_synthetic_corpus(n_sequences=30, seed=0)
        assert len(seqs) == 30, f"Attendu 30 séquences, obtenu {len(seqs)}"
        assert len(labels) == 30

    def test_corpus_alignment(self) -> None:
        """Chaque séquence doit avoir autant de tokens que de labels."""
        seqs, labels = _make_small_corpus(n=15, seed=1)
        for i, (seq, lbl) in enumerate(zip(seqs, labels)):
            assert len(seq) == len(lbl), (
                f"Séquence {i} : {len(seq)} tokens ≠ {len(lbl)} labels"
            )

    def test_corpus_bio_tags(self) -> None:
        """Tous les tags BIO doivent appartenir à GRAMMAR_TAGS."""
        _, labels = _make_small_corpus(n=20, seed=2)
        valid_tags = set(GRAMMAR_TAGS)
        for lbl_seq in labels:
            for tag in lbl_seq:
                assert tag in valid_tags, f"Tag invalide rencontré : '{tag}'"

    def test_corpus_reproducibility(self) -> None:
        """Même seed → même premier label de la première séquence."""
        _, labels_a = generate_synthetic_corpus(n_sequences=10, seed=42)
        _, labels_b = generate_synthetic_corpus(n_sequences=10, seed=42)
        assert labels_a[0][0] == labels_b[0][0], (
            "La reproductibilité est rompue : labels[0][0] diffèrent"
        )

    def test_corpus_different_seeds(self) -> None:
        """Seeds différentes doivent produire des résultats différents."""
        _, labels_a = generate_synthetic_corpus(n_sequences=20, seed=10)
        _, labels_b = generate_synthetic_corpus(n_sequences=20, seed=99)
        # Au moins un label doit différer sur les 20 premières séquences
        any_diff = any(
            la != lb
            for seq_a, seq_b in zip(labels_a, labels_b)
            for la, lb in zip(seq_a, seq_b)
        )
        assert any_diff, "Les deux seeds donnent exactement les mêmes labels"

    def test_token_length_in_range(self) -> None:
        """La longueur des séquences doit respecter min_len et max_len."""
        seqs, _ = generate_synthetic_corpus(n_sequences=30, min_len=3, max_len=8, seed=5)
        for seq in seqs:
            assert len(seq) >= 3, f"Séquence trop courte : {len(seq)}"
            assert len(seq) <= 8, f"Séquence trop longue : {len(seq)}"


# ── TestTokenFeatures ──────────────────────────────────────────────────────────


class TestTokenFeatures:
    def _make_seq_of_three(self) -> list[SignToken]:
        """Séquence de 3 tokens : SUJET, VERBE, OBJET."""
        return [
            _make_token("MOI", role=GrammarRole.SUBJECT),
            _make_token("MANGER", role=GrammarRole.VERB),
            _make_token("PAIN", role=GrammarRole.OBJECT),
        ]

    def test_feature_keys_present(self) -> None:
        """Les clés fondamentales doivent être présentes dans le dict de features."""
        tokens = self._make_seq_of_three()
        feats = _token_features(tokens, 1)  # token central
        for key in ("label", "confidence_high", "role_curr", "bigram_prev_curr"):
            assert key in feats, f"Clé manquante : '{key}'"

    def test_signing_space_features(self) -> None:
        """18 clés 'space_0'…'space_17' doivent être présentes."""
        ss = np.random.default_rng(0).random(18).astype(np.float32)
        tokens = [
            _make_token("MOI", signing_space=ss),
            _make_token("VOIR", signing_space=np.zeros(18, dtype=np.float32)),
        ]
        feats = _token_features(tokens, 0)
        for si in range(18):
            key = f"space_{si}"
            assert key in feats, f"Clé manquante : '{key}'"
            assert isinstance(feats[key], float), f"'{key}' doit être un float"

    def test_signing_space_zeros_when_absent(self) -> None:
        """Si signing_space est nul (zeros), toutes les dims doivent être 0.0."""
        tok = _make_token("TEST", signing_space=np.zeros(18, dtype=np.float32))
        feats = _token_features([tok], 0)
        for si in range(18):
            assert feats[f"space_{si}"] == 0.0

    def test_is_first_in_domain(self) -> None:
        """La feature 'is_first_in_domain' doit être présente et booléenne."""
        tokens = self._make_seq_of_three()
        feats = _token_features(tokens, 0)
        assert "is_first_in_domain" in feats, "Clé 'is_first_in_domain' manquante"
        assert isinstance(feats["is_first_in_domain"], bool), (
            "'is_first_in_domain' doit être un bool"
        )

    def test_is_first_in_domain_neutral_space(self) -> None:
        """Un token dans l'espace neutre (space≈0) doit avoir is_first_in_domain=True."""
        neutral_space = np.zeros(18, dtype=np.float32)  # norme < 0.15
        tok = _make_token(signing_space=neutral_space)
        feats = _token_features([tok], 0)
        assert feats["is_first_in_domain"] is True

    def test_bigram_prev_curr_bos_at_start(self) -> None:
        """Premier token → bigram_prev_curr doit commencer par '__BOS__'."""
        tokens = self._make_seq_of_three()
        feats = _token_features(tokens, 0)
        assert str(feats["bigram_prev_curr"]).startswith("__BOS__"), (
            "Premier token : bigram_prev_curr doit commencer par '__BOS__'"
        )

    def test_bigram_curr_next_eos_at_end(self) -> None:
        """Dernier token → bigram_curr_next doit se terminer par '__EOS__'."""
        tokens = self._make_seq_of_three()
        last_idx = len(tokens) - 1
        feats = _token_features(tokens, last_idx)
        assert str(feats["bigram_curr_next"]).endswith("__EOS__"), (
            "Dernier token : bigram_curr_next doit se terminer par '__EOS__'"
        )


# ── TestLSFBSequenceTaggerFit ──────────────────────────────────────────────────


class TestLSFBSequenceTaggerFit:
    def test_fit_returns_self(self) -> None:
        """fit() doit retourner l'instance du tagger (chaînage)."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=20, seed=7)
        tagger = LSFBSequenceTagger(max_iterations=10)
        result = tagger.fit(seqs, labels)
        assert result is tagger, "fit() doit retourner self"

    def test_predict_after_fit(self) -> None:
        """Après fit(), predict() doit retourner une liste de la bonne longueur."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=20, seed=8)
        tagger = LSFBSequenceTagger(max_iterations=10)
        tagger.fit(seqs, labels)
        # Prédire sur la première séquence
        preds = tagger.predict(seqs[0])
        assert len(preds) == len(seqs[0]), (
            f"Longueur prédite {len(preds)} ≠ longueur de la séquence {len(seqs[0])}"
        )

    def test_predict_uses_crf_after_fit(self) -> None:
        """Après fit(), tagger.crf doit être non nul si sklearn_crfsuite est dispo."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=20, seed=9)
        tagger = LSFBSequenceTagger(max_iterations=10)
        tagger.fit(seqs, labels)
        assert tagger.crf is not None, "Le modèle CRF doit être non None après fit()"

    def test_predict_fallback_without_crf(self) -> None:
        """Sans CRF chargé, predict() doit quand même retourner une liste valide."""
        seqs, _ = _make_small_corpus(n=5, seed=11)
        tagger = LSFBSequenceTagger()  # pas de fit → crf is None
        assert tagger.crf is None
        for seq in seqs:
            preds = tagger.predict(seq)
            assert isinstance(preds, list), "predict() doit retourner une liste"
            # Le repli sur les règles peut ajouter un token __QYN__ donc len >= len(seq)
            assert len(preds) >= len(seq), (
                "predict() (repli règles) doit retourner au moins autant de tags que de tokens"
            )

    def test_predict_empty_sequence(self) -> None:
        """predict([]) doit retourner une liste vide."""
        tagger = LSFBSequenceTagger()
        preds = tagger.predict([])
        assert preds == [], "predict([]) doit retourner []"

    def test_fit_raises_without_sklearn_crfsuite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fit() doit lever ImportError si sklearn_crfsuite n'est pas disponible."""
        import app.ml.grammar.lsfb_crf as crf_module
        monkeypatch.setattr(crf_module, "_CRF_AVAILABLE", False)
        seqs, labels = _make_small_corpus(n=5, seed=12)
        tagger = LSFBSequenceTagger()
        with pytest.raises(ImportError, match="sklearn-crfsuite"):
            tagger.fit(seqs, labels)


# ── TestLSFBSequenceTaggerEvaluate ─────────────────────────────────────────────


class TestLSFBSequenceTaggerEvaluate:
    def test_evaluate_raises_without_fit(self) -> None:
        """evaluate() doit lever RuntimeError si crf is None."""
        tagger = LSFBSequenceTagger()
        seqs, labels = _make_small_corpus(n=5, seed=13)
        with pytest.raises(RuntimeError, match="No trained CRF model"):
            tagger.evaluate(seqs, labels)

    def test_evaluate_returns_metrics(self) -> None:
        """evaluate() doit retourner un dict avec les 4 clés attendues."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=30, seed=14)
        tagger = LSFBSequenceTagger(max_iterations=20)
        tagger.fit(seqs, labels)
        metrics = tagger.evaluate(seqs, labels)
        for key in ("weighted_f1", "weighted_precision", "weighted_recall", "report_str"):
            assert key in metrics, f"Clé manquante dans les métriques : '{key}'"

    def test_evaluate_f1_range(self) -> None:
        """weighted_f1 doit être dans [0.0, 1.0]."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=30, seed=15)
        tagger = LSFBSequenceTagger(max_iterations=20)
        tagger.fit(seqs, labels)
        metrics = tagger.evaluate(seqs, labels)
        f1 = metrics["weighted_f1"]
        assert 0.0 <= f1 <= 1.0, f"weighted_f1 hors [0, 1] : {f1}"

    def test_evaluate_report_is_string(self) -> None:
        """report_str doit être une chaîne non vide."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=20, seed=16)
        tagger = LSFBSequenceTagger(max_iterations=10)
        tagger.fit(seqs, labels)
        metrics = tagger.evaluate(seqs, labels)
        assert isinstance(metrics["report_str"], str), "report_str doit être une str"
        assert len(metrics["report_str"]) > 0, "report_str ne doit pas être vide"

    def test_evaluate_precision_recall_range(self) -> None:
        """precision et recall doivent être dans [0.0, 1.0]."""
        pytest.importorskip("sklearn_crfsuite")
        seqs, labels = _make_small_corpus(n=20, seed=17)
        tagger = LSFBSequenceTagger(max_iterations=10)
        tagger.fit(seqs, labels)
        metrics = tagger.evaluate(seqs, labels)
        assert 0.0 <= metrics["weighted_precision"] <= 1.0
        assert 0.0 <= metrics["weighted_recall"] <= 1.0
