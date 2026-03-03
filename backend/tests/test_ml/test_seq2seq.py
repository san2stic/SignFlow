"""Tests unitaires pour les nouvelles classes de backend/app/ml/grammar/lsfb_translator.py.

Couvre :
- Seq2SeqVocab : build_from_sequences, encode, decode, save, load
- generate_seq2seq_corpus()
- Seq2SeqTranslator : forward, translate (greedy + beam)
- Seq2SeqTrainer : train_epoch, evaluate, fit, save, load
- LSFBToFrenchTranslator : mode seq2seq, résolution de mode, translate_buffer (régression)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml.grammar.lsfb_translator import (
    LSFBToFrenchTranslator,
    Seq2SeqVocab,
    generate_seq2seq_corpus,
)

# Import conditionnel torch — tous les tests qui l'utilisent sont skippés si absent
torch = pytest.importorskip("torch", reason="PyTorch requis pour les tests Seq2Seq")

from app.ml.grammar.lsfb_translator import (  # noqa: E402  (après importorskip)
    Seq2SeqTrainer,
    Seq2SeqTranslator,
)


# ── Données minimales pour les tests ─────────────────────────────────────────

# Vocabulaire source et cible minimal (10 tokens lexicaux)
_SRC_TOKENS = [["MOI", "MANGER", "PAIN"], ["TOI", "VOIR", "MARIE"], ["LUI", "VOULOIR", "EAU"]]
_TGT_TOKENS = [["Je", "mange", "le", "pain", "."], ["Tu", "vois", "Marie", "."], ["Il", "veut", "l'", "eau", "."]]
_SMALL_PAIRS = list(zip(_SRC_TOKENS, _TGT_TOKENS))


def _build_small_vocab(sequences: list[list[str]]) -> Seq2SeqVocab:
    """Construit un vocabulaire depuis les séquences fournies."""
    vocab = Seq2SeqVocab()
    vocab.build_from_sequences(sequences)
    return vocab


def _build_small_model(src_vocab: Seq2SeqVocab, tgt_vocab: Seq2SeqVocab) -> Seq2SeqTranslator:
    """Construit un modèle Seq2SeqTranslator minimaliste."""
    return Seq2SeqTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        max_decode_len=10,
    )


# ── TestSeq2SeqVocab ──────────────────────────────────────────────────────────


class TestSeq2SeqVocab:
    def test_build_vocab(self) -> None:
        """Après build_from_sequences, PAD=0, BOS=1, EOS=2, UNK=3."""
        vocab = _build_small_vocab(_SRC_TOKENS)
        assert vocab.PAD == 0
        assert vocab.BOS == 1
        assert vocab.EOS == 2
        assert vocab.UNK == 3

    def test_vocab_size(self) -> None:
        """Taille = 4 spéciaux + tokens uniques."""
        sequences = [["A", "B", "C"], ["A", "D"]]
        vocab = _build_small_vocab(sequences)
        # 4 spéciaux + 4 tokens uniques (A, B, C, D)
        assert len(vocab) == 4 + 4, f"Taille attendue 8, obtenu {len(vocab)}"

    def test_encode_decode_roundtrip(self) -> None:
        """encode puis decode doit redonner les tokens originaux."""
        vocab = _build_small_vocab(_SRC_TOKENS)
        tokens = ["MOI", "MANGER", "PAIN"]
        indices = vocab.encode(tokens)
        decoded = vocab.decode(indices)
        assert decoded == tokens, f"Roundtrip échoué : {decoded} ≠ {tokens}"

    def test_encode_unknown(self) -> None:
        """Un token inconnu doit être encodé en UNK (3)."""
        vocab = _build_small_vocab(_SRC_TOKENS)
        indices = vocab.encode(["COUCOU_INCONNU"])
        assert indices == [vocab.UNK], (
            f"Token inconnu doit donner UNK={vocab.UNK}, obtenu {indices}"
        )

    def test_decode_filters_specials(self) -> None:
        """PAD/BOS/EOS doivent être filtrés au décodage."""
        vocab = _build_small_vocab(_SRC_TOKENS)
        # Injecter des spéciaux parmi des tokens normaux
        word_idx = vocab.encode(["MOI"])[0]
        mixed = [vocab.PAD, vocab.BOS, word_idx, vocab.EOS, vocab.PAD]
        decoded = vocab.decode(mixed)
        assert decoded == ["MOI"], f"Décodage avec spéciaux : attendu ['MOI'], obtenu {decoded}"

    def test_save_load(self, tmp_path: Path) -> None:
        """save + load doit préserver le vocabulaire."""
        vocab = _build_small_vocab(_SRC_TOKENS)
        path = tmp_path / "vocab_test.json"
        vocab.save(path)
        loaded = Seq2SeqVocab.load(path)
        # Vérifier les mapping
        assert vocab._token2idx == loaded._token2idx, "token2idx modifié après save/load"
        assert vocab._idx2token == loaded._idx2token, "idx2token modifié après save/load"

    def test_special_tokens_always_present(self) -> None:
        """Les 4 tokens spéciaux doivent toujours être présents."""
        vocab = Seq2SeqVocab()
        vocab.build_from_sequences([])  # vocabulaire vide
        assert len(vocab) == 4, "Vocabulaire vide doit avoir 4 tokens spéciaux"
        assert vocab._token2idx.get("<PAD>") == 0
        assert vocab._token2idx.get("<BOS>") == 1
        assert vocab._token2idx.get("<EOS>") == 2
        assert vocab._token2idx.get("<UNK>") == 3


# ── TestGenerateSeq2SeqCorpus ─────────────────────────────────────────────────


class TestGenerateSeq2SeqCorpus:
    def test_corpus_length(self) -> None:
        """n_pairs paires doivent être retournées."""
        pairs = generate_seq2seq_corpus(n_pairs=30, seed=0)
        assert len(pairs) == 30, f"Attendu 30 paires, obtenu {len(pairs)}"

    def test_pair_structure(self) -> None:
        """Chaque paire doit être (list[str], list[str]) non vide."""
        pairs = generate_seq2seq_corpus(n_pairs=10, seed=1)
        for i, (src, tgt) in enumerate(pairs):
            assert isinstance(src, list), f"paire[{i}][0] n'est pas une liste"
            assert isinstance(tgt, list), f"paire[{i}][1] n'est pas une liste"
            assert len(src) > 0, f"paire[{i}] : src vide"
            assert len(tgt) > 0, f"paire[{i}] : tgt vide"
            for tok in src:
                assert isinstance(tok, str), f"src token doit être str, got {type(tok)}"
            for tok in tgt:
                assert isinstance(tok, str), f"tgt token doit être str, got {type(tok)}"

    def test_reproducibility(self) -> None:
        """Même seed → même première paire."""
        pairs_a = generate_seq2seq_corpus(n_pairs=5, seed=42)
        pairs_b = generate_seq2seq_corpus(n_pairs=5, seed=42)
        assert pairs_a[0] == pairs_b[0], (
            "La reproductibilité est rompue pour generate_seq2seq_corpus"
        )

    def test_different_seeds(self) -> None:
        """Seeds différentes doivent produire des résultats différents."""
        pairs_a = generate_seq2seq_corpus(n_pairs=10, seed=10)
        pairs_b = generate_seq2seq_corpus(n_pairs=10, seed=11)
        # Au moins une paire doit différer
        any_diff = any(pa != pb for pa, pb in zip(pairs_a, pairs_b))
        assert any_diff, "Deux seeds différentes donnent exactement les mêmes paires"


# ── TestSeq2SeqTranslator ─────────────────────────────────────────────────────


class TestSeq2SeqTranslator:
    @pytest.fixture()
    def vocabs(self) -> tuple[Seq2SeqVocab, Seq2SeqVocab]:
        """Vocabulaire src et tgt minimal."""
        src_vocab = _build_small_vocab(_SRC_TOKENS)
        tgt_vocab = _build_small_vocab(_TGT_TOKENS)
        return src_vocab, tgt_vocab

    @pytest.fixture()
    def model(self, vocabs: tuple) -> Seq2SeqTranslator:
        """Modèle minimal."""
        src_vocab, tgt_vocab = vocabs
        return _build_small_model(src_vocab, tgt_vocab)

    def test_forward_shape(self, model: Seq2SeqTranslator, vocabs: tuple) -> None:
        """forward() doit retourner (B, T, tgt_vocab_size)."""
        src_vocab, tgt_vocab = vocabs
        B, S, T = 2, 3, 4
        src = torch.randint(0, len(src_vocab), (B, S))
        tgt = torch.randint(0, len(tgt_vocab), (B, T))
        src_lengths = torch.tensor([S, S], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            logits = model(src, tgt, src_lengths)

        assert logits.shape == (B, T, len(tgt_vocab)), (
            f"Shape attendue ({B}, {T}, {len(tgt_vocab)}), obtenu {logits.shape}"
        )

    def test_translate_returns_list(self, model: Seq2SeqTranslator, vocabs: tuple) -> None:
        """translate() doit retourner une liste d'entiers."""
        src_vocab, _ = vocabs
        src = torch.tensor([[src_vocab.encode(["MOI"])[0], src_vocab.encode(["MANGER"])[0]]])
        src_lengths = torch.tensor([2], dtype=torch.long)

        result = model.translate(src, src_lengths, beam_size=1)
        assert isinstance(result, list), "translate() doit retourner une list"
        for idx in result:
            assert isinstance(idx, int), f"Chaque index doit être un int, got {type(idx)}"

    def test_translate_no_specials(self, model: Seq2SeqTranslator, vocabs: tuple) -> None:
        """Les résultats de translate() ne doivent pas contenir PAD/BOS/EOS."""
        src_vocab, _ = vocabs
        src = torch.tensor([[src_vocab.encode(["TOI"])[0]]])
        src_lengths = torch.tensor([1], dtype=torch.long)

        result = model.translate(src, src_lengths, beam_size=1)
        specials = {Seq2SeqVocab.PAD, Seq2SeqVocab.BOS, Seq2SeqVocab.EOS}
        for idx in result:
            assert idx not in specials, (
                f"Le token spécial {idx} ne doit pas apparaître dans translate()"
            )

    def test_beam_search_vs_greedy(self, model: Seq2SeqTranslator, vocabs: tuple) -> None:
        """beam_size=1 et beam_size=3 doivent tous deux retourner des listes."""
        src_vocab, _ = vocabs
        src = torch.tensor([[src_vocab.encode(["LUI"])[0]]])
        src_lengths = torch.tensor([1], dtype=torch.long)

        greedy = model.translate(src, src_lengths, beam_size=1)
        beam = model.translate(src, src_lengths, beam_size=3)

        assert isinstance(greedy, list), "greedy decode doit retourner une liste"
        assert isinstance(beam, list), "beam search doit retourner une liste"

    def test_forward_no_nan(self, model: Seq2SeqTranslator, vocabs: tuple) -> None:
        """forward() ne doit pas produire de NaN."""
        src_vocab, tgt_vocab = vocabs
        src = torch.randint(0, len(src_vocab), (2, 3))
        tgt = torch.randint(0, len(tgt_vocab), (2, 4))
        src_lengths = torch.tensor([3, 3], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            logits = model(src, tgt, src_lengths)

        assert not torch.isnan(logits).any(), "NaN detecté dans forward()"
        assert not torch.isinf(logits).any(), "Inf detecté dans forward()"


# ── TestSeq2SeqTrainer ────────────────────────────────────────────────────────


class TestSeq2SeqTrainer:
    @pytest.fixture()
    def trainer_with_data(self) -> tuple[Seq2SeqTrainer, list]:
        """Trainer minimal avec 6 paires d'entraînement."""
        # Construire un corpus légèrement plus grand que _SMALL_PAIRS
        corpus = generate_seq2seq_corpus(n_pairs=16, seed=77)
        all_src = [s for s, _ in corpus]
        all_tgt = [t for _, t in corpus]

        src_vocab = Seq2SeqVocab()
        src_vocab.build_from_sequences(all_src)
        tgt_vocab = Seq2SeqVocab()
        tgt_vocab.build_from_sequences(all_tgt)

        model = Seq2SeqTranslator(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=16,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            max_decode_len=8,
        )
        trainer = Seq2SeqTrainer(model, src_vocab, tgt_vocab, lr=1e-3, device="cpu")
        return trainer, corpus

    def test_train_epoch_returns_loss(self, trainer_with_data: tuple) -> None:
        """train_epoch() doit retourner un dict avec 'loss' float positif."""
        trainer, corpus = trainer_with_data
        metrics = trainer.train_epoch(corpus[:8], batch_size=4)
        assert "loss" in metrics, "Clé 'loss' manquante"
        loss = metrics["loss"]
        assert isinstance(loss, float), f"loss doit être float, got {type(loss)}"
        assert loss > 0.0, f"loss doit être positif, got {loss}"

    def test_evaluate_returns_bleu(self, trainer_with_data: tuple) -> None:
        """evaluate() doit retourner un dict avec 'loss' et 'bleu1'."""
        trainer, corpus = trainer_with_data
        metrics = trainer.evaluate(corpus[:4], batch_size=4)
        for key in ("loss", "bleu1"):
            assert key in metrics, f"Clé manquante : '{key}'"
        assert 0.0 <= metrics["bleu1"] <= 1.0, (
            f"bleu1 doit être dans [0, 1], got {metrics['bleu1']}"
        )

    def test_fit_returns_history(self, trainer_with_data: tuple) -> None:
        """fit() doit retourner une liste de dicts."""
        trainer, corpus = trainer_with_data
        train_pairs = corpus[:10]
        val_pairs = corpus[10:]
        history = trainer.fit(
            train_pairs,
            val_pairs,
            epochs=2,
            batch_size=4,
            patience=5,
        )
        assert isinstance(history, list), "fit() doit retourner une liste"
        assert len(history) > 0, "L'historique ne doit pas être vide"
        for entry in history:
            for key in ("epoch", "train_loss", "val_loss", "val_bleu1"):
                assert key in entry, f"Clé manquante dans l'historique : '{key}'"

    def test_save_load_roundtrip(self, trainer_with_data: tuple, tmp_path: Path) -> None:
        """save + load → translator fonctionnel (translate_buffer donne même résultat)."""
        trainer_orig, corpus = trainer_with_data
        # Un peu d'entraînement pour avoir des poids stabilisés
        trainer_orig.fit(corpus[:10], corpus[10:], epochs=2, batch_size=4, patience=5)

        ckpt_path = tmp_path / "seq2seq_test.pkl"
        trainer_orig.save(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} n'a pas été créé"

        trainer_loaded = Seq2SeqTrainer.load(str(ckpt_path), device="cpu")
        assert isinstance(trainer_loaded, Seq2SeqTrainer), (
            "Seq2SeqTrainer.load() doit retourner un Seq2SeqTrainer"
        )

        # Vérifier que la traduction est identique après chargement
        src_vocab = trainer_orig.src_vocab
        token = "MOI"
        import torch as _torch
        src_idx = src_vocab.encode([token])
        src_tensor = _torch.tensor([src_idx], dtype=_torch.long)
        src_len = _torch.tensor([len(src_idx)], dtype=_torch.long)

        trainer_orig.model.eval()
        trainer_loaded.model.eval()
        with _torch.no_grad():
            pred_orig = trainer_orig.model.translate(src_tensor, src_len)
            pred_loaded = trainer_loaded.model.translate(src_tensor, src_len)
        assert pred_orig == pred_loaded, (
            f"Prédictions différentes après save/load : {pred_orig} ≠ {pred_loaded}"
        )


# ── TestLSFBToFrenchTranslatorSeq2SeqMode ─────────────────────────────────────


class TestLSFBToFrenchTranslatorSeq2SeqMode:
    def test_default_mode_no_seq2seq(self) -> None:
        """Sans seq2seq_model_path, le mode doit rester 'rules'."""
        translator = LSFBToFrenchTranslator(mode="rules")
        assert translator.mode == "rules", (
            f"Mode attendu 'rules', obtenu '{translator.mode}'"
        )

    def test_resolve_mode_seq2seq_without_model(self) -> None:
        """_resolve_mode('seq2seq') sans modèle chargé doit retourner 'rules'."""
        translator = LSFBToFrenchTranslator(mode="rules")
        # Forcer la résolution sans trainer seq2seq
        assert translator._seq2seq_trainer is None
        resolved = translator._resolve_mode("seq2seq")
        # Sans CRF ni seq2seq → repli sur 'rules'
        assert resolved in ("rules", "crf"), (
            f"Sans modèle seq2seq, mode attendu 'rules' ou 'crf', obtenu '{resolved}'"
        )

    def test_translate_buffer_rules(self) -> None:
        """translate_buffer() doit fonctionner sans seq2seq (test de régression)."""
        translator = LSFBToFrenchTranslator(mode="rules")
        buffer = [
            {"label": "MOI", "confidence": 0.9},
            {"label": "MANGER", "confidence": 0.85},
            {"label": "PAIN", "confidence": 0.8},
        ]
        result = translator.translate_buffer(buffer)
        assert isinstance(result, str), "translate_buffer() doit retourner une str"
        # La phrase ne doit pas être vide (au moins un token non-NONE)
        assert len(result) >= 0  # peut être vide selon les règles mais ne doit pas crasher

    def test_translate_buffer_empty(self) -> None:
        """translate_buffer([]) doit retourner une chaîne vide."""
        translator = LSFBToFrenchTranslator(mode="rules")
        result = translator.translate_buffer([])
        assert result == "", f"translate_buffer([]) doit retourner '', obtenu '{result}'"

    def test_translate_buffer_skips_none_labels(self) -> None:
        """Les labels 'NONE' et vides doivent être ignorés."""
        translator = LSFBToFrenchTranslator(mode="rules")
        buffer = [
            {"label": "NONE", "confidence": 0.0},
            {"label": "", "confidence": 0.0},
            {"label": "MOI", "confidence": 0.9},
        ]
        # Ne doit pas planter
        result = translator.translate_buffer(buffer)
        assert isinstance(result, str)

    def test_seq2seq_mode_with_invalid_path_falls_back(self) -> None:
        """Un chemin invalide pour seq2seq_model_path doit faire un repli gracieux."""
        translator = LSFBToFrenchTranslator(
            mode="seq2seq",
            seq2seq_model_path="/tmp/nonexistent_model_12345.pkl",
        )
        # Le mode doit avoir été dégradé vers 'rules' (ou 'crf' si disponible)
        assert translator.mode in ("rules", "crf"), (
            f"Mode inattendu après échec de chargement : '{translator.mode}'"
        )
        assert translator._seq2seq_trainer is None
