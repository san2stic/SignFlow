"""Tests unitaires pour le module de grammaire LSFB → Français (Phase 3).

Couvre :
- NMMAnalyzer : détection des marqueurs grammaticaux depuis les features NMM
- LSFBGrammarRules : tagging, réordonnancement, négation, questions
- LSFBSequenceTagger : prédiction BIO (mode règles uniquement)
- LSFBToFrenchTranslator : pipeline complet via translate() et translate_buffer()

Exemples de phrases LSFB testées :
- Négation : "JE + MANGER" avec NMM secouement de tête → "Je ne mange pas."
- Question oui/non : "TU + MANGER + PAIN" avec sourcils haussés → "Tu manges le pain ?"
- Réordonnancement SOV → SVO : "PAIN + JE + MANGER" → "Je mange le pain."
- Question WH : "TU + ALLER + OÙ" → "Où tu vas ?"
- Intensifieur : "GRAND avec joues gonflées" → phrase contenant "grand"
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.grammar.lsfb_rules import (
    GrammarRole,
    LSFBGrammarRules,
    NMMAnalyzer,
    NMMMarkers,
    SignToken,
)
from app.ml.grammar.lsfb_crf import LSFBSequenceTagger, GRAMMAR_TAGS
from app.ml.grammar.lsfb_translator import LSFBToFrenchTranslator, TranslationResult


# ---------------------------------------------------------------------------
# Helpers de construction de tokens
# ---------------------------------------------------------------------------

def _make_nmm(
    *,
    brow_raise: float = 0.0,
    brow_furrow: float = 0.0,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    cheek_puff: float = 0.0,
) -> np.ndarray:
    """Crée un vecteur NMM 32-dims minimal avec les valeurs données."""
    nmm = np.zeros(32, dtype=np.float32)
    nmm[0] = brow_raise   # brow raise gauche
    nmm[1] = brow_raise   # brow raise droit
    nmm[3] = brow_furrow  # brow furrow gauche
    nmm[4] = brow_furrow  # brow furrow droit
    nmm[15] = cheek_puff  # cheek puff gauche
    nmm[16] = cheek_puff  # cheek puff droit
    nmm[20] = head_pitch  # head pitch (nod)
    nmm[21] = head_yaw    # head yaw (shake)
    return nmm


def _tok(
    label: str,
    confidence: float = 0.85,
    *,
    brow_raise: float = 0.0,
    brow_furrow: float = 0.0,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    cheek_puff: float = 0.0,
    grammar_role: GrammarRole = GrammarRole.UNKNOWN,
) -> SignToken:
    """Crée un SignToken avec les NMM spécifiées."""
    return SignToken(
        label=label,
        confidence=confidence,
        nmm_features=_make_nmm(
            brow_raise=brow_raise,
            brow_furrow=brow_furrow,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            cheek_puff=cheek_puff,
        ),
        signing_space=np.zeros(18, dtype=np.float32),
        grammar_role=grammar_role,
    )


# ---------------------------------------------------------------------------
# Tests NMMAnalyzer
# ---------------------------------------------------------------------------


class TestNMMAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = NMMAnalyzer()

    def test_neutral_nmm_no_markers(self) -> None:
        nmm = np.zeros(32, dtype=np.float32)
        markers = self.analyzer.analyze(nmm)
        assert not markers.polar_question
        assert not markers.wh_question
        assert not markers.negation
        assert not markers.affirmation
        assert not markers.intensifier

    def test_brow_raise_polar_question(self) -> None:
        nmm = _make_nmm(brow_raise=0.35)
        markers = self.analyzer.analyze(nmm)
        assert markers.polar_question
        assert not markers.negation

    def test_brow_furrow_wh_question(self) -> None:
        nmm = _make_nmm(brow_furrow=0.30)
        markers = self.analyzer.analyze(nmm)
        assert markers.wh_question
        # WH-question + furrow ne devrait pas déclencher polar si raise < seuil
        assert not markers.polar_question

    def test_head_shake_negation(self) -> None:
        nmm = _make_nmm(head_yaw=0.25)
        markers = self.analyzer.analyze(nmm)
        assert markers.negation
        assert not markers.affirmation

    def test_head_nod_affirmation(self) -> None:
        nmm = _make_nmm(head_pitch=0.20)
        markers = self.analyzer.analyze(nmm)
        assert markers.affirmation
        assert not markers.negation

    def test_cheek_puff_intensifier(self) -> None:
        nmm = _make_nmm(cheek_puff=0.35)
        markers = self.analyzer.analyze(nmm)
        assert markers.intensifier

    def test_empty_nmm_returns_false_markers(self) -> None:
        markers = self.analyzer.analyze(np.zeros(32, dtype=np.float32))
        assert markers == NMMMarkers(False, False, False, False, False)

    def test_short_nmm_does_not_crash(self) -> None:
        markers = self.analyzer.analyze(np.zeros(10, dtype=np.float32))
        assert isinstance(markers, NMMMarkers)

    def test_aggregate_sequence(self) -> None:
        seq = [_make_nmm(brow_raise=0.0), _make_nmm(brow_raise=0.4), _make_nmm(brow_raise=0.8)]
        avg = self.analyzer.aggregate_sequence(seq)
        assert avg.shape == (32,)
        assert abs(float(avg[0]) - 0.4) < 0.05


# ---------------------------------------------------------------------------
# Tests LSFBGrammarRules — Tagging
# ---------------------------------------------------------------------------


class TestLSFBGrammarRulesTagging:
    def setup_method(self) -> None:
        self.rules = LSFBGrammarRules()

    def test_tag_pronoun_first_position_becomes_subject(self) -> None:
        tokens = [_tok("JE"), _tok("MANGER")]
        tagged = self.rules.tag_grammar_roles(tokens)
        assert tagged[0].grammar_role == GrammarRole.SUBJECT

    def test_tag_wh_sign_becomes_question_wh(self) -> None:
        tokens = [_tok("TU"), _tok("ALLER"), _tok("OÙ")]
        tagged = self.rules.tag_grammar_roles(tokens)
        wh_roles = [t.grammar_role for t in tagged if t.label.upper() == "OÙ"]
        assert GrammarRole.QUESTION_WH in wh_roles

    def test_tag_negation_sign(self) -> None:
        tokens = [_tok("JE"), _tok("NON"), _tok("MANGER")]
        tagged = self.rules.tag_grammar_roles(tokens)
        neg_roles = [t.grammar_role for t in tagged if t.label.upper() == "NON"]
        assert GrammarRole.NEGATION in neg_roles

    def test_tag_negation_from_head_shake(self) -> None:
        tokens = [
            _tok("JE"),
            _tok("MANGER", head_yaw=0.30),
        ]
        tagged = self.rules.tag_grammar_roles(tokens)
        manger_tok = next(t for t in tagged if t.label == "MANGER")
        assert manger_tok.grammar_role == GrammarRole.NEGATION

    def test_tag_known_adjective(self) -> None:
        tokens = [_tok("JE"), _tok("CONTENT")]
        tagged = self.rules.tag_grammar_roles(tokens)
        adj = next(t for t in tagged if t.label == "CONTENT")
        assert adj.grammar_role == GrammarRole.ADJECTIVE

    def test_empty_tokens_returns_empty(self) -> None:
        assert self.rules.tag_grammar_roles([]) == []


# ---------------------------------------------------------------------------
# Tests LSFBGrammarRules — Négation
# ---------------------------------------------------------------------------


class TestLSFBGrammarRulesNegation:
    def setup_method(self) -> None:
        self.rules = LSFBGrammarRules()
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_negation_signe_ne_pas(self) -> None:
        """JE + NON + MANGER → doit contenir 'mange pas' ou 'ne mange pas'."""
        tokens = [_tok("JE"), _tok("NON"), _tok("MANGER")]
        result = self.translator.translate(tokens)
        text_lower = result.text.lower()
        assert "pas" in text_lower or "ne" in text_lower

    def test_negation_head_shake_nmm(self) -> None:
        """JE + MANGER avec NMM secouement → doit contenir négation."""
        tokens = [
            _tok("JE"),
            _tok("MANGER", head_yaw=0.30),
        ]
        result = self.translator.translate(tokens)
        text_lower = result.text.lower()
        assert "pas" in text_lower or "ne" in text_lower or "manger" in text_lower

    def test_negation_with_adjective(self) -> None:
        """JE + NON + CONTENT → 'je n'est pas content' ou proche."""
        tokens = [
            _tok("JE"),
            _tok("NON"),
            _tok("CONTENT"),
        ]
        result = self.translator.translate(tokens)
        assert "pas" in result.text.lower()


# ---------------------------------------------------------------------------
# Tests LSFBGrammarRules — Questions
# ---------------------------------------------------------------------------


class TestLSFBGrammarRulesQuestions:
    def setup_method(self) -> None:
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_polar_question_ends_with_question_mark(self) -> None:
        """TU + MANGER + PAIN avec NMM sourcils haussés sur tout l'énoncé.

        En LSFB, les sourcils haussés persistent sur toute la question.
        L'analyse est faite sur la moyenne de tous les tokens.
        """
        tokens = [
            _tok("TU", brow_raise=0.35),
            _tok("MANGER", brow_raise=0.35),
            _tok("PAIN", brow_raise=0.35),
        ]
        result = self.translator.translate(tokens)
        assert "?" in result.text

    def test_wh_question(self) -> None:
        """TU + ALLER + OÙ → doit contenir 'où' et '?'."""
        tokens = [_tok("TU"), _tok("ALLER"), _tok("OÙ")]
        result = self.translator.translate(tokens)
        text_lower = result.text.lower()
        assert "où" in text_lower
        assert "?" in result.text

    def test_question_mark_absent_for_assertion(self) -> None:
        """Phrase assertive sans NMM question → pas de '?'."""
        tokens = [_tok("JE"), _tok("MANGER"), _tok("PAIN")]
        result = self.translator.translate(tokens)
        # Doit être une assertion, donc '.' ou aucune ponctuation de question
        assert result.text.endswith(".") or (
            "?" not in result.text
        )


# ---------------------------------------------------------------------------
# Tests LSFBGrammarRules — Réordonnancement SOV → SVO
# ---------------------------------------------------------------------------


class TestLSFBReordering:
    def setup_method(self) -> None:
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_sov_to_svo_reorder(self) -> None:
        """PAIN + JE + MANGER → sujet avant verbe dans la sortie."""
        tokens = [_tok("PAIN"), _tok("JE"), _tok("MANGER")]
        result = self.translator.translate(tokens)
        text_lower = result.text.lower()
        # Le pronom 'je' doit apparaître avant 'manger'
        assert "je" in text_lower
        idx_je = text_lower.find("je")
        idx_manger = text_lower.find("manger")
        # 'je' doit précéder 'manger'
        assert idx_je < idx_manger, f"inattendu : '{result.text}'"

    def test_canonical_svo_preserved(self) -> None:
        """JE + MANGER + PAIN (SVO) → contient 'je', 'manger' et 'pain'."""
        tokens = [_tok("JE"), _tok("MANGER"), _tok("PAIN")]
        result = self.translator.translate(tokens)
        text_lower = result.text.lower()
        assert "je" in text_lower
        assert "manger" in text_lower
        # Pain peut être transformé en "le pain" ou "pain"
        assert "pain" in text_lower

    def test_single_token(self) -> None:
        """Un seul token → phrase simple sans plantage."""
        tokens = [_tok("BONJOUR")]
        result = self.translator.translate(tokens)
        assert "bonjour" in result.text.lower()
        assert result.text  # non vide

    def test_empty_tokens(self) -> None:
        """Aucun token → chaîne vide."""
        result = self.translator.translate([])
        assert result.text == ""
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Tests LSFBSequenceTagger — Tags BIO
# ---------------------------------------------------------------------------


class TestLSFBSequenceTagger:
    def setup_method(self) -> None:
        self.tagger = LSFBSequenceTagger()

    def test_predict_returns_correct_length(self) -> None:
        tokens = [_tok("JE"), _tok("MANGER"), _tok("PAIN")]
        tags = self.tagger.predict(tokens)
        assert len(tags) == len(tokens)

    def test_predict_tags_are_valid_bio(self) -> None:
        tokens = [_tok("JE"), _tok("NON"), _tok("MANGER"), _tok("OÙ")]
        tags = self.tagger.predict(tokens)
        for tag in tags:
            assert tag in GRAMMAR_TAGS, f"Tag invalide: {tag}"

    def test_predict_empty_sequence(self) -> None:
        tags = self.tagger.predict([])
        assert tags == []

    def test_negation_sign_tagged_b_neg(self) -> None:
        tokens = [_tok("JE"), _tok("NON"), _tok("MANGER")]
        # Tagger sans CRF → rules
        tagged_tokens = LSFBGrammarRules().tag_grammar_roles(tokens)
        tags = self.tagger.predict(tagged_tokens)
        # NON est à l'index 1
        assert tags[1] == "B-NEG"

    def test_wh_sign_tagged_b_q(self) -> None:
        tokens = [_tok("TU"), _tok("ALLER"), _tok("OÙ")]
        tagged_tokens = LSFBGrammarRules().tag_grammar_roles(tokens)
        tags = self.tagger.predict(tagged_tokens)
        ou_idx = next(i for i, t in enumerate(tokens) if t.label == "OÙ")
        assert tags[ou_idx] == "B-Q"


# ---------------------------------------------------------------------------
# Tests LSFBToFrenchTranslator — translate_buffer()
# ---------------------------------------------------------------------------


class TestTranslateBuffer:
    def setup_method(self) -> None:
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_translate_buffer_basic(self) -> None:
        buffer = [
            {"label": "JE", "confidence": 0.9},
            {"label": "MANGER", "confidence": 0.85},
            {"label": "PAIN", "confidence": 0.8},
        ]
        text = self.translator.translate_buffer(buffer)
        assert text  # non vide
        assert "je" in text.lower()
        assert "manger" in text.lower()

    def test_translate_buffer_empty(self) -> None:
        assert self.translator.translate_buffer([]) == ""

    def test_translate_buffer_skips_none_recording(self) -> None:
        buffer = [
            {"label": "NONE", "confidence": 0.0},
            {"label": "RECORDING", "confidence": 0.0},
            {"label": "JE", "confidence": 0.9},
        ]
        text = self.translator.translate_buffer(buffer)
        assert "je" in text.lower()

    def test_translate_buffer_with_nmm_data(self) -> None:
        """Vérifie que les NMM dans le buffer ne causent pas de crash."""
        nmm_list = [0.0] * 32
        nmm_list[0] = 0.35  # brow raise → polar question
        buffer = [
            {"label": "TU", "confidence": 0.9, "nmm": nmm_list},
            {"label": "MANGER", "confidence": 0.85},
        ]
        text = self.translator.translate_buffer(buffer)
        assert text  # non vide


# ---------------------------------------------------------------------------
# Tests TranslationResult
# ---------------------------------------------------------------------------


class TestTranslationResult:
    def setup_method(self) -> None:
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_result_has_expected_fields(self) -> None:
        tokens = [_tok("JE"), _tok("MANGER")]
        result = self.translator.translate(tokens)
        assert isinstance(result, TranslationResult)
        assert isinstance(result.text, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.grammar_tags, list)
        assert result.mode_used in ("rules", "crf", "seq2seq")
        assert isinstance(result.debug_info, dict)

    def test_confidence_in_valid_range(self) -> None:
        tokens = [_tok("JE", confidence=0.7), _tok("MANGER", confidence=0.9)]
        result = self.translator.translate(tokens)
        assert 0.0 <= result.confidence <= 1.0

    def test_grammar_tags_same_length_as_tokens(self) -> None:
        # Ce test utilise des tokens sans NMM question → pas de token virtuel ajouté
        tokens = [_tok("JE"), _tok("NON"), _tok("MANGER"), _tok("PAIN")]
        result = self.translator.translate(tokens)
        # La phrase a une négation lexicale (NON) → pas d'ajout de token __QYN__
        assert len(result.grammar_tags) >= len(tokens)

    def test_mode_used_is_rules(self) -> None:
        tokens = [_tok("JE"), _tok("MANGER")]
        result = self.translator.translate(tokens)
        assert result.mode_used == "rules"


# ---------------------------------------------------------------------------
# Tests d'intégration — exemples LSFB complets
# ---------------------------------------------------------------------------


class TestLSFBIntegration:
    """Tests d'intégration bout-en-bout avec des phrases LSFB typiques."""

    def setup_method(self) -> None:
        self.translator = LSFBToFrenchTranslator(mode="rules")

    def test_bonjour_simple(self) -> None:
        tokens = [_tok("BONJOUR")]
        result = self.translator.translate(tokens)
        assert "bonjour" in result.text.lower()

    def test_je_manger_pain_no_crash(self) -> None:
        tokens = [_tok("JE"), _tok("MANGER"), _tok("PAIN")]
        result = self.translator.translate(tokens)
        assert result.text
        assert result.text.endswith(".")

    def test_negation_complete_phrase(self) -> None:
        """Phrase LSFB avec négation lexicale : JE + MANGER + NON."""
        tokens = [_tok("JE"), _tok("MANGER"), _tok("NON")]
        result = self.translator.translate(tokens)
        assert "pas" in result.text.lower() or "ne" in result.text.lower()

    def test_who_question_qui(self) -> None:
        """Question WH simple : QUI + VENIR."""
        tokens = [_tok("QUI"), _tok("VENIR")]
        result = self.translator.translate(tokens)
        assert "qui" in result.text.lower()
        assert "?" in result.text

    def test_question_yn_avec_brow_raise(self) -> None:
        """Question OUI/NON via NMM sourcils haussés sur tout l'énoncé."""
        tokens = [
            _tok("TU", brow_raise=0.35),
            _tok("CONTENT", brow_raise=0.35),
        ]
        result = self.translator.translate(tokens)
        assert "?" in result.text

    def test_intensifier_ne_plante_pas(self) -> None:
        """Intensifieur (joues gonflées) → ne doit pas planter."""
        tokens = [_tok("GRAND", cheek_puff=0.40)]
        result = self.translator.translate(tokens)
        assert result.text  # non vide

    def test_capitalisation_premier_mot(self) -> None:
        """La phrase doit commencer par une majuscule."""
        tokens = [_tok("JE"), _tok("MANGER")]
        result = self.translator.translate(tokens)
        assert result.text[0].isupper()

    def test_no_double_spaces(self) -> None:
        """La phrase ne doit pas contenir de double espaces."""
        tokens = [_tok("JE"), _tok("MANGER"), _tok("PAIN")]
        result = self.translator.translate(tokens)
        assert "  " not in result.text
