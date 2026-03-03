"""Tests for ConversationContext — historique conversationnel LSFB.

Covers:
  - add_turn(): insertion, dé-duplication temporelle, compteur
  - is_new_turn(): silence court vs silence long
  - resolve_anaphora(): résolution IL/ELLE depuis sujets et référents spatiaux
  - update_spatial_referent() et get_context_summary()
  - clear(): réinitialisation complète
  - max_history respecté (deque maxlen)
  - detect_speech_pause(): classification des pauses
  - touch_sign_time()
"""

from __future__ import annotations

import time
from collections import deque

import pytest

from app.ml.conversation_context import ConversationContext, ConversationTurn, SpatialReferent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ConversationContext:
    """ConversationContext standard pour les tests."""
    return ConversationContext(
        max_history=10,
        turn_gap_seconds=3.0,
        resolution_window=5,
        dedup_window_seconds=2.0,
    )


def _make_turn(ctx: ConversationContext, text: str, signs: list[str] | None = None) -> ConversationTurn:
    """Helper: ajoute un tour avec des valeurs par défaut."""
    return ctx.add_turn(
        text=text,
        raw_signs=signs or ["SIGNE_A"],
        grammar_tags=["B-SIGN"],
        confidence=0.85,
        speaker="signer",
    )


# ---------------------------------------------------------------------------
# Tests add_turn
# ---------------------------------------------------------------------------


def test_add_turn_basic_insertion(ctx: ConversationContext) -> None:
    """add_turn doit insérer un tour dans l'historique."""
    turn = _make_turn(ctx, "Bonjour le monde")
    assert len(ctx.history) == 1
    assert ctx.history[-1].text == "Bonjour le monde"
    assert turn.id == 1
    assert turn.speaker == "signer"


def test_add_turn_increments_counter(ctx: ConversationContext) -> None:
    """Chaque nouveau tour doit avoir un id incrémental unique."""
    t1 = _make_turn(ctx, "Phrase un")
    t2 = _make_turn(ctx, "Phrase deux")
    t3 = _make_turn(ctx, "Phrase trois")
    assert t1.id == 1
    assert t2.id == 2
    assert t3.id == 3
    assert len(ctx.history) == 3


def test_add_turn_stores_raw_signs(ctx: ConversationContext) -> None:
    """Les raw_signs doivent être copiés dans le tour."""
    signs = ["JE", "ALLER", "MAISON"]
    turn = ctx.add_turn(
        text="Je vais à la maison",
        raw_signs=signs,
        grammar_tags=["B-S", "B-V", "B-N"],
        confidence=0.9,
    )
    assert turn.raw_signs == signs
    # Vérifier que c'est une copie et pas une référence
    signs.append("EXTRA")
    assert len(turn.raw_signs) == 3


def test_add_turn_deduplication_same_text_same_window(ctx: ConversationContext) -> None:
    """Même texte dans la fenêtre de dé-duplication → retourne le tour existant."""
    t1 = _make_turn(ctx, "Bonjour")
    t2 = _make_turn(ctx, "Bonjour")  # même texte, immédiatement après
    # Doit retourner le premier tour, pas créer un second
    assert len(ctx.history) == 1
    assert t1.id == t2.id


def test_add_turn_deduplication_case_insensitive(ctx: ConversationContext) -> None:
    """La dé-duplication doit être insensible à la casse."""
    _make_turn(ctx, "bonjour")
    _make_turn(ctx, "BONJOUR")
    assert len(ctx.history) == 1


def test_add_turn_no_deduplication_different_text(ctx: ConversationContext) -> None:
    """Textes différents dans la fenêtre → deux tours distincts."""
    _make_turn(ctx, "Bonjour")
    _make_turn(ctx, "Au revoir")
    assert len(ctx.history) == 2


def test_add_turn_updates_last_sign_time(ctx: ConversationContext) -> None:
    """add_turn doit mettre à jour last_sign_time."""
    before = time.time()
    _make_turn(ctx, "Test")
    assert ctx.last_sign_time >= before


def test_add_turn_speaker_field(ctx: ConversationContext) -> None:
    """Le champ speaker doit être transmis correctement."""
    turn = ctx.add_turn(
        text="Message système",
        raw_signs=[],
        grammar_tags=[],
        confidence=1.0,
        speaker="system",
    )
    assert turn.speaker == "system"


# ---------------------------------------------------------------------------
# Tests is_new_turn
# ---------------------------------------------------------------------------


def test_is_new_turn_returns_false_without_any_sign(ctx: ConversationContext) -> None:
    """Sans aucun signe reçu (last_sign_time == 0), is_new_turn doit retourner False."""
    assert ctx.last_sign_time == 0.0
    assert ctx.is_new_turn() is False


def test_is_new_turn_returns_false_after_recent_sign(ctx: ConversationContext) -> None:
    """Après un signe récent (< 3 s), is_new_turn doit retourner False."""
    ctx.touch_sign_time()
    assert ctx.is_new_turn() is False


def test_is_new_turn_returns_true_after_long_silence(ctx: ConversationContext) -> None:
    """Après une pause longue simulée, is_new_turn doit retourner True."""
    # Simuler un signe il y a 10 secondes
    ctx.last_sign_time = time.time() - 10.0
    assert ctx.is_new_turn() is True


def test_is_new_turn_boundary_exceeds_gap(ctx: ConversationContext) -> None:
    """Juste au-delà de turn_gap_seconds, is_new_turn doit retourner True."""
    ctx.last_sign_time = time.time() - ctx.turn_gap_seconds - 0.001
    assert ctx.is_new_turn() is True


# ---------------------------------------------------------------------------
# Tests detect_speech_pause
# ---------------------------------------------------------------------------


def test_detect_speech_pause_turn_end(ctx: ConversationContext) -> None:
    """Pause >= turn_gap_seconds → 'turn_end'."""
    result = ctx.detect_speech_pause(ctx.turn_gap_seconds + 0.1)
    assert result == "turn_end"


def test_detect_speech_pause_phrase_break(ctx: ConversationContext) -> None:
    """Pause entre 0.5 s et turn_gap_seconds → 'phrase_break'."""
    result = ctx.detect_speech_pause(1.0)
    assert result == "phrase_break"


def test_detect_speech_pause_none(ctx: ConversationContext) -> None:
    """Pause < 0.5 s → 'none'."""
    result = ctx.detect_speech_pause(0.3)
    assert result == "none"


def test_detect_speech_pause_exact_gap_seconds(ctx: ConversationContext) -> None:
    """Pause exactement égale à turn_gap_seconds → 'turn_end'."""
    result = ctx.detect_speech_pause(ctx.turn_gap_seconds)
    assert result == "turn_end"


# ---------------------------------------------------------------------------
# Tests resolve_anaphora
# ---------------------------------------------------------------------------


def test_resolve_anaphora_no_pronouns(ctx: ConversationContext) -> None:
    """Phrase sans pronoms → retournée inchangée."""
    text = "Marie mange une pomme"
    assert ctx.resolve_anaphora(text) == text


def test_resolve_anaphora_empty_string(ctx: ConversationContext) -> None:
    """Chaîne vide → retournée telle quelle."""
    assert ctx.resolve_anaphora("") == ""


def test_resolve_anaphora_via_spatial_referent_masculine(ctx: ConversationContext) -> None:
    """IL doit être résolu via un référent spatial masculin (locus A = Pierre)."""
    ctx.update_spatial_referent("A", "Pierre", gender="masc")
    resolved = ctx.resolve_anaphora("IL travaille")
    assert "Pierre" in resolved
    assert "IL" not in resolved


def test_resolve_anaphora_via_spatial_referent_feminine(ctx: ConversationContext) -> None:
    """ELLE doit être résolu via un référent spatial féminin (locus B = Marie)."""
    ctx.update_spatial_referent("B", "Marie", gender="fem")
    resolved = ctx.resolve_anaphora("ELLE est là")
    assert "Marie" in resolved
    assert "ELLE" not in resolved


def test_resolve_anaphora_no_resolution_without_context(ctx: ConversationContext) -> None:
    """Sans contexte, le pronom doit rester intact."""
    resolved = ctx.resolve_anaphora("IL mange")
    # Sans sujets ni référents, "IL" reste inchangé
    assert "il" in resolved.lower()


def test_resolve_anaphora_lui_via_spatial_referent_masculine(ctx: ConversationContext) -> None:
    """LUI doit être résolu via un référent spatial masculin."""
    ctx.update_spatial_referent("A", "Jean", gender="masc")
    resolved = ctx.resolve_anaphora("Je lui parle")
    assert "Jean" in resolved


# ---------------------------------------------------------------------------
# Tests update_spatial_referent
# ---------------------------------------------------------------------------


def test_update_spatial_referent_stores_entity(ctx: ConversationContext) -> None:
    """update_spatial_referent doit stocker le référent dans spatial_referents."""
    ctx.update_spatial_referent("A", "Jean", gender="masc")
    assert "A" in ctx.spatial_referents
    ref = ctx.spatial_referents["A"]
    assert ref.entity == "Jean"
    assert ref.gender == "masc"
    assert ref.locus == "A"


def test_update_spatial_referent_normalizes_locus_to_uppercase(ctx: ConversationContext) -> None:
    """Le locus doit être normalisé en majuscule."""
    ctx.update_spatial_referent("b", "Sophie", gender="fem")
    assert "B" in ctx.spatial_referents
    assert ctx.spatial_referents["B"].entity == "Sophie"


def test_update_spatial_referent_overwrite(ctx: ConversationContext) -> None:
    """Un second appel doit écraser le référent précédent."""
    ctx.update_spatial_referent("A", "Jean", gender="masc")
    ctx.update_spatial_referent("A", "Paul", gender="masc")
    assert ctx.spatial_referents["A"].entity == "Paul"


def test_update_spatial_referent_introduced_at_set(ctx: ConversationContext) -> None:
    """introduced_at doit être un timestamp positif."""
    before = time.time()
    ctx.update_spatial_referent("C", "La voiture", gender="neut")
    assert ctx.spatial_referents["C"].introduced_at >= before


def test_spatial_referent_to_dict(ctx: ConversationContext) -> None:
    """SpatialReferent.to_dict() doit produire les clés attendues."""
    ctx.update_spatial_referent("A", "Pierre", gender="masc")
    d = ctx.spatial_referents["A"].to_dict()
    assert set(d.keys()) == {"locus", "entity", "introduced_at", "gender"}
    assert d["locus"] == "A"
    assert d["entity"] == "Pierre"
    assert d["gender"] == "masc"


# ---------------------------------------------------------------------------
# Tests get_context_summary
# ---------------------------------------------------------------------------


def test_get_context_summary_empty(ctx: ConversationContext) -> None:
    """get_context_summary sur un contexte vide doit retourner les valeurs par défaut."""
    summary = ctx.get_context_summary()
    assert summary["turn_count"] == 0
    assert summary["last_turns"] == []
    assert summary["spatial_referents"] == {}
    assert summary["current_subjects"] == []
    assert "session_age_seconds" in summary


def test_get_context_summary_with_data(ctx: ConversationContext) -> None:
    """get_context_summary doit refléter l'état réel du contexte."""
    _make_turn(ctx, "Première phrase", signs=["PREMIER"])
    _make_turn(ctx, "Deuxième phrase", signs=["DEUXIEME"])
    ctx.update_spatial_referent("A", "Marie", gender="fem")

    summary = ctx.get_context_summary()
    assert summary["turn_count"] == 2
    assert len(summary["last_turns"]) == 2
    assert summary["last_turns"][-1]["text"] == "Deuxième phrase"
    assert "A" in summary["spatial_referents"]
    assert summary["spatial_referents"]["A"] == "Marie"


def test_get_context_summary_max_three_recent_turns(ctx: ConversationContext) -> None:
    """get_context_summary ne doit retourner que les 3 derniers tours au maximum."""
    for i in range(5):
        _make_turn(ctx, f"Phrase {i}", signs=[f"SIGNE_{i}"])
    summary = ctx.get_context_summary()
    assert summary["turn_count"] == 5
    assert len(summary["last_turns"]) == 3
    assert summary["last_turns"][-1]["text"] == "Phrase 4"


# ---------------------------------------------------------------------------
# Tests clear
# ---------------------------------------------------------------------------


def test_clear_resets_history(ctx: ConversationContext) -> None:
    """clear() doit vider l'historique des tours."""
    for i in range(3):
        _make_turn(ctx, f"Turn {i}")
    ctx.clear()
    assert len(ctx.history) == 0


def test_clear_resets_spatial_referents(ctx: ConversationContext) -> None:
    """clear() doit vider les référents spatiaux."""
    ctx.update_spatial_referent("A", "Pierre", gender="masc")
    ctx.clear()
    assert ctx.spatial_referents == {}


def test_clear_resets_turn_counter(ctx: ConversationContext) -> None:
    """Après clear(), les nouveaux tours recommencent à id=1."""
    _make_turn(ctx, "Avant clear")
    ctx.clear()
    new_turn = _make_turn(ctx, "Après clear")
    assert new_turn.id == 1


def test_clear_resets_last_sign_time(ctx: ConversationContext) -> None:
    """clear() doit remettre last_sign_time à 0.0."""
    ctx.touch_sign_time()
    assert ctx.last_sign_time > 0.0
    ctx.clear()
    assert ctx.last_sign_time == 0.0


def test_clear_resets_current_subjects(ctx: ConversationContext) -> None:
    """clear() doit vider current_subjects."""
    ctx.add_turn(
        text="Test sujet",
        raw_signs=["homme", "femme"],
        grammar_tags=["B-S", "B-S"],
        confidence=0.9,
    )
    ctx.clear()
    assert ctx.current_subjects == []


# ---------------------------------------------------------------------------
# Tests max_history respecté (deque)
# ---------------------------------------------------------------------------


def test_max_history_deque_drops_oldest(ctx: ConversationContext) -> None:
    """Après max_history tours, les plus anciens doivent être supprimés."""
    ctx_small = ConversationContext(max_history=5)
    for i in range(7):
        _make_turn(ctx_small, f"Turn {i}")

    assert len(ctx_small.history) == 5
    remaining_texts = [t.text for t in ctx_small.history]
    assert "Turn 0" not in remaining_texts
    assert "Turn 6" in remaining_texts


def test_max_history_is_deque_type(ctx: ConversationContext) -> None:
    """L'historique doit être un deque avec maxlen correct."""
    assert isinstance(ctx.history, deque)
    assert ctx.history.maxlen == 10


def test_max_history_large_volume_stable(ctx: ConversationContext) -> None:
    """Ajouter beaucoup de tours sans erreur ne doit pas lever d'exception."""
    ctx_large = ConversationContext(max_history=3)
    for i in range(100):
        _make_turn(ctx_large, f"Turn {i}")
    assert len(ctx_large.history) == 3


# ---------------------------------------------------------------------------
# Tests ConversationTurn.to_dict
# ---------------------------------------------------------------------------


def test_turn_to_dict_has_expected_keys(ctx: ConversationContext) -> None:
    """to_dict() doit produire un dict JSON-sérialisable avec toutes les clés requises."""
    turn = _make_turn(ctx, "Test phrase", signs=["SIGNE_X"])
    d = turn.to_dict()
    expected_keys = {"id", "text", "raw_signs", "grammar_tags", "confidence", "start_time", "end_time", "speaker"}
    assert expected_keys.issubset(set(d.keys()))
    assert d["text"] == "Test phrase"
    assert d["speaker"] == "signer"
    assert 0.0 <= d["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Tests get_full_history
# ---------------------------------------------------------------------------


def test_get_full_history_returns_serializable_list(ctx: ConversationContext) -> None:
    """get_full_history() doit retourner une liste de dicts sérialisables."""
    _make_turn(ctx, "Tour un", signs=["A"])
    _make_turn(ctx, "Tour deux", signs=["B"])
    history = ctx.get_full_history()
    assert isinstance(history, list)
    assert len(history) == 2
    assert all(isinstance(h, dict) for h in history)
    assert history[0]["text"] == "Tour un"
    assert history[1]["text"] == "Tour deux"
