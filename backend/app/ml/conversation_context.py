"""
Contexte conversationnel pour la traduction LSFB temps réel.

Maintient l'historique des phrases déjà traduites pour :
1. Résoudre les références anaphoriques (IL/ELLE → dernier sujet mentionné)
2. Éviter les répétitions (supprimer les doublons temporels)
3. Détecter les tours de parole (pause longue → nouveau tour)
4. Mémoriser les référents spatiaux (locus A = Pierre, locus B = Marie)
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Patterns anaphoriques LSFB → Français
# ---------------------------------------------------------------------------

# Pronoms LSFB susceptibles d'être résolus (insensible à la casse)
_ANAPHORA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bIL\b", re.IGNORECASE), "IL"),
    (re.compile(r"\bELLE\b", re.IGNORECASE), "ELLE"),
    (re.compile(r"\bILS\b", re.IGNORECASE), "ILS"),
    (re.compile(r"\bELLES\b", re.IGNORECASE), "ELLES"),
    (re.compile(r"\bLUI\b", re.IGNORECASE), "LUI"),
    (re.compile(r"\bLEUR\b", re.IGNORECASE), "LEUR"),
]

# Genres grammaticaux pour accord anaphorique
_MASCULINE_MARKERS: frozenset[str] = frozenset({
    "homme", "garçon", "père", "frère", "fils", "oncle", "monsieur",
    "roi", "ami", "collègue", "médecin",
})
_FEMININE_MARKERS: frozenset[str] = frozenset({
    "femme", "fille", "mère", "sœur", "tante", "madame", "reine",
    "amie", "collègue", "infirmière",
})


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """Un tour de parole dans la conversation."""

    id: int
    text: str                          # Phrase française traduite
    raw_signs: list[str]               # Labels bruts des signes
    grammar_tags: list[str]            # Tags BIO
    confidence: float
    start_time: float                  # timestamp Unix
    end_time: float
    speaker: str = "signer"           # "signer" | "user" | "system"

    def to_dict(self) -> dict:
        """Sérialise le tour pour envoi JSON."""
        return {
            "id": self.id,
            "text": self.text,
            "raw_signs": self.raw_signs,
            "grammar_tags": self.grammar_tags,
            "confidence": round(self.confidence, 3),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
        }


@dataclass
class SpatialReferent:
    """Référent spatial LSFB (locus)."""

    locus: str              # "A", "B", "C" etc.
    entity: str             # "Pierre", "la voiture", etc.
    introduced_at: float    # timestamp Unix
    gender: str = "unknown" # "masc" | "fem" | "neut" | "unknown"

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "entity": self.entity,
            "introduced_at": self.introduced_at,
            "gender": self.gender,
        }


# ---------------------------------------------------------------------------
# Contexte conversationnel principal
# ---------------------------------------------------------------------------


class ConversationContext:
    """
    Contexte conversationnel maintenu sur la durée d'une session de traduction.

    Responsabilités :
    - Garder l'historique glissant des tours de parole (``ConversationTurn``).
    - Résoudre les références anaphoriques simples (IL/ELLE → dernier sujet).
    - Détecter les séparations de tours sur pause longue.
    - Mémoriser les référents spatiaux LSFB (locus A/B/C → entité).
    - Éliminer les doublons temporels évidents (même phrase dans < 2 s).
    """

    def __init__(
        self,
        max_history: int = 20,
        turn_gap_seconds: float = 3.0,
        resolution_window: int = 5,
        dedup_window_seconds: float = 2.0,
    ) -> None:
        """
        Args:
            max_history: Nombre maximum de tours gardés en mémoire.
            turn_gap_seconds: Silence (en secondes) pour détecter un nouveau tour.
            resolution_window: Nombre de tours récents consultés pour la résolution
                anaphorique.
            dedup_window_seconds: Fenêtre de temps pour la dé-duplication.
        """
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)
        self.spatial_referents: dict[str, SpatialReferent] = {}
        self.current_subjects: list[str] = []  # Sujets récents pour résolution anaphorique
        self.turn_gap_seconds = turn_gap_seconds
        self.resolution_window = resolution_window
        self.dedup_window_seconds = dedup_window_seconds
        self.last_sign_time: float = 0.0
        self._turn_counter: int = 0
        self._session_start: float = time.time()

    # ------------------------------------------------------------------
    # Ajout de tours
    # ------------------------------------------------------------------

    def add_turn(
        self,
        text: str,
        raw_signs: list[str],
        grammar_tags: list[str],
        confidence: float,
        speaker: str = "signer",
    ) -> ConversationTurn:
        """Ajoute un tour de parole à l'historique.

        Applique la dé-duplication temporelle avant insertion.

        Args:
            text: Phrase française traduite.
            raw_signs: Labels bruts des signes détectés.
            grammar_tags: Tags BIO produits par le CRF.
            confidence: Confiance agrégée du tour.
            speaker: Locuteur du tour.

        Returns:
            Le ``ConversationTurn`` créé.
        """
        now = time.time()

        # Dé-duplication : ignorer si même texte que le tour précédent dans la fenêtre
        if self.history:
            last = self.history[-1]
            if (
                last.text.strip().lower() == text.strip().lower()
                and (now - last.end_time) < self.dedup_window_seconds
            ):
                logger.debug(
                    "conversation_turn_deduplicated",
                    text=text[:60],
                    age_ms=round((now - last.end_time) * 1000),
                )
                return last  # Retourner le tour existant, pas de doublon

        self._turn_counter += 1
        turn = ConversationTurn(
            id=self._turn_counter,
            text=text,
            raw_signs=list(raw_signs),
            grammar_tags=list(grammar_tags),
            confidence=float(confidence),
            start_time=now,
            end_time=now,
            speaker=speaker,
        )
        self.history.append(turn)
        self._update_subjects_from_turn(turn)
        self.last_sign_time = now
        logger.debug(
            "conversation_turn_added",
            turn_id=self._turn_counter,
            text=text[:60],
            signs=len(raw_signs),
        )
        return turn

    def touch_sign_time(self) -> None:
        """Met à jour le timestamp du dernier signe reçu (appelé à chaque frame active)."""
        self.last_sign_time = time.time()

    # ------------------------------------------------------------------
    # Détection de tours
    # ------------------------------------------------------------------

    def is_new_turn(self) -> bool:
        """Détecte si on est dans un nouveau tour de parole (pause longue).

        Returns ``True`` uniquement si on a déjà reçu au moins un signe
        (``last_sign_time > 0``) et que la pause dépasse ``turn_gap_seconds``.
        """
        if self.last_sign_time == 0.0:
            return False
        return (time.time() - self.last_sign_time) > self.turn_gap_seconds

    def detect_speech_pause(self, pause_duration_seconds: float) -> str:
        """Classifie la durée d'une pause entre signes.

        Args:
            pause_duration_seconds: Durée de la pause en secondes.

        Returns:
            ``'turn_end'``    si pause > turn_gap_seconds (fin de tour),
            ``'phrase_break'``si pause entre 0.5 s et turn_gap_seconds,
            ``'none'``        si pause < 0.5 s.
        """
        if pause_duration_seconds >= self.turn_gap_seconds:
            return "turn_end"
        if pause_duration_seconds >= 0.5:
            return "phrase_break"
        return "none"

    # ------------------------------------------------------------------
    # Résolution anaphorique
    # ------------------------------------------------------------------

    def resolve_anaphora(self, text: str) -> str:
        """Résout les références anaphoriques basiques.

        Stratégie :
        - Cherche IL / ELLE / ILS / ELLES dans le texte.
        - Résout depuis ``current_subjects`` (sujets récents extraits des tours).
        - Résout depuis les ``spatial_referents`` (locus LSFB).

        Args:
            text: Phrase traduite (peut contenir des pronoms à résoudre).

        Returns:
            Phrase avec références résolues (si possible).
        """
        if not text:
            return text

        resolved = text

        # Résolution depuis les référents spatiaux enregistrés
        for locus_pronoun_map in self._build_locus_pronoun_map():
            pronoun, entity = locus_pronoun_map
            pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
            resolved = pattern.sub(entity, resolved)

        # Résolution depuis les sujets récents (dernier sujet connu)
        last_subject = self._get_last_subject()
        if last_subject:
            gender = self._infer_gender(last_subject)
            if gender == "masc":
                pronoun_map = {
                    "IL": last_subject,
                    "LUI": last_subject,
                }
            elif gender == "fem":
                pronoun_map = {
                    "ELLE": last_subject,
                    "LUI": last_subject,
                }
            else:
                pronoun_map = {}

            for pronoun, replacement in pronoun_map.items():
                pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
                if pattern.search(resolved):
                    resolved = pattern.sub(replacement, resolved)
                    logger.debug(
                        "anaphora_resolved",
                        pronoun=pronoun,
                        replacement=replacement,
                        hint="recent_subject",
                    )

        return resolved

    # ------------------------------------------------------------------
    # Référents spatiaux
    # ------------------------------------------------------------------

    def update_spatial_referent(self, locus: str, entity: str, gender: str = "unknown") -> None:
        """Enregistre un référent spatial LSFB.

        Args:
            locus: Identifiant du locus (ex. "A", "B", "C").
            entity: Entité désignée (ex. "Pierre", "la voiture").
            gender: Genre grammatical pour accord (``'masc'``, ``'fem'``,
                ``'neut'``, ``'unknown'``).
        """
        self.spatial_referents[locus.upper()] = SpatialReferent(
            locus=locus.upper(),
            entity=entity,
            introduced_at=time.time(),
            gender=gender,
        )
        logger.debug("spatial_referent_updated", locus=locus, entity=entity)

    # ------------------------------------------------------------------
    # Résumé de contexte
    # ------------------------------------------------------------------

    def get_context_summary(self) -> dict:
        """Retourne un résumé sérialisable du contexte pour le frontend."""
        recent_turns = list(self.history)[-3:]
        return {
            "turn_count": len(self.history),
            "last_turns": [
                {"id": t.id, "text": t.text, "time": t.start_time, "confidence": round(t.confidence, 3)}
                for t in recent_turns
            ],
            "spatial_referents": {
                k: v.entity for k, v in self.spatial_referents.items()
            },
            "current_subjects": self.current_subjects[-3:],
            "session_age_seconds": round(time.time() - self._session_start, 1),
        }

    def get_full_history(self) -> list[dict]:
        """Retourne l'historique complet sérialisé."""
        return [t.to_dict() for t in self.history]

    # ------------------------------------------------------------------
    # Réinitialisation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Réinitialise le contexte (nouvelle conversation)."""
        self.history.clear()
        self.spatial_referents.clear()
        self.current_subjects.clear()
        self.last_sign_time = 0.0
        self._turn_counter = 0
        self._session_start = time.time()
        logger.info("conversation_context_cleared")

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _update_subjects_from_turn(self, turn: ConversationTurn) -> None:
        """Extrait et mémorise les sujets potentiels depuis un tour."""
        # On considère les tokens de signes bruts comme candidats sujets
        # (heuristique simple : premier token souvent = sujet en LSFB)
        for sign in turn.raw_signs[:2]:
            normalized = sign.strip().lower()
            if normalized and normalized not in ("none", "recording", "[signe-inconnu]"):
                # Éviter les doublons immédiats
                if not self.current_subjects or self.current_subjects[-1] != normalized:
                    self.current_subjects.append(normalized)
        # Garder les N derniers sujets
        if len(self.current_subjects) > 10:
            self.current_subjects = self.current_subjects[-10:]

    def _get_last_subject(self) -> str | None:
        """Retourne le dernier sujet connu (depuis l'historique récent)."""
        recent = list(self.history)[-(self.resolution_window):]
        for turn in reversed(recent):
            for sign in turn.raw_signs[:1]:
                normalized = sign.strip().lower()
                if normalized and normalized not in ("none", "recording"):
                    return normalized
        return None

    def _infer_gender(self, word: str) -> str:
        """Inférence heuristique du genre grammatical.

        Returns:
            ``'masc'``, ``'fem'``, ou ``'unknown'``.
        """
        lower = word.lower()
        if lower in _MASCULINE_MARKERS:
            return "masc"
        if lower in _FEMININE_MARKERS:
            return "fem"
        # Heuristique terminaison -e → féminin (très simpliste)
        if lower.endswith("e") and len(lower) > 3:
            return "fem"
        return "unknown"

    def _build_locus_pronoun_map(self) -> list[tuple[str, str]]:
        """Construit la liste (pronom_attendu, entité_de_remplacement) depuis les locus.

        Convention LSFB courante :
        - Locus A → pronom "il/lui" (ou "elle" selon genre)
        - Locus B → pronom "il/lui" pour le second référent
        """
        result: list[tuple[str, str]] = []
        for locus, ref in self.spatial_referents.items():
            if ref.gender == "masc":
                result.append(("IL", ref.entity))
                result.append(("LUI", ref.entity))
            elif ref.gender == "fem":
                result.append(("ELLE", ref.entity))
                result.append(("LUI", ref.entity))
        return result
