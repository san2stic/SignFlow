"""
Règles grammaticales heuristiques pour LSFB → Français.

Ces règles sont utilisées comme pré/post-traitement autour des modèles ML.
Elles couvrent les principales structures de la Langue des Signes de Belgique
Francophone (LSFB) :

- Ordre des constituants (Topique-Commentaire, SOV → SVO)
- Marqueurs non-manuels (NMM) : sourcils, tête, joues
- Négation (secouement de tête + signes NEG)
- Questions (oui/non : sourcils haussés ; WH : sourcils froncés + signe WH)
- Intensifieurs (joues gonflées)

Ce module fonctionne avec numpy uniquement (pas de dépendance ML).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Constantes lexicales LSFB
# ---------------------------------------------------------------------------

#: Signes WH connus en LSFB (question ouverte)
LSFB_WH_SIGNS: frozenset[str] = frozenset({
    "QUI", "QUOI", "OÙ", "OU", "QUAND", "COMMENT", "POURQUOI",
    "COMBIEN", "LEQUEL", "LAQUELLE", "LESQUELS", "LESQUELLES",
})

#: Signes de négation standalone
LSFB_NEGATION_SIGNS: frozenset[str] = frozenset({
    "NON", "PAS", "JAMAIS", "RIEN", "PERSONNE", "NE-PAS",
    "PAS-ENCORE", "PLUS", "NUL",
})

#: Pronoms LSFB (pointage directionnel)
LSFB_PRONOUNS: frozenset[str] = frozenset({
    "JE", "TU", "IL", "ELLE", "NOUS", "VOUS", "ILS", "ELLES",
    "MOI", "TOI", "LUI",
})

#: Verbes directionnels (utilisent l'espace signé pour indiquer sujet/objet)
LSFB_DIRECTIONAL_VERBS: frozenset[str] = frozenset({
    "DONNER", "DIRE", "MONTRER", "DEMANDER", "OFFRIR", "ENVOYER",
    "AIDER", "RÉPONDRE", "ÉCRIRE",
})

#: Copule être (souvent omise en LSFB, à réinsérer en français)
LSFB_ADJECTIVES: frozenset[str] = frozenset({
    "GRAND", "PETIT", "BEAU", "BELLE", "BON", "MAUVAIS", "CONTENT",
    "TRISTE", "FATIGUÉ", "MALADE", "LIBRE", "OCCUPÉ", "VIEUX", "JEUNE",
    "RAPIDE", "LENT", "FORT", "FAIBLE", "NOUVEAU", "ANCIEN",
})

#: Articles indéfinis par défaut (utilisés si aucun déterminant explicite)
_DEFAULT_ARTICLE = "le"
_VOWEL_START = re.compile(r"^[aeiouéèêëàâùûîïôœhAEIOUÉÈÊËÀÂÙÛÎÏÔŒH]")

# ---------------------------------------------------------------------------
# Énumérations et structures de données
# ---------------------------------------------------------------------------


class GrammarRole(str, Enum):
    """Rôle grammatical d'un token signe."""

    TOPIC = "topic"           # Topicalisation (en tête d'énoncé LSFB)
    SUBJECT = "subject"       # Sujet
    VERB = "verb"             # Verbe
    OBJECT = "object"         # Objet
    ADJECTIVE = "adjective"   # Adjectif (copule implicite)
    NEGATION = "negation"     # NMM secouement tête / signe NEG
    QUESTION_YN = "q_yn"      # NMM sourcils haussés + fin d'énoncé
    QUESTION_WH = "q_wh"      # NMM sourcils froncés + signe WH
    INTENSIFIER = "intensifier"  # NMM joues gonflées
    CLASSIFIER = "classifier" # Classifieur handshape
    PRONOUN = "pronoun"       # Pronom (pointage)
    UNKNOWN = "unknown"


class NMMMarkers(NamedTuple):
    """Marqueurs grammaticaux actifs déduits depuis les NMM."""

    polar_question: bool   # Question oui/non (sourcils haussés)
    wh_question: bool      # Question WH (sourcils froncés + WH)
    negation: bool         # Négation (secouement de tête)
    affirmation: bool      # Affirmation (hochement de tête)
    intensifier: bool      # Intensifieur (joues gonflées)


@dataclass
class SignToken:
    """Un signe avec ses métadonnées NMM et espace signé."""

    label: str                               # Ex: "BONJOUR", "MANGER", "JE"
    confidence: float                        # Confiance de la reconnaissance (0-1)
    start_frame: int = 0                     # Frame de début du signe
    end_frame: int = 0                       # Frame de fin du signe
    nmm_features: np.ndarray = field(        # 32 dims NMM moyennés sur la durée
        default_factory=lambda: np.zeros(32, dtype=np.float32)
    )
    signing_space: np.ndarray = field(       # 18 dims espace signé
        default_factory=lambda: np.zeros(18, dtype=np.float32)
    )
    grammar_role: GrammarRole = GrammarRole.UNKNOWN


# ---------------------------------------------------------------------------
# Analyseur NMM
# ---------------------------------------------------------------------------

# Indices dans le vecteur NMM 32-dims (cf. facial_action_units.py) :
#   [0] : brow raise gauche (AU1+AU2)
#   [1] : brow raise droit (AU1+AU2)
#   [3] : brow furrow gauche (AU4)
#   [4] : brow furrow droit (AU4)
#   [15]: cheek puff gauche
#   [16]: cheek puff droit
#   [20]: head pitch (nod, positif = hochement vers l'avant)
#   [21]: head yaw (shake, |yaw| > seuil → secouement)
#   [22]: head roll (tilt)

_IDX_BROW_RAISE_L = 0
_IDX_BROW_RAISE_R = 1
_IDX_BROW_FURROW_L = 3
_IDX_BROW_FURROW_R = 4
_IDX_CHEEK_PUFF_L = 15
_IDX_CHEEK_PUFF_R = 16
_IDX_HEAD_PITCH = 20
_IDX_HEAD_YAW = 21


class NMMAnalyzer:
    """Analyse les NMM pour déduire les marqueurs grammaticaux.

    Toutes les valeurs NMM sont normalisées dans [0, 1] ou [-1, 1]
    conformément à ``facial_action_units.py``.
    """

    # Seuils calibrés pour LSFB
    EYEBROW_RAISE_THRESHOLD: float = 0.28   # AU0+AU1 > seuil → question oui/non
    EYEBROW_FROWN_THRESHOLD: float = 0.22   # AU3+AU4 > seuil → question WH / négation
    HEAD_SHAKE_THRESHOLD: float = 0.18      # |yaw| > seuil → négation
    HEAD_NOD_THRESHOLD: float = 0.15        # pitch > seuil → affirmation
    PUFFED_CHEEKS_THRESHOLD: float = 0.28   # cheeks → intensifieur

    def analyze(self, nmm_features: np.ndarray) -> NMMMarkers:
        """Retourne les marqueurs grammaticaux actifs.

        Args:
            nmm_features: Vecteur NMM 32-dims (peut être zeros si non disponible).

        Returns:
            NMMMarkers avec les booléens des marqueurs actifs.
        """
        if nmm_features is None or nmm_features.shape[0] < 22:
            return NMMMarkers(False, False, False, False, False)

        brow_raise = float(
            0.5 * (nmm_features[_IDX_BROW_RAISE_L] + nmm_features[_IDX_BROW_RAISE_R])
        )
        brow_furrow = float(
            0.5 * (nmm_features[_IDX_BROW_FURROW_L] + nmm_features[_IDX_BROW_FURROW_R])
        )
        head_yaw = float(abs(nmm_features[_IDX_HEAD_YAW]))
        head_pitch = float(nmm_features[_IDX_HEAD_PITCH])

        cheek_puff = 0.0
        if nmm_features.shape[0] > _IDX_CHEEK_PUFF_R:
            cheek_puff = float(
                0.5 * (nmm_features[_IDX_CHEEK_PUFF_L] + nmm_features[_IDX_CHEEK_PUFF_R])
            )

        polar_question = brow_raise >= self.EYEBROW_RAISE_THRESHOLD
        wh_question = brow_furrow >= self.EYEBROW_FROWN_THRESHOLD and not polar_question
        negation = head_yaw >= self.HEAD_SHAKE_THRESHOLD
        affirmation = head_pitch >= self.HEAD_NOD_THRESHOLD and not negation
        intensifier = cheek_puff >= self.PUFFED_CHEEKS_THRESHOLD

        return NMMMarkers(
            polar_question=polar_question,
            wh_question=wh_question,
            negation=negation,
            affirmation=affirmation,
            intensifier=intensifier,
        )

    def aggregate_sequence(self, nmm_sequence: list[np.ndarray]) -> np.ndarray:
        """Agrège une séquence de vecteurs NMM en un seul vecteur moyen.

        Args:
            nmm_sequence: Liste de vecteurs NMM par frame.

        Returns:
            Vecteur NMM 32-dims moyen.
        """
        if not nmm_sequence:
            return np.zeros(32, dtype=np.float32)
        stack = np.stack(nmm_sequence, axis=0)
        return stack.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Moteur de règles LSFB
# ---------------------------------------------------------------------------


class LSFBGrammarRules:
    """Applique les règles grammaticales LSFB pour produire une phrase française.

    Règles principales :
    1. Détection de la structure de phrase (Topique-Commentaire vs SOV)
    2. Marquage de la négation (NMM secouement + signe NEG)
    3. Détection questions (NMM sourcils)
    4. Réordonnancement SOV → SVO pour le français
    5. Insertion d'articles/copule manquants (règles de surface)
    """

    def __init__(self, nmm_analyzer: NMMAnalyzer | None = None) -> None:
        self.nmm_analyzer = nmm_analyzer or NMMAnalyzer()
        self.wh_signs = LSFB_WH_SIGNS
        self.negation_signs = LSFB_NEGATION_SIGNS
        self.pronouns = LSFB_PRONOUNS
        self.directional_verbs = LSFB_DIRECTIONAL_VERBS
        self.adjectives = LSFB_ADJECTIVES

    # ------------------------------------------------------------------
    # Tagging grammatical
    # ------------------------------------------------------------------

    def tag_grammar_roles(self, tokens: list[SignToken]) -> list[SignToken]:
        """Assigne un GrammarRole à chaque token via les NMM et la position.

        Heuristiques utilisées :
        - Pronom en position 0 → SUBJECT
        - Pronom après VERB  → OBJECT (pronom objet)
        - Signe WH            → QUESTION_WH
        - Signe NEG           → NEGATION
        - Signe dans pronouns → PRONOUN (puis SUBJECT/OBJECT selon position)
        - Signe en position 0 non-pronom, non-verbe → TOPIC
        - Dernier signe avant NEG = VERB
        - Adjectifs connus   → ADJECTIVE

        Returns:
            Copie de la liste avec grammar_role renseigné.
        """
        if not tokens:
            return tokens

        result = [
            SignToken(
                label=t.label,
                confidence=t.confidence,
                start_frame=t.start_frame,
                end_frame=t.end_frame,
                nmm_features=t.nmm_features.copy(),
                signing_space=t.signing_space.copy(),
                grammar_role=t.grammar_role,
            )
            for t in tokens
        ]

        labels = [t.label.upper() for t in result]
        n = len(labels)

        # --- Pré-pass : Analyse NMM au niveau de l'énoncé ---
        # Les marqueurs de question YN et de négation sont souvent portés tout
        # au long de l'énoncé par les sourcils/la tête. On agrège les NMM de
        # tous les tokens pour détecter les marqueurs d'énoncé.
        all_nmm = [tok.nmm_features for tok in result]
        utterance_nmm = self.nmm_analyzer.aggregate_sequence(all_nmm)
        utterance_markers = self.nmm_analyzer.analyze(utterance_nmm)

        # --- Pass 1 : Rôles lexicaux évidents ---
        verb_idx: int | None = None
        subject_idx: int | None = None
        object_idx: int | None = None

        # Ensemble des marqueurs NMM per-token (pour négation locale)
        has_lexical_negation = any(label in self.negation_signs for label in labels)
        has_wh_sign = any(label in self.wh_signs for label in labels)

        for i, (tok, label) in enumerate(zip(result, labels)):
            if label in self.wh_signs:
                tok.grammar_role = GrammarRole.QUESTION_WH
            elif label in self.negation_signs:
                tok.grammar_role = GrammarRole.NEGATION
            elif label in self.adjectives:
                tok.grammar_role = GrammarRole.ADJECTIVE
            elif label in self.pronouns:
                tok.grammar_role = GrammarRole.PRONOUN
            else:
                # Analyser les NMM du token individuel pour la négation locale
                token_markers = self.nmm_analyzer.analyze(tok.nmm_features)
                if token_markers.negation and not has_lexical_negation:
                    tok.grammar_role = GrammarRole.NEGATION
                # Note : polar_question est traité au niveau énoncé (post-pass 2)

        # --- Pass 2 : Rôles syntaxiques (position + contexte) ---
        # Le premier token non-WH non-NEG est souvent topique ou sujet
        for i, (tok, label) in enumerate(zip(result, labels)):
            if tok.grammar_role in (GrammarRole.UNKNOWN, GrammarRole.PRONOUN):
                if i == 0:
                    if label in self.pronouns:
                        tok.grammar_role = GrammarRole.SUBJECT
                        subject_idx = i
                    else:
                        tok.grammar_role = GrammarRole.TOPIC
                elif (
                    tok.grammar_role == GrammarRole.UNKNOWN
                    and verb_idx is None
                    and i > 0
                ):
                    # Premier token inconnu après le sujet → verbe plausible
                    if subject_idx is not None or i > 0:
                        tok.grammar_role = GrammarRole.VERB
                        verb_idx = i
                elif tok.grammar_role == GrammarRole.UNKNOWN and verb_idx is not None:
                    if label in self.pronouns:
                        tok.grammar_role = GrammarRole.OBJECT
                        object_idx = i
                    else:
                        tok.grammar_role = GrammarRole.OBJECT
                        object_idx = i
                elif tok.grammar_role == GrammarRole.PRONOUN and verb_idx is not None:
                    tok.grammar_role = GrammarRole.OBJECT
                    object_idx = i

        # Cas où la séquence commence par pronom + verbe déjà identifié
        # → réajuster les inconnus restants en OBJECT
        if verb_idx is None:
            # Chercher le verbe parmi les UNKNOWN restants (pas pronoms ni NEG ni WH)
            for i, (tok, label) in enumerate(zip(result, labels)):
                if tok.grammar_role in (GrammarRole.UNKNOWN, GrammarRole.TOPIC) and i > 0:
                    tok.grammar_role = GrammarRole.VERB
                    verb_idx = i
                    break

        # --- Pass 3 : Marqueurs d'énoncé (question YN détectée via NMM globaux) ---
        # Une question YN sans signe WH et sans signe NEG dominant est signalée
        # par un token virtuel ajouté à la fin. On le place en queue de liste
        # pour que reorder_to_french() le traite comme marqueur post-proposition.
        if (
            utterance_markers.polar_question
            and not has_wh_sign
            and not has_lexical_negation
            and not any(t.grammar_role == GrammarRole.NEGATION for t in result)
        ):
            # Injecter un token marqueur question YN (label vide, rôle explicite)
            yn_marker = SignToken(
                label="__QYN__",
                confidence=1.0,
                nmm_features=utterance_nmm.copy(),
                signing_space=np.zeros(18, dtype=np.float32),
                grammar_role=GrammarRole.QUESTION_YN,
            )
            result.append(yn_marker)

        return result

    # ------------------------------------------------------------------
    # Réordonnancement SOV → SVO
    # ------------------------------------------------------------------

    def reorder_to_french(self, tokens: list[SignToken]) -> list[SignToken]:
        """Réordonne les tokens de SOV/OSV LSFB vers SVO français.

        Stratégie :
        1. Séparer TOPIC (pré-proposition) et NEG/Q (post-proposition)
        2. Identifier SUJET, VERBE, OBJET dans le noyau
        3. Reconstruire dans l'ordre : [TOPIC] + SUJET + VERBE + OBJET + [NEG/Q]

        Returns:
            Tokens réordonnés (ordre SVO).
        """
        if not tokens:
            return tokens

        pre: list[SignToken] = []     # Marqueurs en tête (topics)
        core: list[SignToken] = []    # Noyau de la phrase
        post: list[SignToken] = []    # Marqueurs en queue (NEG, Q, INTENS)

        for tok in tokens:
            if tok.grammar_role == GrammarRole.TOPIC:
                pre.append(tok)
            elif tok.grammar_role in (
                GrammarRole.QUESTION_WH,
                GrammarRole.QUESTION_YN,
                GrammarRole.NEGATION,
                GrammarRole.INTENSIFIER,
            ):
                post.append(tok)
            else:
                core.append(tok)

        # Réordonnancement du noyau : S → V → O
        subjects = [t for t in core if t.grammar_role in (GrammarRole.SUBJECT, GrammarRole.PRONOUN)]
        verbs = [t for t in core if t.grammar_role == GrammarRole.VERB]
        adjectives = [t for t in core if t.grammar_role == GrammarRole.ADJECTIVE]
        objects = [t for t in core if t.grammar_role in (GrammarRole.OBJECT,)]
        others = [
            t for t in core
            if t.grammar_role not in (
                GrammarRole.SUBJECT, GrammarRole.PRONOUN,
                GrammarRole.VERB, GrammarRole.OBJECT, GrammarRole.ADJECTIVE,
            )
        ]

        reordered_core = subjects + verbs + adjectives + objects + others
        # Si aucun sujet/verbe identifié, conserver l'ordre d'origine
        if not reordered_core:
            reordered_core = core

        return pre + reordered_core + post

    # ------------------------------------------------------------------
    # Règles de surface : négation, questions, articles, copule
    # ------------------------------------------------------------------

    def _label_to_french(self, label: str) -> str:
        """Convertit un label LSFB en mot français approximatif.

        Stratégie minimale : lowercase du label, avec quelques mappings
        courants.  Un vrai dictionnaire sera fourni à terme.
        """
        _MAPPINGS: dict[str, str] = {
            "JE": "je",
            "TU": "tu",
            "IL": "il",
            "ELLE": "elle",
            "NOUS": "nous",
            "VOUS": "vous",
            "ILS": "ils",
            "ELLES": "elles",
            "MOI": "moi",
            "TOI": "toi",
            "LUI": "lui",
            "NON": "non",
            "PAS": "pas",
            "JAMAIS": "jamais",
            "RIEN": "rien",
            "PERSONNE": "personne",
            "NE-PAS": "pas",
            "PAS-ENCORE": "pas encore",
            "QUI": "qui",
            "QUOI": "quoi",
            "OÙ": "où",
            "OU": "où",
            "QUAND": "quand",
            "COMMENT": "comment",
            "POURQUOI": "pourquoi",
            "COMBIEN": "combien",
            "LEQUEL": "lequel",
            "LAQUELLE": "laquelle",
            "BONJOUR": "bonjour",
            "MERCI": "merci",
            "OUI": "oui",
            "MANGER": "manger",
            "BOIRE": "boire",
            "DORMIR": "dormir",
            "ALLER": "aller",
            "VENIR": "venir",
            "AVOIR": "avoir",
            "ÊTRE": "être",
            "VOULOIR": "vouloir",
            "POUVOIR": "pouvoir",
            "SAVOIR": "savoir",
            "AIMER": "aimer",
            "COMPRENDRE": "comprendre",
            "PAIN": "pain",
            "EAU": "eau",
            "MAISON": "maison",
            "VOITURE": "voiture",
            "ÉCOLE": "école",
            "TRAVAIL": "travail",
            "FAMILLE": "famille",
            "AMI": "ami",
            "CHAT": "chat",
            "CHIEN": "chien",
            "GRAND": "grand",
            "PETIT": "petit",
            "BON": "bon",
            "MAUVAIS": "mauvais",
            "CONTENT": "content",
            "TRISTE": "triste",
            "FATIGUÉ": "fatigué",
            "MALADE": "malade",
            "BEAU": "beau",
            "BELLE": "belle",
        }
        key = label.upper().strip()
        if key in _MAPPINGS:
            return _MAPPINGS[key]
        # Fallback : lowercase avec accents préservés
        return label.lower().strip()

    def _add_article(self, word: str) -> str:
        """Prépose un article défaut selon l'initiale (voyelle/consonne)."""
        if _VOWEL_START.match(word):
            return f"l'{word}"
        return f"{_DEFAULT_ARTICLE} {word}"

    def _should_add_article(self, tok: SignToken) -> bool:
        """Décide si un article doit être inséré devant ce token."""
        label_up = tok.label.upper()
        return (
            tok.grammar_role == GrammarRole.OBJECT
            and label_up not in self.pronouns
            and label_up not in self.negation_signs
            and label_up not in self.wh_signs
        )

    def apply_negation(self, tokens: list[SignToken]) -> str:
        """Insère la négation française (ne … pas) aux bonnes positions.

        Si la liste contient un token NEGATION, la sortie inclut "ne…pas".
        La copule "être" est insérée si le verbe manque (p.ex. adj seul).

        Returns:
            Chaîne française avec négation.
        """
        neg_tokens = [t for t in tokens if t.grammar_role == GrammarRole.NEGATION]
        other_tokens = [t for t in tokens if t.grammar_role != GrammarRole.NEGATION]

        if not neg_tokens:
            return self._tokens_to_raw_string(tokens)

        # Identifier le verbe ou insérer "être"
        verb_toks = [t for t in other_tokens if t.grammar_role == GrammarRole.VERB]
        if not verb_toks:
            adj_toks = [t for t in other_tokens if t.grammar_role == GrammarRole.ADJECTIVE]
            if adj_toks:
                # Copule implicite → "n'est pas <adj>"
                subj_parts = [
                    self._label_to_french(t.label)
                    for t in other_tokens
                    if t.grammar_role in (GrammarRole.SUBJECT, GrammarRole.PRONOUN)
                ]
                adj_parts = [self._label_to_french(t.label) for t in adj_toks]
                subj_str = " ".join(subj_parts)
                adj_str = " ".join(adj_parts)
                if subj_str and _VOWEL_START.match(subj_str):
                    return f"{subj_str} n'est pas {adj_str}"
                elif subj_str:
                    return f"{subj_str} n'est pas {adj_str}"
                return f"n'est pas {adj_str}"

        # Construire la phrase avec "ne … pas"
        parts: list[str] = []
        neg_inserted = False

        for tok in other_tokens:
            word = self._label_to_french(tok.label)
            if self._should_add_article(tok):
                word = self._add_article(word)
            if tok.grammar_role == GrammarRole.VERB and not neg_inserted:
                # Insérer "ne" avant le verbe et "pas" après
                if parts and _VOWEL_START.match(word):
                    parts.append("n'")
                else:
                    parts.append("ne")
                parts.append(word)
                parts.append("pas")
                neg_inserted = True
            else:
                parts.append(word)

        if not neg_inserted:
            # Aucun verbe → insertion par défaut à la fin
            if parts and _VOWEL_START.match(parts[0]):
                parts = ["n'"] + parts + ["pas"]
            else:
                parts = ["ne"] + parts + ["pas"]

        return " ".join(parts).replace("n' ", "n'").strip()

    def apply_question(self, tokens: list[SignToken], q_type: str) -> str:
        """Formate la phrase en question.

        Args:
            tokens: Tokens grammaticalement ordonnés.
            q_type: ``'yn'`` (oui/non) ou ``'wh'`` (WH).

        Returns:
            Phrase française terminée par ``?``.
        """
        # Filtrer les marqueurs de question eux-mêmes
        content_tokens = [
            t for t in tokens
            if t.grammar_role not in (GrammarRole.QUESTION_YN, GrammarRole.QUESTION_WH)
        ]

        if q_type == "wh":
            wh_tokens = [
                t for t in tokens if t.grammar_role == GrammarRole.QUESTION_WH
            ]
            wh_word = ""
            if wh_tokens:
                wh_word = self._label_to_french(wh_tokens[0].label)

            rest = self._tokens_to_raw_string(content_tokens)
            if wh_word:
                return f"{wh_word.capitalize()} {rest} ?".strip()
            return f"{rest} ?".strip()
        else:
            # Question oui/non : "Est-ce que <phrase> ?"  ou simple ajout de "?"
            base = self._tokens_to_raw_string(content_tokens)
            if not base:
                return "?"
            return f"{base} ?".strip()

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------

    def _tokens_to_raw_string(self, tokens: list[SignToken]) -> str:
        """Convertit les tokens ordonnés en chaîne brute sans post-traitement."""
        parts: list[str] = []
        for tok in tokens:
            if tok.grammar_role in (GrammarRole.NEGATION, GrammarRole.QUESTION_YN,
                                    GrammarRole.QUESTION_WH):
                continue
            word = self._label_to_french(tok.label)
            if self._should_add_article(tok):
                word = self._add_article(word)
            parts.append(word)
        return " ".join(parts)

    def tokens_to_french(self, tokens: list[SignToken]) -> str:
        """Pipeline complet LSFB → français.

        Étapes :
        1. ``tag_grammar_roles()``
        2. ``reorder_to_french()``
        3. ``apply_negation()`` si nécessaire
        4. ``apply_question()`` si nécessaire
        5. Capitalisation + ponctuation

        Args:
            tokens: Tokens bruts (grammar_role peut être UNKNOWN).

        Returns:
            Phrase française finale.
        """
        if not tokens:
            return ""

        # 1. Tagging
        tagged = self.tag_grammar_roles(tokens)

        # 2. Réordonnancement
        ordered = self.reorder_to_french(tagged)

        # 3. Détecter la modalité
        has_negation = any(t.grammar_role == GrammarRole.NEGATION for t in ordered)
        has_yn_q = any(t.grammar_role == GrammarRole.QUESTION_YN for t in ordered)
        has_wh_q = any(t.grammar_role == GrammarRole.QUESTION_WH for t in ordered)

        # 4. Appliquer les règles de modalité
        if has_negation and not has_wh_q:
            result = self.apply_negation(ordered)
        elif has_wh_q:
            result = self.apply_question(ordered, q_type="wh")
        elif has_yn_q:
            result = self.apply_question(ordered, q_type="yn")
        else:
            # Phrase assertive ordinaire
            result = self._tokens_to_raw_string(ordered)

        # 5. Capitalisation et ponctuation de base
        result = result.strip()
        if not result:
            return ""
        result = result[0].upper() + result[1:]

        # S'assurer qu'un point final est présent (sauf si déjà ponctué)
        if result and result[-1] not in ".?!":
            result += "."

        # Nettoyage des doubles espaces
        result = re.sub(r"\s{2,}", " ", result)
        return result
