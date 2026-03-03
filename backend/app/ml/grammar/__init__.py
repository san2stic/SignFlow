"""Module de traduction grammaticale LSFB → Français.

Ce package implémente la Phase 3 de la grammaire LSFB automatique :

- Règles heuristiques déterministes (``lsfb_rules``)
- Tagger CRF supervisé (``lsfb_crf``)
- Pipeline de traduction complet (``lsfb_translator``)

Utilisation minimale (mode règles, sans dépendance ML) ::

    from app.ml.grammar import LSFBToFrenchTranslator, SignToken

    translator = LSFBToFrenchTranslator(mode="rules")
    result = translator.translate_buffer([
        {"label": "JE", "confidence": 0.9},
        {"label": "MANGER", "confidence": 0.85},
        {"label": "PAIN", "confidence": 0.8},
    ])
    # → "Je mange le pain."
"""

from app.ml.grammar.lsfb_crf import GRAMMAR_TAGS, LSFBSequenceTagger
from app.ml.grammar.lsfb_rules import (
    GrammarRole,
    LSFBGrammarRules,
    LSFB_NEGATION_SIGNS,
    LSFB_PRONOUNS,
    LSFB_WH_SIGNS,
    NMMAnalyzer,
    NMMMarkers,
    SignToken,
)
from app.ml.grammar.lsfb_translator import LSFBToFrenchTranslator, TranslationResult

__all__ = [
    # Règles
    "LSFBGrammarRules",
    "NMMAnalyzer",
    "NMMMarkers",
    "SignToken",
    "GrammarRole",
    "LSFB_WH_SIGNS",
    "LSFB_NEGATION_SIGNS",
    "LSFB_PRONOUNS",
    # CRF
    "LSFBSequenceTagger",
    "GRAMMAR_TAGS",
    # Traducteur
    "LSFBToFrenchTranslator",
    "TranslationResult",
]
