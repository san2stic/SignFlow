"""
CRF (Conditional Random Field) supervisé pour le tagging grammatical des
séquences de signes LSFB.

Utilise sklearn-crfsuite si disponible ; repli sur les règles heuristiques
LSFBGrammarRules si la bibliothèque est absente ou si aucun modèle n'est chargé.

Étiquettes BIO utilisées :
    B-SUBJ / I-SUBJ  : Sujet
    B-VERB / I-VERB  : Verbe
    B-OBJ  / I-OBJ   : Objet
    B-TOPIC/ I-TOPIC : Topique
    B-NEG            : Négation
    B-Q              : Marqueur de question
    B-INT            : Intensifieur
    O                : Autre

Le tagger peut être entraîné avec ``fit()``, sauvegardé avec ``save()`` et
rechargé avec ``load()``.  En l'absence de modèle, ``predict()`` convertit
les GrammarRole issus des règles heuristiques en tags BIO.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.ml.grammar.lsfb_rules import (
    GrammarRole,
    LSFBGrammarRules,
    LSFB_NEGATION_SIGNS,
    LSFB_PRONOUNS,
    LSFB_WH_SIGNS,
    NMMAnalyzer,
    SignToken,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tentative d'import de sklearn-crfsuite (optionnel)
# ---------------------------------------------------------------------------

try:
    import sklearn_crfsuite  # type: ignore[import]
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False
    logger.debug("sklearn_crfsuite not available — CRF tagger will use rule-based fallback")

# ---------------------------------------------------------------------------
# Étiquettes BIO
# ---------------------------------------------------------------------------

GRAMMAR_TAGS: list[str] = [
    "B-SUBJ", "I-SUBJ",
    "B-VERB", "I-VERB",
    "B-OBJ",  "I-OBJ",
    "B-TOPIC","I-TOPIC",
    "B-NEG",
    "B-Q",
    "B-INT",
    "O",
]

_ROLE_TO_BIO: dict[GrammarRole, str] = {
    GrammarRole.SUBJECT:     "B-SUBJ",
    GrammarRole.PRONOUN:     "B-SUBJ",
    GrammarRole.VERB:        "B-VERB",
    GrammarRole.OBJECT:      "B-OBJ",
    GrammarRole.TOPIC:       "B-TOPIC",
    GrammarRole.NEGATION:    "B-NEG",
    GrammarRole.QUESTION_YN: "B-Q",
    GrammarRole.QUESTION_WH: "B-Q",
    GrammarRole.INTENSIFIER: "B-INT",
    GrammarRole.ADJECTIVE:   "B-OBJ",   # Adjectifs traités comme objets en BIO
    GrammarRole.UNKNOWN:     "O",
    GrammarRole.CLASSIFIER:  "O",
}

# Mapping rôle grammatical → prédicat court (pour bigrammes)
_ROLE_SHORT: dict[GrammarRole, str] = {
    GrammarRole.SUBJECT:     "SUBJ",
    GrammarRole.PRONOUN:     "SUBJ",
    GrammarRole.VERB:        "VERB",
    GrammarRole.OBJECT:      "OBJ",
    GrammarRole.TOPIC:       "TOPIC",
    GrammarRole.NEGATION:    "NEG",
    GrammarRole.QUESTION_YN: "Q",
    GrammarRole.QUESTION_WH: "Q_WH",
    GrammarRole.INTENSIFIER: "INT",
    GrammarRole.ADJECTIVE:   "ADJ",
    GrammarRole.UNKNOWN:     "UNK",
    GrammarRole.CLASSIFIER:  "CLS",
}


# ---------------------------------------------------------------------------
# Feature extraction pour le CRF
# ---------------------------------------------------------------------------

def _safe_nmm_binary(nmm: np.ndarray, idx: int, threshold: float) -> bool:
    """Retourne True si la valeur NMM[idx] dépasse le seuil."""
    if nmm is None or nmm.shape[0] <= idx:
        return False
    return bool(float(nmm[idx]) >= threshold)


def _token_features(tokens: list[SignToken], i: int) -> dict[str, object]:
    """Extrait les features CRF pour le token i dans son contexte.

    Features calculées :
    - label du signe (lowercase)
    - position relative (début/milieu/fin)
    - appartenance aux ensembles lexicaux LSFB
    - features NMM binarisées
    - contexte ±2 tokens (label + appartenance aux ensembles)
    - 18 dimensions de signing_space encodées comme "space_{i}" → float
    - bigrammes de rôle grammatical (rôle du token précédent/suivant)
    - feature "is_first_in_domain" (heuristique : signing_space[:3] ≈ 0 → espace neutre)
    """
    tok = tokens[i]
    label = tok.label.upper().strip()
    n = len(tokens)
    nmm = tok.nmm_features
    signing_space = tok.signing_space

    # --- Features intrinsèques ---
    features: dict[str, object] = {
        "label": label.lower(),
        "is_wh": label in LSFB_WH_SIGNS,
        "is_neg": label in LSFB_NEGATION_SIGNS,
        "is_pronoun": label in LSFB_PRONOUNS,
        "pos_relative": round(i / max(n - 1, 1), 2),
        "is_first": i == 0,
        "is_last": i == n - 1,
        # NMM binarisées (seuils issus de NMMAnalyzer)
        "nmm_brow_raise": _safe_nmm_binary(nmm, 0, NMMAnalyzer.EYEBROW_RAISE_THRESHOLD)
                          or _safe_nmm_binary(nmm, 1, NMMAnalyzer.EYEBROW_RAISE_THRESHOLD),
        "nmm_brow_furrow": _safe_nmm_binary(nmm, 3, NMMAnalyzer.EYEBROW_FROWN_THRESHOLD)
                           or _safe_nmm_binary(nmm, 4, NMMAnalyzer.EYEBROW_FROWN_THRESHOLD),
        "nmm_head_shake": _safe_nmm_binary(nmm, 21, NMMAnalyzer.HEAD_SHAKE_THRESHOLD),
        "nmm_head_nod":   _safe_nmm_binary(nmm, 20, NMMAnalyzer.HEAD_NOD_THRESHOLD),
        "nmm_cheek_puff": (
            _safe_nmm_binary(nmm, 15, NMMAnalyzer.PUFFED_CHEEKS_THRESHOLD)
            or _safe_nmm_binary(nmm, 16, NMMAnalyzer.PUFFED_CHEEKS_THRESHOLD)
        ),
        "confidence_high": tok.confidence >= 0.75,
        "confidence_mid":  0.50 <= tok.confidence < 0.75,
    }

    # --- Signing space : 18 dimensions encodées individuellement ---
    if signing_space is not None and len(signing_space) >= 18:
        for si in range(18):
            features[f"space_{si}"] = round(float(signing_space[si]), 4)
    else:
        for si in range(18):
            features[f"space_{si}"] = 0.0

    # --- Heuristique is_first_in_domain ---
    # Si les 3 premières dimensions de signing_space sont proches de 0,
    # le signe est exécuté dans l'espace neutre central → potentiellement
    # marqueur de premier constituant dans son domaine grammatical.
    if signing_space is not None and len(signing_space) >= 3:
        space_norm = float(np.linalg.norm(signing_space[:3]))
        features["is_first_in_domain"] = space_norm < 0.15
    else:
        features["is_first_in_domain"] = False

    # --- Bigrammes de rôle : rôle du token précédent et suivant ---
    prev_role_str = "__BOS__"
    next_role_str = "__EOS__"

    if i - 1 >= 0:
        prev_tok = tokens[i - 1]
        prev_role_str = _ROLE_SHORT.get(prev_tok.grammar_role, "UNK")
    if i + 1 < n:
        next_tok = tokens[i + 1]
        next_role_str = _ROLE_SHORT.get(next_tok.grammar_role, "UNK")

    curr_role_str = _ROLE_SHORT.get(tok.grammar_role, "UNK")
    features["role_curr"] = curr_role_str
    features["role_prev"] = prev_role_str
    features["role_next"] = next_role_str
    features["bigram_prev_curr"] = f"{prev_role_str}+{curr_role_str}"
    features["bigram_curr_next"] = f"{curr_role_str}+{next_role_str}"

    # --- Contexte −2 ---
    for offset, prefix in [(-2, "m2"), (-1, "m1"), (1, "p1"), (2, "p2")]:
        j = i + offset
        if 0 <= j < n:
            ctx = tokens[j]
            ctx_label = ctx.label.upper().strip()
            features[f"{prefix}_label"] = ctx_label.lower()
            features[f"{prefix}_is_wh"] = ctx_label in LSFB_WH_SIGNS
            features[f"{prefix}_is_neg"] = ctx_label in LSFB_NEGATION_SIGNS
            features[f"{prefix}_is_pronoun"] = ctx_label in LSFB_PRONOUNS
        else:
            features[f"{prefix}_label"] = "__BOS__" if offset < 0 else "__EOS__"
            features[f"{prefix}_is_wh"] = False
            features[f"{prefix}_is_neg"] = False
            features[f"{prefix}_is_pronoun"] = False

    return features


def _sequence_features(tokens: list[SignToken]) -> list[dict[str, object]]:
    """Extrait les features CRF pour toute une séquence."""
    return [_token_features(tokens, i) for i in range(len(tokens))]


# ---------------------------------------------------------------------------
# Corpus synthétique (bootstrap CRF sans données réelles)
# ---------------------------------------------------------------------------

def generate_synthetic_corpus(
    n_sequences: int = 200,
    min_len: int = 3,
    max_len: int = 12,
    seed: int = 42,
) -> tuple[list[list[SignToken]], list[list[str]]]:
    """Génère un corpus synthétique BIO annoté pour bootstrapper le CRF.

    Règles de génération :
    - Pioche aléatoirement des signes dans LSFB_WH_SIGNS, LSFB_NEGATION_SIGNS,
      LSFB_PRONOUNS et dans une liste de signes lexicaux génériques.
    - Assigne un GrammarRole cohérent.
    - Génère des tags BIO cohérents (B- pour début de groupe, I- pour continuité).

    Patterns de séquences générés (tirés aléatoirement) :
      1. SVO simple : [SUBJECT, PREDICATE, OBJECT]     → B-SUBJ, B-VERB, B-OBJ
      2. Question WH : [WH, SUBJECT, PREDICATE]        → B-Q, B-SUBJ, B-VERB
      3. Négation : [SUBJECT, NEG, PREDICATE]          → B-SUBJ, B-NEG, B-VERB
      4. Topique : [TOPIC, SUBJECT, PREDICATE]         → B-TOPIC, B-SUBJ, B-VERB

    Args:
        n_sequences: Nombre de séquences à générer.
        min_len: Longueur minimale des séquences.
        max_len: Longueur maximale des séquences.
        seed: Graine aléatoire (numpy.random.default_rng).

    Returns:
        Tuple ``(sequences, labels)`` où chaque liste est alignée par index.
    """
    rng = np.random.default_rng(seed)

    # Vocabulaire lexical générique
    _LEXICAL_SIGNS = [
        "MANGER", "BOIRE", "ALLER", "VOIR", "FAIRE", "VOULOIR",
        "POUVOIR", "AVOIR", "ÊTRE", "DONNER", "PRENDRE", "VENIR",
    ]

    # Listes accessibles sous forme de list
    _wh_list = sorted(LSFB_WH_SIGNS)
    _neg_list = sorted(LSFB_NEGATION_SIGNS)
    _pronoun_list = sorted(LSFB_PRONOUNS)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _make_token(
        label: str,
        role: GrammarRole,
        rng: np.random.Generator,
    ) -> SignToken:
        nmm = rng.random(32).astype(np.float32) * 0.4  # valeurs basses (neutre)
        space = rng.random(18).astype(np.float32) * 0.5
        conf = float(rng.uniform(0.7, 1.0))
        return SignToken(
            label=label,
            confidence=conf,
            nmm_features=nmm,
            signing_space=space,
            grammar_role=role,
        )

    def _pick(lst: list[str], rng: np.random.Generator) -> str:
        return lst[int(rng.integers(len(lst)))]

    def _pick_lexical(rng: np.random.Generator) -> str:
        return _LEXICAL_SIGNS[int(rng.integers(len(_LEXICAL_SIGNS)))]

    def _extend_group(
        base_label: str,
        base_role: GrammarRole,
        base_bio: str,
        extra_len: int,
        rng: np.random.Generator,
    ) -> tuple[list[SignToken], list[str]]:
        """Génère un groupe de tokens avec tags I- de continuité."""
        tokens_g = [_make_token(base_label, base_role, rng)]
        tags_g = [base_bio]
        head = base_bio[2:]  # ex. "SUBJ"
        continuation = f"I-{head}"
        if continuation not in GRAMMAR_TAGS:
            continuation = base_bio  # repli sur B- si pas de I-
        for _ in range(extra_len):
            tokens_g.append(_make_token(_pick_lexical(rng), base_role, rng))
            tags_g.append(continuation)
        return tokens_g, tags_g

    # ----------------------------------------------------------------
    # Patterns
    # ----------------------------------------------------------------

    def _pattern_svo(rng: np.random.Generator, extra: int) -> tuple[list[SignToken], list[str]]:
        """SVO : [SUBJECT+, PREDICATE+, OBJECT+]"""
        subj_label = _pick(_pronoun_list, rng)
        verb_label = _pick_lexical(rng)
        obj_label = _pick(_pronoun_list if rng.random() > 0.5 else _LEXICAL_SIGNS, rng)

        # répartir les tokens supplémentaires
        extra_s = int(rng.integers(0, max(1, extra // 2 + 1)))
        extra_v = int(rng.integers(0, max(1, extra - extra_s + 1)))

        toks, tags = [], []
        t, g = _extend_group(subj_label, GrammarRole.SUBJECT, "B-SUBJ", extra_s, rng)
        toks += t; tags += g
        t, g = _extend_group(verb_label, GrammarRole.VERB, "B-VERB", extra_v, rng)
        toks += t; tags += g
        t, g = _extend_group(obj_label, GrammarRole.OBJECT, "B-OBJ", 0, rng)
        toks += t; tags += g
        return toks, tags

    def _pattern_wh(rng: np.random.Generator, extra: int) -> tuple[list[SignToken], list[str]]:
        """Question WH : [WH, SUBJECT+, PREDICATE+]"""
        wh_label = _pick(_wh_list, rng)
        subj_label = _pick(_pronoun_list, rng)
        verb_label = _pick_lexical(rng)

        extra_s = int(rng.integers(0, max(1, extra + 1)))

        toks, tags = [], []
        toks.append(_make_token(wh_label, GrammarRole.QUESTION_WH, rng))
        tags.append("B-Q")
        t, g = _extend_group(subj_label, GrammarRole.SUBJECT, "B-SUBJ", extra_s, rng)
        toks += t; tags += g
        t, g = _extend_group(verb_label, GrammarRole.VERB, "B-VERB", 0, rng)
        toks += t; tags += g
        return toks, tags

    def _pattern_neg(rng: np.random.Generator, extra: int) -> tuple[list[SignToken], list[str]]:
        """Négation : [SUBJECT, NEG, PREDICATE+]"""
        subj_label = _pick(_pronoun_list, rng)
        neg_label = _pick(_neg_list, rng)
        verb_label = _pick_lexical(rng)

        extra_v = int(rng.integers(0, max(1, extra + 1)))

        toks, tags = [], []
        toks.append(_make_token(subj_label, GrammarRole.SUBJECT, rng))
        tags.append("B-SUBJ")
        toks.append(_make_token(neg_label, GrammarRole.NEGATION, rng))
        tags.append("B-NEG")
        t, g = _extend_group(verb_label, GrammarRole.VERB, "B-VERB", extra_v, rng)
        toks += t; tags += g
        return toks, tags

    def _pattern_topic(rng: np.random.Generator, extra: int) -> tuple[list[SignToken], list[str]]:
        """Topique : [TOPIC, SUBJECT+, PREDICATE+]"""
        topic_label = _pick_lexical(rng)
        subj_label = _pick(_pronoun_list, rng)
        verb_label = _pick_lexical(rng)

        extra_s = int(rng.integers(0, max(1, extra + 1)))

        toks, tags = [], []
        toks.append(_make_token(topic_label, GrammarRole.TOPIC, rng))
        tags.append("B-TOPIC")
        t, g = _extend_group(subj_label, GrammarRole.SUBJECT, "B-SUBJ", extra_s, rng)
        toks += t; tags += g
        t, g = _extend_group(verb_label, GrammarRole.VERB, "B-VERB", 0, rng)
        toks += t; tags += g
        return toks, tags

    _patterns = [_pattern_svo, _pattern_wh, _pattern_neg, _pattern_topic]
    _pattern_weights = np.array([0.45, 0.20, 0.20, 0.15])
    _pattern_weights /= _pattern_weights.sum()

    # ----------------------------------------------------------------
    # Génération des n_sequences séquences
    # ----------------------------------------------------------------
    all_sequences: list[list[SignToken]] = []
    all_labels: list[list[str]] = []

    for _ in range(n_sequences):
        # Longueur cible
        target_len = int(rng.integers(min_len, max_len + 1))

        # Choisir un pattern
        pattern_idx = int(rng.choice(len(_patterns), p=_pattern_weights))
        pattern_fn = _patterns[pattern_idx]

        # Nbre de tokens extra à distribuer dans le pattern
        # (le pattern de base crée ~3 tokens)
        extra = max(0, target_len - 3)

        tokens_seq, labels_seq = pattern_fn(rng, extra)

        # Tronquer ou compléter si nécessaire
        if len(tokens_seq) > max_len:
            tokens_seq = tokens_seq[:max_len]
            labels_seq = labels_seq[:max_len]
        elif len(tokens_seq) < min_len:
            # Compléter avec des tokens lexicaux neutres
            while len(tokens_seq) < min_len:
                tokens_seq.append(_make_token(_pick_lexical(rng), GrammarRole.UNKNOWN, rng))
                labels_seq.append("O")

        all_sequences.append(tokens_seq)
        all_labels.append(labels_seq)

    return all_sequences, all_labels


# ---------------------------------------------------------------------------
# Tagger principal
# ---------------------------------------------------------------------------


class LSFBSequenceTagger:
    """CRF pour tagger les séquences de signes LSFB avec des rôles BIO.

    Mode de fonctionnement :
    - Si sklearn-crfsuite est disponible et qu'un modèle est chargé via
      ``load()`` ou ``fit()``, utilise le CRF supervisé.
    - Sinon, repli sur les règles heuristiques ``LSFBGrammarRules``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        self.crf: object | None = None
        self.fallback_rules = LSFBGrammarRules()
        self._c1 = c1
        self._c2 = c2
        self._max_iterations = max_iterations

        if model_path is not None:
            try:
                self.load(model_path)
            except Exception as exc:
                logger.warning(
                    "lsfb_crf_load_failed_using_rules",
                    path=model_path,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(self, tokens: list[SignToken]) -> list[str]:
        """Prédit les tags BIO pour une séquence de tokens.

        Args:
            tokens: Séquence de SignToken.

        Returns:
            Liste de tags BIO (même longueur que ``tokens``).
        """
        if not tokens:
            return []

        if self.crf is not None and _CRF_AVAILABLE:
            try:
                feats = _sequence_features(tokens)
                tags: list[str] = self.crf.predict([feats])[0]  # type: ignore[union-attr]
                return tags
            except Exception as exc:
                logger.warning(
                    "crf_predict_failed_fallback_rules",
                    error=str(exc),
                )

        # Repli heuristique
        return self._predict_from_rules(tokens)

    def _predict_from_rules(self, tokens: list[SignToken]) -> list[str]:
        """Convertit les GrammarRole heuristiques en tags BIO.

        Appelle toujours ``tag_grammar_roles`` pour s'assurer que l'analyse
        NMM au niveau énoncé (question YN, négation) est bien effectuée.

        Note : ``tag_grammar_roles`` peut ajouter un token virtuel ``__QYN__``
        à la fin (pour signaler une question OUI/NON détectée via les NMM).
        Ce token est inclus dans les tags BIO retournés (marqué ``B-Q``).
        """
        tokens = self.fallback_rules.tag_grammar_roles(tokens)

        tags: list[str] = []
        prev_bio: str | None = None

        for tok in tokens:
            bio = _ROLE_TO_BIO.get(tok.grammar_role, "O")

            # Continuité I- : si le rôle BIO courant correspond au contexte
            if (
                bio.startswith("B-")
                and prev_bio is not None
                and prev_bio[2:] == bio[2:]
            ):
                # Même entité que le précédent → transformer B- en I-
                head = bio[2:]
                continuation = f"I-{head}"
                if continuation in GRAMMAR_TAGS:
                    bio = continuation

            tags.append(bio)
            prev_bio = bio

        return tags

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: list[list[SignToken]],
        labels: list[list[str]],
    ) -> "LSFBSequenceTagger":
        """Entraîne le CRF sur des données annotées.

        Args:
            sequences: Liste de séquences de SignToken.
            labels: Labels BIO correspondants (même structure).

        Returns:
            ``self`` (pour chaînage).

        Raises:
            ImportError: Si sklearn-crfsuite n'est pas installé.
        """
        if not _CRF_AVAILABLE:
            raise ImportError(
                "sklearn-crfsuite is required for CRF training. "
                "Install it with: pip install sklearn-crfsuite"
            )

        X = [_sequence_features(seq) for seq in sequences]
        y = labels

        crf = sklearn_crfsuite.CRF(  # type: ignore[name-defined]
            algorithm="lbfgs",
            c1=self._c1,
            c2=self._c2,
            max_iterations=self._max_iterations,
            all_possible_transitions=True,
        )
        crf.fit(X, y)
        self.crf = crf
        logger.info(
            "lsfb_crf_trained",
            num_sequences=len(sequences),
            classes=list(getattr(crf, "classes_", [])),
        )
        return self

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sequences: list[list[SignToken]],
        labels: list[list[str]],
    ) -> dict[str, float | str]:
        """Calcule precision, recall, f1 par classe et weighted avg.

        Utilise ``sklearn_crfsuite.metrics.flat_classification_report`` pour
        obtenir un rapport détaillé, puis extrait les métriques globales
        (weighted average) dans un dict.

        Args:
            sequences: Séquences de SignToken à évaluer.
            labels: Labels BIO de référence (même structure).

        Returns:
            Dict avec les clés :
            - ``"weighted_f1"`` (float)
            - ``"weighted_precision"`` (float)
            - ``"weighted_recall"`` (float)
            - ``"report_str"`` (str) : rapport complet multi-classe.

        Raises:
            RuntimeError: Si aucun modèle entraîné n'est disponible
                (``self.crf is None``).
            ImportError: Si sklearn-crfsuite n'est pas installé.
        """
        if self.crf is None:
            raise RuntimeError(
                "No trained CRF model available. Call fit() or load() first."
            )
        if not _CRF_AVAILABLE:
            raise ImportError(
                "sklearn-crfsuite is required for evaluation. "
                "Install it with: pip install sklearn-crfsuite"
            )

        try:
            from sklearn_crfsuite import metrics as crf_metrics  # type: ignore[import]
            from sklearn.metrics import classification_report  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sklearn and sklearn-crfsuite are required for evaluate(). "
                f"Original error: {exc}"
            ) from exc

        X = [_sequence_features(seq) for seq in sequences]
        y_pred = self.crf.predict(X)  # type: ignore[union-attr]
        y_true = labels

        # Flatten pour sklearn metrics
        y_true_flat = [tag for seq in y_true for tag in seq]
        y_pred_flat = [tag for seq in y_pred for tag in seq]

        # Rapport texte complet
        all_labels_present = sorted(set(y_true_flat) | set(y_pred_flat))
        report_str: str = classification_report(
            y_true_flat,
            y_pred_flat,
            labels=all_labels_present,
            zero_division=0,
        )

        # Métriques weighted avg via flat_classification_report ou sklearn
        try:
            from sklearn.metrics import precision_recall_fscore_support  # type: ignore[import]
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_flat,
                y_pred_flat,
                average="weighted",
                zero_division=0,
            )
            weighted_precision = float(prec)
            weighted_recall = float(rec)
            weighted_f1 = float(f1)
        except Exception:
            # Repli minimaliste si sklearn n'est pas dispo
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0

        logger.info(
            "lsfb_crf_evaluated",
            weighted_f1=weighted_f1,
            weighted_precision=weighted_precision,
            weighted_recall=weighted_recall,
            n_sequences=len(sequences),
        )

        return {
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "report_str": report_str,
        }

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle CRF entraîné via pickle.

        Args:
            path: Chemin de destination (typiquement ``*.crfsuite.pkl``).

        Raises:
            RuntimeError: Si aucun modèle n'est entraîné.
        """
        if self.crf is None:
            raise RuntimeError("No trained CRF model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self.crf, fh)
        logger.info("lsfb_crf_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Charge un modèle CRF depuis un fichier pickle.

        Args:
            path: Chemin vers le fichier ``.pkl`` sauvegardé par ``save()``.

        Raises:
            FileNotFoundError: Si le fichier est introuvable.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CRF model not found: {path}")
        with path.open("rb") as fh:
            self.crf = pickle.load(fh)
        logger.info("lsfb_crf_loaded", path=str(path))
