"""
Pipeline de traduction LSFB → Français.

Pipeline :

    SignToken[] ──► LSFBSequenceTagger (CRF) ──► LSFBGrammarRules ──► str français
                           │                              │
                    NMM features               Règles de surface
                    (32 dims/token)        (articles, copule, ponctuation)

    SignToken[] ──► Seq2SeqTranslator (encoder-decoder) ──► str français
                           │
                    Embedding + BiLSTM + Attention

Ce module orchestre les trois phases :

- ``'rules'``   : règles heuristiques uniquement (numpy, toujours disponible)
- ``'crf'``     : règles + CRF supervisé (si sklearn-crfsuite et modèle chargé)
- ``'seq2seq'`` : modèle encoder-decoder PyTorch (si torch disponible et modèle chargé)

La traduction peut être invoquée soit avec des ``SignToken`` complets
(incluant NMM et espace signé), soit via ``translate_buffer()`` qui accepte
le format natif du pipeline d'inférence.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from app.ml.grammar.lsfb_crf import GRAMMAR_TAGS, LSFBSequenceTagger
from app.ml.grammar.lsfb_rules import (
    GrammarRole,
    LSFBGrammarRules,
    NMMAnalyzer,
    SignToken,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import conditionnel PyTorch
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import LongTensor, Tensor

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Résultat de traduction
# ---------------------------------------------------------------------------


@dataclass
class TranslationResult:
    """Résultat complet d'une traduction LSFB → Français."""

    text: str                    # Phrase française produite
    confidence: float            # Confiance agrégée (moyenne des tokens)
    grammar_tags: list[str]      # Tags BIO CRF (pour debug/visualisation)
    mode_used: str               # 'rules' | 'crf' | 'seq2seq'
    debug_info: dict             # Informations de debug (tokens, rôles, NMM…)


# ---------------------------------------------------------------------------
# Vocabulaire Seq2Seq
# ---------------------------------------------------------------------------


class Seq2SeqVocab:
    """Vocabulaire bidirectionnel (token ↔ index) avec tokens spéciaux.

    Tokens spéciaux réservés :
    - PAD = 0  : padding
    - BOS = 1  : début de séquence (Begin Of Sentence)
    - EOS = 2  : fin de séquence (End Of Sentence)
    - UNK = 3  : token inconnu
    """

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3

    _SPECIAL_TOKENS: list[str] = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self) -> None:
        self._token2idx: dict[str, int] = {}
        self._idx2token: dict[int, str] = {}
        # Initialiser les tokens spéciaux
        for idx, tok in enumerate(self._SPECIAL_TOKENS):
            self._token2idx[tok] = idx
            self._idx2token[idx] = tok

    def build_from_sequences(self, sequences: list[list[str]]) -> None:
        """Construit le vocabulaire depuis les séquences.

        Les tokens spéciaux PAD/BOS/EOS/UNK sont toujours aux indices 0-3.
        Les tokens lexicaux sont ajoutés dans l'ordre d'apparition.

        Args:
            sequences: Liste de séquences de tokens.
        """
        # Réinitialiser (garder les spéciaux)
        self._token2idx = {tok: idx for idx, tok in enumerate(self._SPECIAL_TOKENS)}
        self._idx2token = {idx: tok for idx, tok in enumerate(self._SPECIAL_TOKENS)}

        # Compter et trier pour la reproductibilité
        counter: Counter[str] = Counter()
        for seq in sequences:
            counter.update(seq)

        next_idx = len(self._SPECIAL_TOKENS)
        for token, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if token not in self._token2idx:
                self._token2idx[token] = next_idx
                self._idx2token[next_idx] = token
                next_idx += 1

    def encode(self, tokens: list[str]) -> list[int]:
        """Convertit une liste de tokens en indices.

        Les tokens inconnus sont remplacés par UNK.

        Args:
            tokens: Liste de tokens à encoder.

        Returns:
            Liste d'indices correspondants.
        """
        unk = self.UNK
        return [self._token2idx.get(t, unk) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """Convertit des indices en tokens (filtre PAD/BOS/EOS).

        Args:
            indices: Liste d'indices à décoder.

        Returns:
            Liste de tokens (sans tokens spéciaux de contrôle).
        """
        skip = {self.PAD, self.BOS, self.EOS}
        return [
            self._idx2token.get(i, "<UNK>")
            for i in indices
            if i not in skip
        ]

    def __len__(self) -> int:
        return len(self._token2idx)

    def save(self, path: str | Path) -> None:
        """Sérialise le vocabulaire en JSON.

        Args:
            path: Chemin du fichier de sortie.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token2idx": self._token2idx,
            "idx2token": {str(k): v for k, v in self._idx2token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Seq2SeqVocab":
        """Charge un vocabulaire depuis un fichier JSON.

        Args:
            path: Chemin du fichier JSON.

        Returns:
            Instance de Seq2SeqVocab reconstruite.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab._token2idx = data["token2idx"]
        vocab._idx2token = {int(k): v for k, v in data["idx2token"].items()}
        return vocab


# ---------------------------------------------------------------------------
# Modèle Seq2Seq (encoder-decoder avec attention)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class Seq2SeqTranslator(nn.Module):  # type: ignore[misc]
        """Encoder-decoder pour LSFB → Français.

        Architecture :
        - Encoder : Embedding(src_vocab_size, embed_dim) + BiLSTM(embed_dim,
          hidden_dim, num_layers=2)
        - Decoder : Embedding(tgt_vocab_size, embed_dim) + LSTM(embed_dim,
          hidden_dim*2) + Linear(hidden_dim*2, tgt_vocab_size)
        - Attention : dot-product attention encoder→decoder (Luong-style)

        Tokens spéciaux : PAD=0, BOS=1, EOS=2, UNK=3
        """

        def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
            max_decode_len: int = 30,
        ) -> None:
            super().__init__()
            self.src_vocab_size = src_vocab_size
            self.tgt_vocab_size = tgt_vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.max_decode_len = max_decode_len

            # Encoder
            self.src_embedding = nn.Embedding(
                src_vocab_size, embed_dim, padding_idx=Seq2SeqVocab.PAD
            )
            self.encoder_lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Decoder
            self.tgt_embedding = nn.Embedding(
                tgt_vocab_size, embed_dim, padding_idx=Seq2SeqVocab.PAD
            )
            # Le décodeur prend embed_dim + hidden_dim*2 (attention context)
            self.decoder_lstm = nn.LSTM(
                embed_dim + hidden_dim * 2,
                hidden_dim * 2,
                num_layers=1,
                batch_first=True,
            )
            self.output_projection = nn.Linear(hidden_dim * 2, tgt_vocab_size)
            self.dropout = nn.Dropout(dropout)

            # Projection pour ramener l'état caché BiLSTM (hidden_dim*2 par layer)
            # vers hidden_dim*2 pour le décodeur LSTM (1 layer)
            self.encoder_hidden_proj = nn.Linear(hidden_dim * 2 * num_layers, hidden_dim * 2)
            self.encoder_cell_proj = nn.Linear(hidden_dim * 2 * num_layers, hidden_dim * 2)

        def encode(
            self,
            src: "LongTensor",
            src_lengths: "LongTensor",
        ) -> "tuple[Tensor, tuple]":
            """Encode la séquence source.

            Args:
                src: Tensor (B, S) d'indices sources.
                src_lengths: Tensor (B,) de longueurs réelles.

            Returns:
                Tuple (encoder_output, (h_n, c_n)) où encoder_output est de
                forme (B, S, hidden_dim*2).
            """
            embedded = self.dropout(self.src_embedding(src))  # (B, S, embed_dim)

            # Pack pour masquer le padding
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, (h_n, c_n) = self.encoder_lstm(packed)
            encoder_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
            # encoder_output: (B, S, hidden_dim*2)

            return encoder_output, (h_n, c_n)

        def _init_decoder_hidden(
            self,
            h_n: "Tensor",
            c_n: "Tensor",
        ) -> "tuple[Tensor, Tensor]":
            """Projette l'état caché encoder vers l'état initial du décodeur.

            h_n et c_n sont de forme (num_layers*2, B, hidden_dim) car BiLSTM.
            On concatène tous les états et on projette.

            Args:
                h_n: État caché final de l'encodeur (num_layers*2, B, hidden_dim).
                c_n: État de cellule final de l'encodeur (num_layers*2, B, hidden_dim).

            Returns:
                Tuple (h0, c0) pour le décodeur, chacun de forme (1, B, hidden_dim*2).
            """
            B = h_n.size(1)
            # (num_layers*2, B, hidden_dim) → (B, num_layers*2*hidden_dim)
            h_cat = h_n.permute(1, 0, 2).contiguous().view(B, -1)
            c_cat = c_n.permute(1, 0, 2).contiguous().view(B, -1)

            h0 = torch.tanh(self.encoder_hidden_proj(h_cat)).unsqueeze(0)  # (1, B, H*2)
            c0 = torch.tanh(self.encoder_cell_proj(c_cat)).unsqueeze(0)    # (1, B, H*2)
            return h0, c0

        def _attention(
            self,
            decoder_hidden: "Tensor",
            encoder_output: "Tensor",
        ) -> "Tensor":
            """Calcule le contexte via dot-product attention (Luong-style).

            Args:
                decoder_hidden: État caché du décodeur (B, 1, H*2).
                encoder_output: Sorties de l'encodeur (B, S, H*2).

            Returns:
                Vecteur de contexte (B, 1, H*2).
            """
            # scores: (B, 1, S)
            scores = torch.bmm(decoder_hidden, encoder_output.transpose(1, 2))
            attn_weights = F.softmax(scores, dim=-1)  # (B, 1, S)
            context = torch.bmm(attn_weights, encoder_output)  # (B, 1, H*2)
            return context

        def decode_step(
            self,
            input_token: "LongTensor",
            hidden: "tuple[Tensor, Tensor]",
            encoder_output: "Tensor",
        ) -> "tuple[Tensor, tuple[Tensor, Tensor]]":
            """Un step de décodage avec attention.

            Args:
                input_token: Tensor (B,) d'indices cibles.
                hidden: Tuple (h, c) état caché du décodeur.
                encoder_output: Sorties de l'encodeur (B, S, H*2).

            Returns:
                Tuple (logits (B, tgt_vocab_size), new_hidden).
            """
            # (B,) → (B, 1) → (B, 1, embed_dim)
            embedded = self.dropout(
                self.tgt_embedding(input_token.unsqueeze(1))
            )  # (B, 1, embed_dim)

            h, c = hidden
            # decoder_hidden: (B, 1, H*2)
            decoder_hidden = h.permute(1, 0, 2)  # (B, 1, H*2) — 1 layer

            context = self._attention(decoder_hidden, encoder_output)  # (B, 1, H*2)

            # Concaténer embedding et contexte
            lstm_input = torch.cat([embedded, context], dim=-1)  # (B, 1, embed_dim + H*2)

            output, new_hidden = self.decoder_lstm(lstm_input, (h, c))
            # output: (B, 1, H*2)

            logits = self.output_projection(output.squeeze(1))  # (B, tgt_vocab_size)
            return logits, new_hidden

        def forward(
            self,
            src: "LongTensor",
            tgt: "LongTensor",
            src_lengths: "LongTensor",
        ) -> "Tensor":
            """Passe avant avec teacher forcing.

            Args:
                src: Tensor (B, S) d'indices sources.
                tgt: Tensor (B, T) d'indices cibles (shifted right, commence par BOS).
                src_lengths: Tensor (B,) de longueurs sources réelles.

            Returns:
                Logits (B, T, tgt_vocab_size) pour cross-entropy.
            """
            encoder_output, (h_n, c_n) = self.encode(src, src_lengths)
            h0, c0 = self._init_decoder_hidden(h_n, c_n)
            hidden = (h0, c0)

            T = tgt.size(1)
            all_logits: list["Tensor"] = []

            for t in range(T):
                logits, hidden = self.decode_step(tgt[:, t], hidden, encoder_output)
                all_logits.append(logits.unsqueeze(1))

            return torch.cat(all_logits, dim=1)  # (B, T, tgt_vocab_size)

        def translate(
            self,
            src: "LongTensor",
            src_lengths: "LongTensor",
            beam_size: int = 1,
        ) -> list[int]:
            """Décode une séquence source vers des indices cibles.

            Args:
                src: Tensor (1, S) d'indices sources.
                src_lengths: Tensor (1,) de longueur source.
                beam_size: Taille du faisceau (1 = greedy, >1 = beam search).

            Returns:
                Liste d'indices cibles (sans BOS/EOS/PAD).
            """
            self.eval()
            with torch.no_grad():
                if beam_size <= 1:
                    return self._greedy_decode(src, src_lengths)
                else:
                    return self._beam_search(src, src_lengths, beam_size)

        def _greedy_decode(
            self,
            src: "LongTensor",
            src_lengths: "LongTensor",
        ) -> list[int]:
            """Décodage glouton (argmax à chaque step).

            Args:
                src: Tensor (1, S).
                src_lengths: Tensor (1,).

            Returns:
                Indices décodés (sans spéciaux).
            """
            encoder_output, (h_n, c_n) = self.encode(src, src_lengths)
            h0, c0 = self._init_decoder_hidden(h_n, c_n)
            hidden = (h0, c0)

            input_token = torch.tensor(
                [Seq2SeqVocab.BOS], device=src.device, dtype=torch.long
            )
            result: list[int] = []

            for _ in range(self.max_decode_len):
                logits, hidden = self.decode_step(input_token, hidden, encoder_output)
                next_token = logits.argmax(dim=-1)  # (1,)
                idx = next_token.item()
                if idx == Seq2SeqVocab.EOS:
                    break
                if idx not in (Seq2SeqVocab.PAD, Seq2SeqVocab.BOS):
                    result.append(idx)
                input_token = next_token

            return result

        def _beam_search(
            self,
            src: "LongTensor",
            src_lengths: "LongTensor",
            beam_size: int,
        ) -> list[int]:
            """Beam search simplifié.

            Args:
                src: Tensor (1, S).
                src_lengths: Tensor (1,).
                beam_size: Taille du faisceau.

            Returns:
                Indices décodés de la meilleure hypothèse (sans spéciaux).
            """
            encoder_output, (h_n, c_n) = self.encode(src, src_lengths)
            h0, c0 = self._init_decoder_hidden(h_n, c_n)

            # Répliquer l'état pour beam_size hypothèses
            encoder_output_exp = encoder_output.expand(beam_size, -1, -1)
            h = h0.expand(-1, beam_size, -1).contiguous()
            c = c0.expand(-1, beam_size, -1).contiguous()

            # Chaque hypothèse est (score, tokens, hidden)
            beam: list[tuple[float, list[int], tuple]] = [
                (0.0, [], (h[:, :1, :].contiguous(), c[:, :1, :].contiguous()))
            ]

            completed: list[tuple[float, list[int]]] = []

            for _ in range(self.max_decode_len):
                candidates: list[tuple[float, list[int], tuple]] = []
                for score, tokens, (bh, bc) in beam:
                    last = tokens[-1] if tokens else Seq2SeqVocab.BOS
                    input_tok = torch.tensor([last], device=src.device, dtype=torch.long)
                    enc_out = encoder_output  # (1, S, H*2)
                    logits, new_hidden = self.decode_step(input_tok, (bh, bc), enc_out)
                    log_probs = F.log_softmax(logits[0], dim=-1)  # (vocab,)

                    topk_log_probs, topk_idxs = log_probs.topk(beam_size)
                    for lp, idx in zip(topk_log_probs.tolist(), topk_idxs.tolist()):
                        new_score = score + lp
                        new_tokens = tokens + [idx]
                        if idx == Seq2SeqVocab.EOS:
                            completed.append((new_score, tokens))
                        else:
                            nh = new_hidden[0]
                            nc = new_hidden[1]
                            candidates.append((new_score, new_tokens, (nh, nc)))

                if not candidates:
                    break

                # Garder les beam_size meilleures hypothèses
                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = candidates[:beam_size]

            if not completed:
                # Fallback : utiliser la meilleure hypothèse incomplète
                beam.sort(key=lambda x: x[0], reverse=True)
                best_tokens = beam[0][1] if beam else []
            else:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_tokens = completed[0][1]

            return [
                t for t in best_tokens
                if t not in (Seq2SeqVocab.PAD, Seq2SeqVocab.BOS, Seq2SeqVocab.EOS)
            ]

else:
    # Stub si torch non disponible
    class Seq2SeqTranslator:  # type: ignore[no-redef]
        """Stub utilisé quand PyTorch n'est pas installé."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "PyTorch est requis pour Seq2SeqTranslator. "
                "Installez torch : pip install torch"
            )


# ---------------------------------------------------------------------------
# Trainer Seq2Seq
# ---------------------------------------------------------------------------


class Seq2SeqTrainer:
    """Entraîne Seq2SeqTranslator avec teacher forcing.

    Utilise Adam + cross-entropy avec masquage du padding.
    Supporte l'early stopping sur la val loss.
    """

    def __init__(
        self,
        model: "Seq2SeqTranslator",
        src_vocab: Seq2SeqVocab,
        tgt_vocab: Seq2SeqVocab,
        *,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """Initialise le trainer.

        Args:
            model: Modèle Seq2SeqTranslator à entraîner.
            src_vocab: Vocabulaire source.
            tgt_vocab: Vocabulaire cible.
            lr: Learning rate pour Adam.
            device: Device PyTorch ('cpu', 'cuda', 'mps').
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch est requis pour Seq2SeqTrainer. "
                "Installez torch : pip install torch"
            )
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Seq2SeqVocab.PAD)

    def _prepare_batch(
        self,
        pairs: list[tuple[list[str], list[str]]],
    ) -> "tuple[Tensor, Tensor, LongTensor, Tensor, Tensor]":
        """Prépare un batch depuis des paires de tokens.

        Args:
            pairs: Liste de (src_tokens, tgt_tokens).

        Returns:
            Tuple (src_padded, src_lengths, tgt_input, tgt_target) où :
            - src_padded: (B, S_max)
            - src_lengths: (B,)
            - tgt_input: (B, T_max) — BOS + tgt[:-1]
            - tgt_target: (B, T_max) — tgt + EOS
        """
        src_seqs = [self.src_vocab.encode(s) for s, _ in pairs]
        tgt_seqs = [self.tgt_vocab.encode(t) for _, t in pairs]

        src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
        max_src = max(len(s) for s in src_seqs)
        max_tgt = max(len(t) for t in tgt_seqs) + 1  # +1 pour BOS/EOS

        B = len(pairs)
        src_padded = torch.full((B, max_src), Seq2SeqVocab.PAD, dtype=torch.long)
        tgt_input = torch.full((B, max_tgt), Seq2SeqVocab.PAD, dtype=torch.long)
        tgt_target = torch.full((B, max_tgt), Seq2SeqVocab.PAD, dtype=torch.long)

        for i, (src_idx, tgt_idx) in enumerate(zip(src_seqs, tgt_seqs)):
            src_padded[i, : len(src_idx)] = torch.tensor(src_idx, dtype=torch.long)
            # tgt_input : BOS + tgt
            tgt_full = [Seq2SeqVocab.BOS] + tgt_idx + [Seq2SeqVocab.EOS]
            tgt_in = tgt_full[:-1]
            tgt_out = tgt_full[1:]
            T = len(tgt_in)
            tgt_input[i, :T] = torch.tensor(tgt_in, dtype=torch.long)
            tgt_target[i, :T] = torch.tensor(tgt_out, dtype=torch.long)

        return (
            src_padded.to(self.device),
            src_lengths.to(self.device),
            tgt_input.to(self.device),
            tgt_target.to(self.device),
        )

    def train_epoch(
        self,
        pairs: list[tuple[list[str], list[str]]],
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Entraîne sur une époque complète.

        Utilise teacher forcing. Calcule la cross-entropy en ignorant le PAD.

        Args:
            pairs: Liste de (src_tokens, tgt_tokens).
            batch_size: Taille des mini-batchs.

        Returns:
            Dictionnaire {"loss": float}.
        """
        self.model.train()
        shuffled = list(pairs)
        random.shuffle(shuffled)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            if not batch:
                continue

            src_padded, src_lengths, tgt_input, tgt_target = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            logits = self.model(src_padded, tgt_input, src_lengths)
            # logits: (B, T, vocab)  →  (B*T, vocab)
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                tgt_target.reshape(B * T),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": round(avg_loss, 6)}

    def evaluate(
        self,
        pairs: list[tuple[list[str], list[str]]],
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Évalue le modèle sur des paires de validation.

        Calcule la loss et le BLEU-1 approximé (précision des unigrammes).

        Args:
            pairs: Liste de (src_tokens, tgt_tokens).
            batch_size: Taille des mini-batchs.

        Returns:
            Dictionnaire {"loss": float, "bleu1": float}.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Pour BLEU-1
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                if not batch:
                    continue

                src_padded, src_lengths, tgt_input, tgt_target = self._prepare_batch(batch)
                logits = self.model(src_padded, tgt_input, src_lengths)
                B, T, V = logits.shape
                loss = self.criterion(
                    logits.reshape(B * T, V),
                    tgt_target.reshape(B * T),
                )
                total_loss += loss.item()
                n_batches += 1

                # BLEU-1 : comparer les prédictions greedy avec les références
                for j, (src_t, tgt_t) in enumerate(batch):
                    src_single = src_padded[j : j + 1]
                    src_len_single = src_lengths[j : j + 1]
                    pred_indices = self.model.translate(src_single, src_len_single)
                    pred_tokens = self.tgt_vocab.decode(pred_indices)
                    ref_tokens = tgt_t

                    # Précision unigramme (BLEU-1 approximé)
                    ref_counter = Counter(ref_tokens)
                    pred_counter = Counter(pred_tokens)
                    for tok, cnt in pred_counter.items():
                        clipped = min(cnt, ref_counter.get(tok, 0))
                        n_correct += clipped
                        n_total += cnt

        avg_loss = total_loss / max(n_batches, 1)
        bleu1 = n_correct / max(n_total, 1)

        return {"loss": round(avg_loss, 6), "bleu1": round(bleu1, 4)}

    def fit(
        self,
        train_pairs: list[tuple[list[str], list[str]]],
        val_pairs: list[tuple[list[str], list[str]]],
        *,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 7,
    ) -> list[dict]:
        """Entraîne avec early stopping sur la val loss.

        Args:
            train_pairs: Paires d'entraînement (src_tokens, tgt_tokens).
            val_pairs: Paires de validation (src_tokens, tgt_tokens).
            epochs: Nombre maximum d'époques.
            batch_size: Taille des mini-batchs.
            patience: Nombre d'époques sans amélioration avant arrêt.

        Returns:
            Historique des métriques par époque (liste de dicts).
        """
        history: list[dict] = []
        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_pairs, batch_size=batch_size)
            val_metrics = self.evaluate(val_pairs, batch_size=batch_size)

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_bleu1": val_metrics["bleu1"],
            }
            history.append(entry)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_bleu1=%.4f",
                epoch,
                epochs,
                entry["train_loss"],
                entry["val_loss"],
                entry["val_bleu1"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stopping à l'époque %d (patience=%d).", epoch, patience
                    )
                    break

        # Restaurer le meilleur état
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle, ses hyperparamètres et les vocabulaires.

        Args:
            path: Chemin du fichier de sauvegarde (.pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": {
                k: v.cpu() for k, v in self.model.state_dict().items()
            },
            "model_config": {
                "src_vocab_size": self.model.src_vocab_size,
                "tgt_vocab_size": self.model.tgt_vocab_size,
                "embed_dim": self.model.embed_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "max_decode_len": self.model.max_decode_len,
            },
            "src_vocab_token2idx": self.src_vocab._token2idx,
            "src_vocab_idx2token": {
                str(k): v for k, v in self.src_vocab._idx2token.items()
            },
            "tgt_vocab_token2idx": self.tgt_vocab._token2idx,
            "tgt_vocab_idx2token": {
                str(k): v for k, v in self.tgt_vocab._idx2token.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Seq2SeqTrainer sauvegardé → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "Seq2SeqTrainer":
        """Charge un trainer sauvegardé depuis un fichier.

        Args:
            path: Chemin du fichier .pkl.
            device: Device PyTorch cible.

        Returns:
            Instance de Seq2SeqTrainer reconstruite.

        Raises:
            ImportError: Si PyTorch n'est pas disponible.
            FileNotFoundError: Si le fichier n'existe pas.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch est requis pour charger Seq2SeqTrainer. "
                "Installez torch : pip install torch"
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Reconstruire les vocabulaires
        src_vocab = Seq2SeqVocab()
        src_vocab._token2idx = payload["src_vocab_token2idx"]
        src_vocab._idx2token = {
            int(k): v for k, v in payload["src_vocab_idx2token"].items()
        }

        tgt_vocab = Seq2SeqVocab()
        tgt_vocab._token2idx = payload["tgt_vocab_token2idx"]
        tgt_vocab._idx2token = {
            int(k): v for k, v in payload["tgt_vocab_idx2token"].items()
        }

        # Reconstruire le modèle
        cfg = payload["model_config"]
        model = Seq2SeqTranslator(**cfg)
        model.load_state_dict(payload["model_state_dict"])

        trainer = cls(model, src_vocab, tgt_vocab, device=device)
        return trainer


# ---------------------------------------------------------------------------
# Générateur de corpus synthétique
# ---------------------------------------------------------------------------

# Vocabulaires LSFB synthétiques
_LSFB_PRONOUNS = ["MOI", "TOI", "LUI", "ELLE", "NOUS", "VOUS", "EUX"]
_LSFB_WH_SIGNS = ["QUOI", "POURQUOI", "OÙ", "QUAND", "COMMENT", "COMBIEN"]
_LSFB_NEGATION_SIGNS = ["NEG", "PAS", "JAMAIS"]
_LSFB_LEXICAL = [
    "JEAN", "MARIE", "MANGER", "BOIRE", "VOIR", "ALLER", "VOULOIR",
    "COMPRENDRE", "FAIRE", "POUVOIR", "AVOIR", "DONNER", "PRENDRE", "VENIR",
    "PAIN", "EAU", "MAISON", "ÉCOLE", "TRAVAIL", "DEMAIN", "AUJOURD'HUI", "HIER",
]

# Tables de traduction simples pour les templates
_PRONOUN_FR: dict[str, str] = {
    "MOI": "Je",
    "TOI": "Tu",
    "LUI": "Il",
    "ELLE": "Elle",
    "NOUS": "Nous",
    "VOUS": "Vous",
    "EUX": "Ils",
}
_VERB_FR: dict[str, str] = {
    "MANGER": "mange",
    "BOIRE": "bois",
    "VOIR": "vois",
    "ALLER": "vais",
    "VOULOIR": "veux",
    "COMPRENDRE": "comprends",
    "FAIRE": "fais",
    "POUVOIR": "peux",
    "AVOIR": "ai",
    "DONNER": "donne",
    "PRENDRE": "prends",
    "VENIR": "viens",
}
_OBJ_FR: dict[str, list[str]] = {
    "PAIN": ["le", "pain"],
    "EAU": ["l'", "eau"],
    "MAISON": ["la", "maison"],
    "ÉCOLE": ["l'", "école"],
    "TRAVAIL": ["le", "travail"],
    "JEAN": ["Jean"],
    "MARIE": ["Marie"],
}
_WH_FR: dict[str, list[str]] = {
    "QUOI": ["Qu'est-ce", "que"],
    "POURQUOI": ["Pourquoi"],
    "OÙ": ["Où"],
    "QUAND": ["Quand"],
    "COMMENT": ["Comment"],
    "COMBIEN": ["Combien"],
}
_TEMPORAL_FR: dict[str, list[str]] = {
    "DEMAIN": ["Demain", ","],
    "AUJOURD'HUI": ["Aujourd'hui", ","],
    "HIER": ["Hier", ","],
}


def generate_seq2seq_corpus(
    n_pairs: int = 500,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Génère des paires synthétiques (séquence LSFB, phrase française).

    Templates de phrases :
    - SVO : [PRON, VERB, OBJ] → ["Je", "mange", "le", "pain"]
    - WH question : [WH, PRON, VERB] → ["Qu'est-ce", "que", "tu", "veux", "?"]
    - Négation : [PRON, NEG, VERB] → ["Je", "ne", "comprends", "pas"]
    - Topique temporel : [TEMP, PRON, VERB, OBJ] → ["Demain", ",", "je", "vais", "à", "l'école"]

    Args:
        n_pairs: Nombre de paires à générer.
        seed: Graine pour la reproductibilité (numpy).

    Returns:
        Liste de (src_tokens: list[str], tgt_tokens: list[str]).
    """
    rng = np.random.default_rng(seed)
    pairs: list[tuple[list[str], list[str]]] = []

    # Listes des composants
    pronouns = _LSFB_PRONOUNS
    wh_signs = _LSFB_WH_SIGNS
    neg_signs = _LSFB_NEGATION_SIGNS
    verbs = [v for v in _LSFB_LEXICAL if v in _VERB_FR]
    objects = [o for o in _LSFB_LEXICAL if o in _OBJ_FR]
    temporal_signs = [t for t in _LSFB_LEXICAL if t in _TEMPORAL_FR]

    templates = ["svo", "wh", "neg", "topique"]
    template_weights = [0.4, 0.2, 0.2, 0.2]
    template_cumsum = list(np.cumsum(template_weights))

    def pick_template() -> str:
        r = rng.random()
        for w, t in zip(template_cumsum, templates):
            if r < w:
                return t
        return "svo"

    def pick(lst: list) -> str:
        return lst[int(rng.integers(0, len(lst)))]

    for _ in range(n_pairs):
        template = pick_template()

        if template == "svo":
            pron = pick(pronouns)
            verb = pick(verbs)
            obj = pick(objects)
            src = [pron, verb, obj]
            pron_fr = _PRONOUN_FR.get(pron, pron.lower())
            verb_fr = _VERB_FR.get(verb, verb.lower())
            obj_fr = _OBJ_FR.get(obj, [obj.lower()])
            # Majuscule sur le pronom si en début de phrase
            tgt = [pron_fr.capitalize()] + [verb_fr] + obj_fr + ["."]

        elif template == "wh":
            wh = pick(wh_signs)
            pron = pick(pronouns)
            verb = pick(verbs)
            src = [wh, pron, verb]
            wh_fr = _WH_FR.get(wh, [wh.lower()])
            pron_fr = _PRONOUN_FR.get(pron, pron.lower()).lower()
            verb_fr = _VERB_FR.get(verb, verb.lower())
            tgt = wh_fr + [pron_fr, verb_fr, "?"]

        elif template == "neg":
            pron = pick(pronouns)
            neg = pick(neg_signs)
            verb = pick(verbs)
            src = [pron, neg, verb]
            pron_fr = _PRONOUN_FR.get(pron, pron.lower())
            verb_fr = _VERB_FR.get(verb, verb.lower())
            tgt = [pron_fr.capitalize(), "ne", verb_fr, "pas", "."]

        else:  # topique temporel
            temp = pick(temporal_signs)
            pron = pick(pronouns)
            verb = pick(verbs)
            obj = pick(objects)
            src = [temp, pron, verb, obj]
            temp_fr = _TEMPORAL_FR.get(temp, [temp.lower(), ","])
            pron_fr = _PRONOUN_FR.get(pron, pron.lower()).lower()
            verb_fr = _VERB_FR.get(verb, verb.lower())
            obj_fr = _OBJ_FR.get(obj, [obj.lower()])
            tgt = temp_fr + [pron_fr, verb_fr] + obj_fr + ["."]

        pairs.append((src, tgt))

    return pairs


# ---------------------------------------------------------------------------
# Traducteur principal
# ---------------------------------------------------------------------------


class LSFBToFrenchTranslator:
    """Traducteur complet LSFB → Français.

    Modes disponibles :
    - ``'rules'``   : Heuristiques uniquement (pas de ML, toujours dispo).
    - ``'crf'``     : CRF supervisé + heuristiques (sklearn-crfsuite requis).
    - ``'seq2seq'`` : Modèle encoder-decoder PyTorch (torch requis + modèle chargé).
                      Repli automatique sur ``'crf'`` puis ``'rules'`` si non disponible.
    """

    def __init__(
        self,
        mode: str = "rules",
        crf_model_path: str | None = None,
        seq2seq_model_path: str | None = None,
    ) -> None:
        """Initialise le traducteur.

        Args:
            mode: Stratégie de traduction parmi ``'rules'``, ``'crf'``,
                ``'seq2seq'``.  Un mode non disponible se replie gracieusement
                vers la stratégie la plus simple disponible.
            crf_model_path: Chemin vers un modèle CRF pré-entraîné (pickle).
                Si ``None``, le tagger CRF est créé sans modèle (repli règles).
            seq2seq_model_path: Chemin vers un Seq2SeqTrainer sauvegardé (pickle).
                Si ``None``, le mode seq2seq n'est pas disponible.
        """
        self.nmm_analyzer = NMMAnalyzer()
        self.grammar_rules = LSFBGrammarRules(nmm_analyzer=self.nmm_analyzer)
        self.sequence_tagger = LSFBSequenceTagger(
            model_path=crf_model_path,
        )
        self._requested_mode = mode

        # Charger le trainer Seq2Seq si un chemin est fourni
        self._seq2seq_trainer: Seq2SeqTrainer | None = None
        self._seq2seq_model: "Seq2SeqTranslator | None" = None
        self._seq2seq_src_vocab: Seq2SeqVocab | None = None
        self._seq2seq_tgt_vocab: Seq2SeqVocab | None = None

        if seq2seq_model_path is not None:
            if not _TORCH_AVAILABLE:
                logger.warning(
                    "lsfb_translator_seq2seq_torch_unavailable: "
                    "PyTorch requis pour seq2seq — mode déclassé."
                )
            else:
                try:
                    self._seq2seq_trainer = Seq2SeqTrainer.load(seq2seq_model_path)
                    self._seq2seq_model = self._seq2seq_trainer.model
                    self._seq2seq_src_vocab = self._seq2seq_trainer.src_vocab
                    self._seq2seq_tgt_vocab = self._seq2seq_trainer.tgt_vocab
                    logger.info(
                        "lsfb_translator_seq2seq_loaded: %s", seq2seq_model_path
                    )
                except Exception as exc:
                    logger.warning(
                        "lsfb_translator_seq2seq_load_failed: %s — mode déclassé.",
                        exc,
                    )

        self.mode = self._resolve_mode(mode)

        if self.mode != mode:
            logger.info(
                "lsfb_translator_mode_downgraded",
                requested=mode,
                effective=self.mode,
            )

    def _resolve_mode(self, requested: str) -> str:
        """Détermine le mode effectif en tenant compte des disponibilités."""
        if requested in ("rules",):
            return "rules"
        if requested == "crf":
            if self.sequence_tagger.crf is not None:
                return "crf"
            # CRF sans modèle chargé → règles (le tagger fera le repli)
            return "rules"
        if requested == "seq2seq":
            # Utiliser seq2seq si le trainer est chargé
            if self._seq2seq_trainer is not None:
                return "seq2seq"
            # Sinon repli gracieux
            if self.sequence_tagger.crf is not None:
                return "crf"
            return "rules"
        return "rules"

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def translate(
        self,
        sign_tokens: list[SignToken],
        nmm_sequence: list[np.ndarray] | None = None,
    ) -> TranslationResult:
        """Traduit une séquence de signes en français.

        Args:
            sign_tokens: Séquence de SignToken (label + confiance + NMM).
            nmm_sequence: Optionnel — séquence brute de vecteurs NMM par frame.
                Si fourni, les NMM moyens par signe sont réinjectés dans les
                tokens (utile si ``SignToken.nmm_features`` est vide).

        Returns:
            ``TranslationResult`` avec le texte, la confiance, les tags et
            les infos de debug.
        """
        if not sign_tokens:
            return TranslationResult(
                text="",
                confidence=0.0,
                grammar_tags=[],
                mode_used=self.mode,
                debug_info={"tokens": [], "reason": "empty_input"},
            )

        # Branche seq2seq
        if self.mode == "seq2seq" and self._seq2seq_model is not None:
            return self._translate_seq2seq(sign_tokens)

        # Optionnellement ré-injecter les NMM depuis la séquence brute
        tokens = self._enrich_nmm(sign_tokens, nmm_sequence)

        # Tagging BIO (CRF ou règles)
        # Note : le tagger en mode règles appelle tag_grammar_roles() en
        # interne, ce qui peut ajouter un token virtuel __QYN__ pour les
        # questions OUI/NON détectées via les NMM.
        grammar_tags = self.sequence_tagger.predict(tokens)

        # Reconstruire la liste de tokens en synchronisant avec grammar_tags
        # (peut avoir un token de plus si __QYN__ a été ajouté)
        if len(grammar_tags) == len(tokens) + 1 and grammar_tags[-1] == "B-Q":
            # Ajouter le token virtuel __QYN__ à la liste pour la cohérence
            import numpy as _np_local
            _qyn_tok = SignToken(
                label="__QYN__",
                confidence=1.0,
                nmm_features=_np_local.zeros(32, dtype=_np_local.float32),
                signing_space=_np_local.zeros(18, dtype=_np_local.float32),
                grammar_role=GrammarRole.QUESTION_YN,
            )
            tokens = list(tokens) + [_qyn_tok]

        # Mise à jour des rôles depuis les tags BIO
        tokens = self._apply_bio_tags(tokens, grammar_tags)

        # Traduction via les règles de surface (tokens déjà taggés)
        text = self.grammar_rules.tokens_to_french(tokens)

        # Confiance agrégée (ignorer le token virtuel __QYN__)
        real_tokens = [t for t in tokens if t.label != "__QYN__"]
        confidence = float(np.mean([t.confidence for t in real_tokens])) if real_tokens else 0.0

        debug_info = {
            "tokens": [
                {
                    "label": t.label,
                    "confidence": round(t.confidence, 3),
                    "grammar_role": t.grammar_role.value,
                    "nmm_has_data": bool(t.nmm_features.any()),
                }
                for t in tokens
                if t.label != "__QYN__"
            ],
            "grammar_tags": grammar_tags,
            "mode": self.mode,
        }

        # Déterminer le mode effectivement utilisé
        mode_used = "crf" if (self.mode == "crf" and self.sequence_tagger.crf is not None) else "rules"

        return TranslationResult(
            text=text,
            confidence=round(confidence, 3),
            grammar_tags=grammar_tags,
            mode_used=mode_used,
            debug_info=debug_info,
        )

    def _translate_seq2seq(self, sign_tokens: list[SignToken]) -> TranslationResult:
        """Traduit en utilisant le modèle encoder-decoder seq2seq.

        Args:
            sign_tokens: Séquence de SignToken.

        Returns:
            TranslationResult avec mode_used='seq2seq'.
        """
        assert self._seq2seq_model is not None
        assert self._seq2seq_src_vocab is not None
        assert self._seq2seq_tgt_vocab is not None

        src_labels = [t.label for t in sign_tokens]
        src_indices = self._seq2seq_src_vocab.encode(src_labels)

        if not src_indices:
            return TranslationResult(
                text="",
                confidence=0.0,
                grammar_tags=[],
                mode_used="seq2seq",
                debug_info={"tokens": src_labels, "reason": "empty_encoded_src"},
            )

        src_tensor = torch.tensor([src_indices], dtype=torch.long)
        src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)

        device = next(self._seq2seq_model.parameters()).device
        src_tensor = src_tensor.to(device)
        src_lengths = src_lengths.to(device)

        pred_indices = self._seq2seq_model.translate(src_tensor, src_lengths)
        pred_tokens = self._seq2seq_tgt_vocab.decode(pred_indices)
        text = " ".join(pred_tokens)

        confidence = float(np.mean([t.confidence for t in sign_tokens]))

        debug_info = {
            "src_tokens": src_labels,
            "pred_tokens": pred_tokens,
            "mode": "seq2seq",
        }

        return TranslationResult(
            text=text,
            confidence=round(confidence, 3),
            grammar_tags=[],
            mode_used="seq2seq",
            debug_info=debug_info,
        )

    def translate_buffer(self, buffer: list[dict]) -> str:
        """Traduit un buffer de prédictions brutes du pipeline d'inférence.

        Compatible avec le format de sortie de
        ``SignFlowInferencePipeline._build_prediction_response()``.

        Args:
            buffer: Liste de dicts au format::

                {
                    'label': str,
                    'confidence': float,
                    'nmm': list[float],          # optionnel, 32 dims
                    'signing_space': list[float], # optionnel, 18 dims
                    'start_frame': int,           # optionnel
                    'end_frame': int,             # optionnel
                }

        Returns:
            Phrase française traduite (chaîne vide si buffer vide).
        """
        if not buffer:
            return ""

        tokens: list[SignToken] = []
        for item in buffer:
            label = str(item.get("label") or item.get("sign") or "UNKNOWN").strip()
            if not label or label in ("NONE", "RECORDING"):
                continue

            confidence = float(item.get("confidence", 0.5))

            nmm_raw = item.get("nmm") or item.get("nmm_features") or []
            if nmm_raw:
                nmm = np.array(nmm_raw, dtype=np.float32)
                if nmm.shape[0] < 32:
                    pad = np.zeros(32, dtype=np.float32)
                    pad[: nmm.shape[0]] = nmm
                    nmm = pad
                elif nmm.shape[0] > 32:
                    nmm = nmm[:32]
            else:
                nmm = np.zeros(32, dtype=np.float32)

            ss_raw = item.get("signing_space") or []
            if ss_raw:
                ss = np.array(ss_raw, dtype=np.float32)
                if ss.shape[0] < 18:
                    pad = np.zeros(18, dtype=np.float32)
                    pad[: ss.shape[0]] = ss
                    ss = pad
                elif ss.shape[0] > 18:
                    ss = ss[:18]
            else:
                ss = np.zeros(18, dtype=np.float32)

            tokens.append(
                SignToken(
                    label=label,
                    confidence=confidence,
                    start_frame=int(item.get("start_frame", 0)),
                    end_frame=int(item.get("end_frame", 0)),
                    nmm_features=nmm,
                    signing_space=ss,
                )
            )

        if not tokens:
            return ""

        result = self.translate(tokens)
        return result.text

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    def _enrich_nmm(
        self,
        tokens: list[SignToken],
        nmm_sequence: list[np.ndarray] | None,
    ) -> list[SignToken]:
        """Réinjecte les NMM moyens par signe si la séquence brute est fournie.

        La séquence NMM brute est uniformément découpée en autant de segments
        qu'il y a de tokens, et les features moyennées sur chaque segment sont
        assignées au token correspondant.

        Args:
            tokens: Tokens avec NMM éventuellement nuls.
            nmm_sequence: Séquence de vecteurs NMM bruts (une entrée par frame).

        Returns:
            Copie des tokens avec NMM enrichis.
        """
        if not nmm_sequence:
            return tokens

        n_frames = len(nmm_sequence)
        n_tokens = len(tokens)

        enriched: list[SignToken] = []
        for i, tok in enumerate(tokens):
            # Segment de frames assigné à ce token
            start_f = int(i * n_frames / n_tokens)
            end_f = int((i + 1) * n_frames / n_tokens)
            segment = nmm_sequence[start_f:end_f]

            if segment:
                avg_nmm = self.nmm_analyzer.aggregate_sequence(segment)
            else:
                avg_nmm = tok.nmm_features.copy()

            enriched.append(
                SignToken(
                    label=tok.label,
                    confidence=tok.confidence,
                    start_frame=tok.start_frame,
                    end_frame=tok.end_frame,
                    nmm_features=avg_nmm,
                    signing_space=tok.signing_space.copy(),
                    grammar_role=tok.grammar_role,
                )
            )

        return enriched

    @staticmethod
    def _bio_to_role(bio_tag: str) -> GrammarRole:
        """Convertit un tag BIO en GrammarRole."""
        mapping: dict[str, GrammarRole] = {
            "B-SUBJ":  GrammarRole.SUBJECT,
            "I-SUBJ":  GrammarRole.SUBJECT,
            "B-VERB":  GrammarRole.VERB,
            "I-VERB":  GrammarRole.VERB,
            "B-OBJ":   GrammarRole.OBJECT,
            "I-OBJ":   GrammarRole.OBJECT,
            "B-TOPIC": GrammarRole.TOPIC,
            "I-TOPIC": GrammarRole.TOPIC,
            "B-NEG":   GrammarRole.NEGATION,
            "B-Q":     GrammarRole.QUESTION_YN,
            "B-INT":   GrammarRole.INTENSIFIER,
            "O":       GrammarRole.UNKNOWN,
        }
        return mapping.get(bio_tag, GrammarRole.UNKNOWN)

    def _apply_bio_tags(
        self,
        tokens: list[SignToken],
        bio_tags: list[str],
    ) -> list[SignToken]:
        """Attribue les GrammarRole issus du tagger BIO aux tokens.

        Si le tagger a utilisé les règles (pas le CRF), les rôles sont déjà
        renseignés par ``tag_grammar_roles`` ; on les préserve.

        Args:
            tokens: Tokens (potentiellement déjà taggés par les règles).
            bio_tags: Tags BIO du tagger.

        Returns:
            Copie des tokens avec grammar_role mis à jour.
        """
        result: list[SignToken] = []
        for tok, bio in zip(tokens, bio_tags):
            new_role = self._bio_to_role(bio)
            # Si le rôle était déjà précis (pas UNKNOWN), conserver le rôle
            # issu des règles (plus riche que le BIO simple)
            if tok.grammar_role != GrammarRole.UNKNOWN and new_role == GrammarRole.UNKNOWN:
                new_role = tok.grammar_role
            result.append(
                SignToken(
                    label=tok.label,
                    confidence=tok.confidence,
                    start_frame=tok.start_frame,
                    end_frame=tok.end_frame,
                    nmm_features=tok.nmm_features.copy(),
                    signing_space=tok.signing_space.copy(),
                    grammar_role=new_role,
                )
            )
        return result
