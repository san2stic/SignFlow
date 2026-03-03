/**
 * ConversationPanel — Historique conversationnel style chat pour SignFlow.
 *
 * Affiche :
 * - Les tours de parole traduits (style sous-titres / chat)
 * - La phrase en cours de construction
 * - Les tags NMM avec icônes (❓ question, 🚫 négation…)
 * - La confiance par tour
 * - Un bouton de réinitialisation
 */

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import type { ConversationTurn } from "../../stores/translateStore";
import { SignConfidenceBar } from "./SignConfidenceBar";

// ---------------------------------------------------------------------------
// Helpers pour les tags NMM
// ---------------------------------------------------------------------------

const NMM_ICONS: Record<string, string> = {
  "B-Q": "❓",
  "B-NEG": "🚫",
  "I-NEG": "🚫",
  "B-COND": "🔀",
  "B-SUBJ": "👤",
  "B-VERB": "⚡",
  "B-OBJ": "📦",
  "B-ADJ": "🎨",
  "B-ADV": "🕐",
  "B-TOPIC": "💡",
};

function nmmBadges(tags: string[]): { icon: string; label: string }[] {
  const seen = new Set<string>();
  const result: { icon: string; label: string }[] = [];
  for (const tag of tags) {
    const icon = NMM_ICONS[tag];
    if (icon && !seen.has(tag)) {
      seen.add(tag);
      result.push({ icon, label: tag.replace(/^[BI]-/, "") });
    }
  }
  return result;
}

function formatTime(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ---------------------------------------------------------------------------
// Composant TurnCard
// ---------------------------------------------------------------------------

interface TurnCardProps {
  turn: ConversationTurn;
  isLatest: boolean;
  showSigns: boolean;
  onToggleSigns: () => void;
}

function TurnCard({ turn, isLatest, showSigns, onToggleSigns }: TurnCardProps): JSX.Element {
  const badges = nmmBadges(turn.grammar_tags);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`group relative rounded-card p-4 ${
        isLatest
          ? "border border-primary/30 bg-primary/10 shadow-glow"
          : "border border-white/5 bg-surface-secondary/60"
      }`}
    >
      {/* En-tête du tour */}
      <div className="mb-1 flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Numéro de tour */}
          <span className="rounded-full bg-surface px-2 py-0.5 font-mono text-[10px] text-text-tertiary">
            #{turn.id}
          </span>
          {/* Timestamp */}
          <span className="font-mono text-[10px] text-text-tertiary">{formatTime(turn.start_time)}</span>
          {/* Badges NMM */}
          {badges.map((b) => (
            <span
              key={b.label}
              className="rounded-full bg-secondary/20 px-1.5 py-0.5 text-[10px] font-medium text-secondary"
              title={b.label}
            >
              {b.icon} {b.label}
            </span>
          ))}
        </div>
        {/* Confiance */}
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold ${
            turn.confidence >= 0.7
              ? "bg-emerald-500/20 text-emerald-400"
              : turn.confidence >= 0.5
                ? "bg-amber-500/20 text-amber-400"
                : "bg-red-500/20 text-red-400"
          }`}
        >
          {Math.round(turn.confidence * 100)}%
        </span>
      </div>

      {/* Texte de la phrase traduite */}
      <p className={`font-body text-base leading-relaxed ${isLatest ? "text-text-primary font-semibold" : "text-text-secondary"}`}>
        {turn.text || <em className="text-text-muted">—</em>}
      </p>

      {/* Barre de confiance */}
      <SignConfidenceBar confidence={turn.confidence} className="mt-2" height={3} />

      {/* Signes bruts (pliables) */}
      {turn.raw_signs.length > 0 && (
        <div className="mt-2">
          <button
            onClick={onToggleSigns}
            className="text-[10px] font-medium text-text-tertiary transition-colors hover:text-primary"
          >
            {showSigns ? "▲ Masquer signes" : "▼ Voir signes bruts"}
          </button>
          <AnimatePresence>
            {showSigns && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {turn.raw_signs.map((sign, idx) => (
                    <span
                      key={`${sign}-${idx}`}
                      className="rounded-md bg-surface px-2 py-0.5 font-mono text-[11px] text-primary"
                    >
                      {sign}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// ConversationPanel principal
// ---------------------------------------------------------------------------

interface Props {
  /** Historique des tours traduits. */
  history: ConversationTurn[];
  /** Phrase actuelle en cours de construction (signes en attente). */
  currentBuilding: string;
  /** Mode de traduction actif (rules / crf / seq2seq). */
  translationMode?: string;
  /** Callback pour réinitialiser la conversation. */
  onReset: () => void;
  /** Classes Tailwind supplémentaires. */
  className?: string;
}

export function ConversationPanel({
  history,
  currentBuilding,
  translationMode = "rules",
  onReset,
  className = "",
}: Props): JSX.Element {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [expandedTurns, setExpandedTurns] = useState<Set<number>>(new Set());

  // Auto-scroll en bas à chaque nouveau tour
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history.length]);

  const toggleSigns = (turnId: number): void => {
    setExpandedTurns((prev) => {
      const next = new Set(prev);
      if (next.has(turnId)) {
        next.delete(turnId);
      } else {
        next.add(turnId);
      }
      return next;
    });
  };

  const modeLabel: Record<string, string> = {
    rules: "Règles",
    crf: "CRF",
    seq2seq: "Seq2Seq",
  };

  return (
    <div className={`flex flex-col overflow-hidden rounded-card border border-white/10 bg-surface/80 backdrop-blur-sm ${className}`}>
      {/* En-tête du panneau */}
      <div className="flex shrink-0 items-center justify-between border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2">
          <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <h2 className="font-display text-sm font-semibold text-text-primary">Conversation</h2>
          {history.length > 0 && (
            <span className="rounded-full bg-primary/20 px-1.5 py-0.5 font-mono text-[10px] text-primary">
              {history.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Badge mode traduction */}
          <span className="rounded-full border border-secondary/30 bg-secondary/10 px-2 py-0.5 text-[10px] font-medium text-secondary">
            {modeLabel[translationMode] ?? translationMode}
          </span>
          {/* Bouton reset */}
          <button
            onClick={onReset}
            className="flex items-center gap-1 rounded-btn px-2 py-1 text-[11px] font-medium text-text-tertiary transition-colors hover:bg-red-500/20 hover:text-red-400"
            title="Réinitialiser la conversation"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Réinitialiser
          </button>
        </div>
      </div>

      {/* Liste des turns */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0 max-h-96">
        {history.length === 0 && !currentBuilding && (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <svg className="mb-3 h-10 w-10 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2v4l.586-.586z" />
            </svg>
            <p className="text-sm text-text-muted">En attente de signes…</p>
            <p className="mt-1 text-xs text-text-muted opacity-60">Commencez à signer pour voir la traduction</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {history.map((turn, idx) => (
            <TurnCard
              key={turn.id}
              turn={turn}
              isLatest={idx === history.length - 1}
              showSigns={expandedTurns.has(turn.id)}
              onToggleSigns={() => toggleSigns(turn.id)}
            />
          ))}
        </AnimatePresence>

        {/* Phrase en cours de construction */}
        <AnimatePresence>
          {currentBuilding && (
            <motion.div
              key="building"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-card border border-dashed border-primary/40 bg-primary/5 px-4 py-3"
            >
              <div className="mb-1 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                <span className="text-[10px] font-medium uppercase tracking-wide text-primary">En cours…</span>
              </div>
              <p className="font-body text-sm text-text-secondary">
                {currentBuilding.split(" ").map((token, i) => (
                  <motion.span
                    key={`${token}-${i}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mr-1 inline-block font-mono text-primary/80"
                  >
                    {token}
                  </motion.span>
                ))}
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
