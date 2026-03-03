import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useFeedbackStore } from "../../stores/feedbackStore";

interface FeedbackPopupProps {
  /**
   * Fonction `send` issue de `useWebSocket`, passée depuis la page parente
   * pour que la soumission passe par le WebSocket existant.
   */
  sendViaWs: (msg: object) => void;
}

/**
 * Popup de correction de signe.
 * Lit son état depuis `feedbackStore` et l'affiche de façon modale.
 */
export function FeedbackPopup({ sendViaWs }: FeedbackPopupProps): JSX.Element {
  const isOpen = useFeedbackStore((s) => s.isOpen);
  const predictedSign = useFeedbackStore((s) => s.predictedSign);
  const submitting = useFeedbackStore((s) => s.submitting);
  const lastResult = useFeedbackStore((s) => s.lastResult);
  const closeFeedback = useFeedbackStore((s) => s.closeFeedback);
  const submitFeedback = useFeedbackStore((s) => s.submitFeedback);

  const [correctedSign, setCorrectedSign] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Réinitialise le champ à chaque ouverture
  useEffect(() => {
    if (isOpen) {
      setCorrectedSign("");
      setLocalError(null);
      // Focus différé pour laisser l'animation s'afficher
      const tid = window.setTimeout(() => inputRef.current?.focus(), 80);
      return () => window.clearTimeout(tid);
    }
  }, [isOpen]);

  // Fermeture automatique 2 secondes après l'acknowledgement
  useEffect(() => {
    if (lastResult) {
      const tid = window.setTimeout(() => {
        closeFeedback();
      }, 2000);
      return () => window.clearTimeout(tid);
    }
  }, [lastResult, closeFeedback]);

  const handleSubmit = (event: React.FormEvent): void => {
    event.preventDefault();
    if (!correctedSign.trim()) {
      setLocalError("Veuillez saisir le signe correct.");
      return;
    }
    setLocalError(null);
    submitFeedback(correctedSign.trim().toUpperCase(), sendViaWs);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            key="feedback-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="fixed inset-0 z-[70] bg-black/60 backdrop-blur-sm"
            onClick={submitting ? undefined : closeFeedback}
            aria-hidden="true"
          />

          {/* Panel */}
          <motion.div
            key="feedback-panel"
            role="dialog"
            aria-modal="true"
            aria-label="Corriger la prédiction"
            initial={{ opacity: 0, scale: 0.92, y: 24 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 24 }}
            transition={{ type: "spring", stiffness: 340, damping: 30 }}
            className="fixed inset-x-4 bottom-8 z-[71] mx-auto max-w-md rounded-2xl border border-primary/30 bg-background-elevated p-6 shadow-2xl sm:inset-auto sm:left-1/2 sm:top-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2"
          >
            {/* Header */}
            <div className="mb-5 flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-secondary/20">
                  <svg
                    className="h-5 w-5 text-primary"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                    />
                  </svg>
                </div>
                <h2 className="font-display text-lg font-semibold text-text-primary">
                  Corriger la prédiction
                </h2>
              </div>
              <button
                type="button"
                onClick={closeFeedback}
                disabled={submitting}
                aria-label="Fermer"
                className="rounded-full p-1 text-text-tertiary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
              >
                <svg
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                  aria-hidden="true"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Predicted sign display */}
            <div className="mb-5 rounded-xl bg-surface-secondary/60 px-4 py-3">
              <p className="mb-0.5 text-xs font-medium uppercase tracking-wider text-text-tertiary">
                Prédiction actuelle
              </p>
              <p className="font-display text-2xl font-bold tracking-tight text-primary">
                {predictedSign}
              </p>
            </div>

            {/* Success state (after acknowledgement) */}
            {lastResult && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`mb-4 rounded-xl px-4 py-3 text-sm font-medium ${
                  lastResult.trainingTriggered
                    ? "bg-accent/15 text-accent"
                    : "bg-primary/15 text-primary"
                }`}
              >
                {lastResult.trainingTriggered
                  ? "🎉 Assez de corrections — réentraînement déclenché !"
                  : "✓ Correction enregistrée. Merci !"}
              </motion.div>
            )}

            {/* Form */}
            {!lastResult && (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="flex flex-col gap-1.5">
                  <label
                    htmlFor="feedback-corrected-sign"
                    className="text-sm font-medium text-text-secondary"
                  >
                    Quel est le vrai signe ?
                  </label>
                  <input
                    ref={inputRef}
                    id="feedback-corrected-sign"
                    type="text"
                    value={correctedSign}
                    onChange={(e) => {
                      setCorrectedSign(e.target.value);
                      setLocalError(null);
                    }}
                    placeholder="Ex : AU-REVOIR"
                    disabled={submitting}
                    autoComplete="off"
                    spellCheck={false}
                    className={`rounded-xl border px-4 py-3 text-base uppercase tracking-wide backdrop-blur-sm transition-all focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:opacity-60 ${
                      localError
                        ? "border-red-400/70 bg-red-500/10 focus:border-red-400 focus:ring-red-400/50"
                        : "border-primary/30 bg-surface-secondary/80 focus:border-primary focus:ring-primary/50"
                    }`}
                  />
                  {localError && (
                    <p className="text-xs text-red-400">{localError}</p>
                  )}
                </div>

                <div className="flex gap-3 pt-1">
                  <button
                    type="button"
                    onClick={closeFeedback}
                    disabled={submitting}
                    className="flex-1 rounded-xl border border-surface-secondary bg-surface-secondary/60 px-4 py-3 text-sm font-medium text-text-secondary transition-all hover:bg-surface-secondary disabled:opacity-50"
                  >
                    Annuler
                  </button>
                  <button
                    type="submit"
                    disabled={submitting || !correctedSign.trim()}
                    className="flex-1 rounded-xl bg-gradient-to-br from-primary to-secondary px-4 py-3 text-sm font-medium text-white shadow-md transition-all hover:opacity-90 disabled:opacity-50"
                  >
                    {submitting ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg
                          className="h-4 w-4 animate-spin"
                          viewBox="0 0 24 24"
                          fill="none"
                          aria-hidden="true"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"
                          />
                        </svg>
                        Envoi...
                      </span>
                    ) : (
                      "Soumettre la correction"
                    )}
                  </button>
                </div>
              </form>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
