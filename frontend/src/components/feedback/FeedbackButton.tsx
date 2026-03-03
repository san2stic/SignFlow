import { useFeedbackStore } from "../../stores/feedbackStore";

interface FeedbackButtonProps {
  /** Signe prédit à corriger. */
  predictedSign: string;
  /** Confiance de la prédiction (valeur entre 0 et 1). */
  confidence?: number;
  /** Landmarks associés à la prédiction courante (optionnel). */
  landmarks?: number[][] | null;
  /**
   * Si `true`, le bouton est toujours rendu (mode flottant).
   * - Prédiction valide → bouton actif normal.
   * - Prédiction vide/NONE/RECORDING → bouton désactivé avec tooltip explicatif.
   * Défaut : `false` (comportement original : null si pas de prédiction).
   */
  alwaysVisible?: boolean;
}

/** Retourne `true` si la valeur est une prédiction absente ou invalide. */
function isEmptyPrediction(value: string): boolean {
  return !value || value === "NONE" || value === "RECORDING";
}

/**
 * Bouton discret qui ouvre le popup de correction.
 * - Mode normal (`alwaysVisible=false`) : null si pas de prédiction valide.
 * - Mode flottant (`alwaysVisible=true`) : toujours rendu, désactivé sans prédiction.
 */
export function FeedbackButton({
  predictedSign,
  confidence,
  landmarks,
  alwaysVisible = false,
}: FeedbackButtonProps): JSX.Element | null {
  const openFeedback = useFeedbackStore((state) => state.openFeedback);

  const empty = isEmptyPrediction(predictedSign);

  // Comportement original : invisible sans prédiction
  if (!alwaysVisible && empty) {
    return null;
  }

  const disabled = empty;

  return (
    <button
      type="button"
      title={disabled ? "Aucune prédiction récente à corriger" : "Corriger ce signe"}
      aria-label={disabled ? "Aucune prédiction récente à corriger" : "Corriger ce signe"}
      disabled={disabled}
      onClick={disabled ? undefined : () => openFeedback(predictedSign, confidence, landmarks ?? null)}
      className={
        "group inline-flex items-center gap-1.5 rounded-full " +
        "border px-3 py-1.5 text-xs font-medium backdrop-blur-sm " +
        "transition-all focus:outline-none focus:ring-2 focus:ring-primary/50 " +
        (disabled
          ? "cursor-not-allowed border-primary/15 bg-primary/5 text-primary/40 opacity-50"
          : "border-primary/30 bg-primary/10 text-primary " +
            "hover:border-primary/60 hover:bg-primary/20 active:scale-95")
      }
    >
      {/* Icône crayon */}
      <svg
        className={
          "h-3.5 w-3.5 " +
          (disabled ? "" : "transition-transform group-hover:scale-110")
        }
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
      Corriger
    </button>
  );
}
