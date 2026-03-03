import { useFeedbackStore } from "../../stores/feedbackStore";

interface FeedbackButtonProps {
  /** Signe prédit à corriger. */
  predictedSign: string;
  /** Confiance de la prédiction (valeur entre 0 et 1). */
  confidence?: number;
  /** Landmarks associés à la prédiction courante (optionnel). */
  landmarks?: number[][] | null;
}

/**
 * Bouton discret qui ouvre le popup de correction.
 * N'est rendu que si `predictedSign` est non vide et différent de "NONE".
 */
export function FeedbackButton({
  predictedSign,
  confidence,
  landmarks,
}: FeedbackButtonProps): JSX.Element | null {
  const openFeedback = useFeedbackStore((state) => state.openFeedback);

  if (!predictedSign || predictedSign === "NONE" || predictedSign === "RECORDING") {
    return null;
  }

  return (
    <button
      type="button"
      title="Corriger ce signe"
      aria-label="Corriger ce signe"
      onClick={() => openFeedback(predictedSign, confidence, landmarks ?? null)}
      className={
        "group inline-flex items-center gap-1.5 rounded-full " +
        "border border-primary/30 bg-primary/10 px-3 py-1.5 " +
        "text-xs font-medium text-primary backdrop-blur-sm " +
        "transition-all hover:border-primary/60 hover:bg-primary/20 " +
        "focus:outline-none focus:ring-2 focus:ring-primary/50 " +
        "active:scale-95"
      }
    >
      {/* Icône crayon */}
      <svg
        className="h-3.5 w-3.5 transition-transform group-hover:scale-110"
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
