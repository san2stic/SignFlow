import { create } from "zustand";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FeedbackState {
  /** Popup ouverte ? */
  isOpen: boolean;
  /** Prédiction courante à corriger */
  predictedSign: string | null;
  /** Confiance de la prédiction au moment de l'ouverture */
  confidence: number | null;
  /** Landmarks courants (tableau de frames sérialisées) */
  currentLandmarks: number[][] | null;
  /** Statut de soumission */
  submitting: boolean;
  /** Dernier retour du serveur après acknowledgement */
  lastResult: { correctionId: number; trainingTriggered: boolean } | null;
  /** Toast de notification globale */
  toast: { message: string; type: "success" | "error" | "info" } | null;
}

interface FeedbackStore extends FeedbackState {
  /** Ouvre le popup de correction pour une prédiction donnée. */
  openFeedback: (
    predictedSign: string,
    confidence?: number,
    landmarks?: number[][] | null
  ) => void;
  /** Ferme le popup sans soumettre. */
  closeFeedback: () => void;
  /**
   * Envoie le message `submit_feedback` via le WebSocket existant.
   * `sendViaWs` est la fonction `send` retournée par `useWebSocket`.
   */
  submitFeedback: (
    correctedSign: string,
    sendViaWs: (msg: object) => void
  ) => void;
  /**
   * Traite la réponse `feedback_acknowledged` du serveur.
   * À appeler depuis le handler `onMessage` de `useWebSocket`.
   */
  handleAcknowledgement: (
    correctionId: number,
    trainingTriggered: boolean
  ) => void;
  /** Efface le toast courant. */
  clearToast: () => void;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useFeedbackStore = create<FeedbackStore>((set, get) => ({
  isOpen: false,
  predictedSign: null,
  confidence: null,
  currentLandmarks: null,
  submitting: false,
  lastResult: null,
  toast: null,

  openFeedback: (predictedSign, confidence, landmarks) =>
    set({
      isOpen: true,
      predictedSign,
      confidence: confidence ?? null,
      currentLandmarks: landmarks ?? null,
      submitting: false,
      lastResult: null,
    }),

  closeFeedback: () =>
    set({
      isOpen: false,
      predictedSign: null,
      confidence: null,
      currentLandmarks: null,
      submitting: false,
    }),

  submitFeedback: (correctedSign, sendViaWs) => {
    const { predictedSign, confidence, currentLandmarks } = get();
    if (!predictedSign) return;

    set({ submitting: true });

    sendViaWs({
      type: "submit_feedback",
      predicted_sign: predictedSign,
      corrected_sign: correctedSign,
      confidence: confidence ?? 0,
      landmarks: currentLandmarks ?? null,
    });

    // L'acknowledgement arrivera via handleAcknowledgement ; on ne ferme pas
    // immédiatement — le popup attend la réponse du serveur ou un timeout.
  },

  handleAcknowledgement: (correctionId, trainingTriggered) => {
    set({
      submitting: false,
      isOpen: false,
      lastResult: { correctionId, trainingTriggered },
      toast: trainingTriggered
        ? {
            message:
              "🎉 Assez de corrections — réentraînement déclenché !",
            type: "success",
          }
        : {
            message: "✓ Correction enregistrée.",
            type: "info",
          },
    });
  },

  clearToast: () => set({ toast: null }),
}));
