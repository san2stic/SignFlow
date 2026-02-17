import { create } from "zustand";

const MIN_VISIBLE_PREDICTION_CONFIDENCE = 0.2;
const MIN_STICKY_CONFIDENCE = 0.55;
const STICKY_PREDICTION_HOLD_MS = 3000;
const TRANSIENT_PREDICTION_HOLD_MS = 1200;

export interface PredictionState {
  prediction: string;
  confidence: number;
  sentenceBuffer: string;
  alternatives: Array<{ sign: string; confidence: number }>;
}

interface TranslateStore {
  live: PredictionState;
  displayedPrediction: string;
  displayedConfidence: number;
  displayUntilMs: number;
  history: string[];
  setLive: (value: PredictionState) => void;
  reset: () => void;
}

export const useTranslateStore = create<TranslateStore>((set) => ({
  live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
  displayedPrediction: "NONE",
  displayedConfidence: 0,
  displayUntilMs: 0,
  history: [],
  setLive: (value) =>
    set((state) => {
      const now = Date.now();
      const isNonEmptyPrediction = value.prediction !== "NONE" && value.prediction !== "RECORDING";
      const isVisiblePrediction = isNonEmptyPrediction && value.confidence >= MIN_VISIBLE_PREDICTION_CONFIDENCE;
      const isConfidentPrediction = isVisiblePrediction && value.confidence >= MIN_STICKY_CONFIDENCE;
      const isSameAsDisplayed = state.displayedPrediction === value.prediction;

      let displayedPrediction = state.displayedPrediction;
      let displayedConfidence = state.displayedConfidence;
      let displayUntilMs = state.displayUntilMs;

      if (isConfidentPrediction) {
        displayedPrediction = value.prediction;
        displayedConfidence = value.confidence;
        displayUntilMs = now + STICKY_PREDICTION_HOLD_MS;
      } else if (isVisiblePrediction) {
        const canPromoteTransientPrediction = now >= state.displayUntilMs || isSameAsDisplayed || state.displayedPrediction === "NONE";

        if (canPromoteTransientPrediction) {
          displayedPrediction = value.prediction;
          displayedConfidence = isSameAsDisplayed ? Math.max(state.displayedConfidence, value.confidence) : value.confidence;
          displayUntilMs = Math.max(state.displayUntilMs, now + TRANSIENT_PREDICTION_HOLD_MS);
        }
      } else if (now >= state.displayUntilMs) {
        displayedPrediction = "NONE";
        displayedConfidence = 0;
        displayUntilMs = 0;
      }

      const shouldPushHistory = isNonEmptyPrediction && value.confidence >= 0.7;
      const history =
        shouldPushHistory && state.history[0] !== value.prediction
          ? [value.prediction, ...state.history].slice(0, 30)
          : state.history;

      return {
        live: value,
        displayedPrediction,
        displayedConfidence,
        displayUntilMs,
        history
      };
    }),
  reset: () =>
    set({
      live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
      displayedPrediction: "NONE",
      displayedConfidence: 0,
      displayUntilMs: 0,
      history: []
    })
}));
