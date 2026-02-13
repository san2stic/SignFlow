import { create } from "zustand";

const MIN_STICKY_CONFIDENCE = 0.55;
const STICKY_PREDICTION_HOLD_MS = 1500;

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
      const isConfidentPrediction =
        value.prediction !== "NONE" && value.prediction !== "RECORDING" && value.confidence >= MIN_STICKY_CONFIDENCE;

      let displayedPrediction = state.displayedPrediction;
      let displayedConfidence = state.displayedConfidence;
      let displayUntilMs = state.displayUntilMs;

      if (isConfidentPrediction) {
        displayedPrediction = value.prediction;
        displayedConfidence = value.confidence;
        displayUntilMs = now + STICKY_PREDICTION_HOLD_MS;
      } else if (value.prediction !== "NONE" && value.prediction !== "RECORDING" && now >= state.displayUntilMs) {
        displayedPrediction = value.prediction;
        displayedConfidence = value.confidence;
        displayUntilMs = now;
      } else if (now >= state.displayUntilMs) {
        displayedPrediction = "NONE";
        displayedConfidence = 0;
        displayUntilMs = 0;
      }

      const shouldPushHistory = value.prediction !== "NONE" && value.prediction !== "RECORDING" && value.confidence >= 0.7;
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
