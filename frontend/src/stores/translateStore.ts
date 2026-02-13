import { create } from "zustand";

export interface PredictionState {
  prediction: string;
  confidence: number;
  sentenceBuffer: string;
  alternatives: Array<{ sign: string; confidence: number }>;
}

interface TranslateStore {
  live: PredictionState;
  history: string[];
  setLive: (value: PredictionState) => void;
  reset: () => void;
}

export const useTranslateStore = create<TranslateStore>((set) => ({
  live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
  history: [],
  setLive: (value) =>
    set((state) => ({
      live: value,
      history:
        value.prediction !== "NONE" && value.confidence >= 0.7
          ? [value.prediction, ...state.history].slice(0, 30)
          : state.history
    })),
  reset: () => set({ live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] }, history: [] })
}));
