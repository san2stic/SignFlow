import { create } from "zustand";

export interface TrainingMetricPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  val_accuracy: number;
}

export interface TrainingProgress {
  status: string;
  progress: number;
  metrics: {
    loss: number;
    accuracy: number;
    val_accuracy: number;
    current_epoch?: number;
  };
  metrics_history?: TrainingMetricPoint[];
  deployment_ready?: boolean;
  deploy_threshold?: number;
  final_val_accuracy?: number | null;
  recommended_next_action?: "deploy" | "collect_more_examples" | "wait" | "review_error";
}

export interface PendingClip {
  file: File;
  suggestedName?: string;
  assignedSign?: {
    signId: string;
    signName: string;
  };
  createdAt: number;
}

interface TrainingStore {
  activeSessionId: string | null;
  progress: TrainingProgress;
  pendingClip: PendingClip | null;
  setSession: (sessionId: string | null) => void;
  setProgress: (progress: TrainingProgress) => void;
  resetProgress: () => void;
  setPendingClip: (clip: PendingClip | null) => void;
  clearPendingClip: () => void;
}

const INITIAL_PROGRESS: TrainingProgress = {
  status: "idle",
  progress: 0,
  metrics: { loss: 0, accuracy: 0, val_accuracy: 0, current_epoch: 0 },
  metrics_history: [],
  deployment_ready: false,
  deploy_threshold: 0.85,
  final_val_accuracy: null,
  recommended_next_action: "wait"
};

export const useTrainingStore = create<TrainingStore>((set) => ({
  activeSessionId: null,
  progress: INITIAL_PROGRESS,
  pendingClip: null,
  setSession: (sessionId) => set({ activeSessionId: sessionId }),
  setProgress: (progress) =>
    set((state) => {
      const existingHistory = progress.metrics_history ?? state.progress.metrics_history ?? [];
      const epoch = progress.metrics.current_epoch;
      let nextHistory = existingHistory;

      if (typeof epoch === "number" && epoch > 0) {
        const nextPoint: TrainingMetricPoint = {
          epoch,
          loss: progress.metrics.loss,
          accuracy: progress.metrics.accuracy,
          val_accuracy: progress.metrics.val_accuracy
        };
        const lastPoint = existingHistory[existingHistory.length - 1];
        const isDuplicate =
          lastPoint &&
          lastPoint.epoch === nextPoint.epoch &&
          lastPoint.loss === nextPoint.loss &&
          lastPoint.accuracy === nextPoint.accuracy &&
          lastPoint.val_accuracy === nextPoint.val_accuracy;
        if (!isDuplicate) {
          nextHistory = [...existingHistory, nextPoint].slice(-120);
        }
      }

      return {
        progress: {
          ...progress,
          metrics_history: nextHistory
        }
      };
    }),
  resetProgress: () => set({ progress: { ...INITIAL_PROGRESS, metrics_history: [] } }),
  setPendingClip: (pendingClip) => set({ pendingClip }),
  clearPendingClip: () => set({ pendingClip: null })
}));
