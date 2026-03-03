import { create } from "zustand";
import type { DeploymentHistory, UpdaterStatus } from "../api/updater";

interface UpdaterStore {
  status: UpdaterStatus | null;
  history: DeploymentHistory[];
  buildLogLines: string[];
  wsConnected: boolean;
  isLoading: boolean;
  error: string | null;

  setStatus: (status: UpdaterStatus) => void;
  setHistory: (history: DeploymentHistory[]) => void;
  appendLogLine: (line: string) => void;
  clearLogs: () => void;
  setWsConnected: (v: boolean) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  addOrUpdateHistoryEntry: (entry: DeploymentHistory) => void;
}

export const useUpdaterStore = create<UpdaterStore>((set) => ({
  status: null,
  history: [],
  buildLogLines: [],
  wsConnected: false,
  isLoading: false,
  error: null,

  setStatus: (status) => set({ status }),

  setHistory: (history) => set({ history }),

  appendLogLine: (line) =>
    set((state) => ({
      buildLogLines: [...state.buildLogLines, line].slice(-2000) // limiter à 2000 lignes
    })),

  clearLogs: () => set({ buildLogLines: [] }),

  setWsConnected: (wsConnected) => set({ wsConnected }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error }),

  addOrUpdateHistoryEntry: (entry) =>
    set((state) => {
      const exists = state.history.some((h) => h.id === entry.id);
      if (exists) {
        return {
          history: state.history.map((h) => (h.id === entry.id ? entry : h))
        };
      }
      // Ajouter en tête et trier par created_at DESC
      const updated = [entry, ...state.history].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      return { history: updated };
    })
}));
