import { create } from "zustand";

export interface DictionaryNode {
  id: string;
  label: string;
  category?: string;
}

interface DictionaryStore {
  graphNodes: DictionaryNode[];
  graphEdges: Array<{ source: string; target: string }>;
  setGraph: (nodes: DictionaryNode[], edges: Array<{ source: string; target: string }>) => void;
}

export const useDictionaryStore = create<DictionaryStore>((set) => ({
  graphNodes: [],
  graphEdges: [],
  setGraph: (nodes, edges) => set({ graphNodes: nodes, graphEdges: edges })
}));
