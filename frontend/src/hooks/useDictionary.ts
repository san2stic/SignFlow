import { useCallback } from "react";

import { getGraph } from "../api/dictionary";
import { useDictionaryStore } from "../stores/dictionaryStore";

export function useDictionary() {
  const setGraph = useDictionaryStore((state) => state.setGraph);

  const refreshGraph = useCallback(async () => {
    const payload = await getGraph();
    setGraph(
      payload.nodes.map((node) => ({ id: node.id, label: node.label, category: node.category })),
      payload.edges.map((edge) => ({ source: edge.source, target: edge.target }))
    );
  }, [setGraph]);

  return { refreshGraph };
}
