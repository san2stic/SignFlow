import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { exportDictionary, getGraph, importDictionary, type GraphPayload } from "../api/dictionary";
import {
  createSign,
  getSign,
  getSignBacklinks,
  listSigns,
  type Backlink,
  type Sign
} from "../api/signs";
import { SearchBar } from "../components/common/SearchBar";
import { BacklinksPanel } from "../components/dictionary/BacklinksPanel";
import { GraphView } from "../components/dictionary/GraphView";
import { SignCard } from "../components/dictionary/SignCard";
import { SignDetail } from "../components/dictionary/SignDetail";

const SIGNS_CACHE_KEY = "signflow.dictionary.signs";
const GRAPH_CACHE_KEY = "signflow.dictionary.graph";

type ExportFormat = "json" | "markdown" | "obsidian-vault";

interface CreateSignDraft {
  name: string;
  category: string;
  tags: string;
  description: string;
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function getStorageItem(key: string): string | null {
  try {
    if (typeof window === "undefined" || !window.localStorage || typeof window.localStorage.getItem !== "function") {
      return null;
    }
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function setStorageItem(key: string, value: string): void {
  try {
    if (typeof window === "undefined" || !window.localStorage || typeof window.localStorage.setItem !== "function") {
      return;
    }
    window.localStorage.setItem(key, value);
  } catch {
    // Intentionally ignore storage errors (quota/security/test env shims).
  }
}

export function DictionaryPage(): JSX.Element {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [view, setView] = useState<"graph" | "cards">("graph");
  const [signs, setSigns] = useState<Sign[]>([]);
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] });
  const [selectedSignId, setSelectedSignId] = useState<string | null>(null);
  const [selectedSign, setSelectedSign] = useState<Sign | null>(null);
  const [backlinks, setBacklinks] = useState<Backlink[]>([]);
  const [isOfflineFallback, setIsOfflineFallback] = useState(false);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isExporting, setIsExporting] = useState<ExportFormat | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<CreateSignDraft>({
    name: "",
    category: "",
    tags: "",
    description: ""
  });

  const selectedSignFromList = useMemo(() => signs.find((item) => item.id === selectedSignId) ?? null, [selectedSignId, signs]);
  const effectiveSelectedSign = selectedSign ?? selectedSignFromList;

  const refreshGraph = useCallback(async () => {
    try {
      const payload = await getGraph();
      setGraph(payload);
      setStorageItem(GRAPH_CACHE_KEY, JSON.stringify(payload));
      setIsOfflineFallback(false);
    } catch {
      const cached = getStorageItem(GRAPH_CACHE_KEY);
      if (cached) {
        try {
          const parsed = JSON.parse(cached) as GraphPayload;
          setGraph(parsed);
          setIsOfflineFallback(true);
        } catch {
          setGraph({ nodes: [], edges: [] });
        }
      }
    }
  }, []);

  const refreshSigns = useCallback(async () => {
    try {
      const response = await listSigns(query);
      setSigns(response.items);
      setStorageItem(SIGNS_CACHE_KEY, JSON.stringify(response.items));
      setIsOfflineFallback(false);
    } catch {
      const cached = getStorageItem(SIGNS_CACHE_KEY);
      if (cached) {
        try {
          const parsed = JSON.parse(cached) as Sign[];
          setSigns(parsed);
          setIsOfflineFallback(true);
        } catch {
          setSigns([]);
        }
      } else {
        setSigns([]);
      }
    }
  }, [query]);

  const refreshSelectedSign = useCallback(async () => {
    if (!selectedSignId) {
      setSelectedSign(null);
      setBacklinks([]);
      return;
    }
    try {
      const [details, backlinksPayload] = await Promise.all([getSign(selectedSignId), getSignBacklinks(selectedSignId)]);
      setSelectedSign(details);
      setBacklinks(backlinksPayload.backlinks);
    } catch {
      setSelectedSign(selectedSignFromList);
      setBacklinks([]);
    }
  }, [selectedSignFromList, selectedSignId]);

  useEffect(() => {
    void refreshGraph();
  }, [refreshGraph]);

  useEffect(() => {
    void refreshSigns();
  }, [refreshSigns]);

  useEffect(() => {
    void refreshSelectedSign();
  }, [refreshSelectedSign]);

  const onCreateSign = async (): Promise<void> => {
    if (!draft.name.trim()) {
      setError("Sign name is required.");
      return;
    }

    setError(null);
    try {
      const created = await createSign({
        name: draft.name.trim(),
        category: draft.category.trim() || undefined,
        tags: draft.tags
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
        description: draft.description || undefined,
        variants: [],
        related_signs: [],
        notes: ""
      });
      setStatusMessage(`Created sign "${created.name}".`);
      setDraft({ name: "", category: "", tags: "", description: "" });
      setIsCreateOpen(false);
      await Promise.all([refreshSigns(), refreshGraph()]);
      setSelectedSignId(created.id);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create sign.");
    }
  };

  const onImport = async (file: File): Promise<void> => {
    setIsImporting(true);
    setError(null);
    setStatusMessage(null);
    try {
      const result = await importDictionary(file);
      setStatusMessage(
        `Import completed: ${result.imported_signs} signs, ${result.imported_notes} notes, ${result.skipped} skipped.`
      );
      if (result.errors.length > 0) {
        setError(result.errors.join(" | "));
      }
      await Promise.all([refreshSigns(), refreshGraph()]);
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Import failed.");
    } finally {
      setIsImporting(false);
    }
  };

  const onExport = async (format: ExportFormat): Promise<void> => {
    setIsExporting(format);
    setError(null);
    try {
      const blob = await exportDictionary(format);
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      downloadBlob(`signflow-dictionary-${format}-${stamp}.zip`, blob);
      setStatusMessage(`Exported dictionary as ${format}.`);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Export failed.");
    } finally {
      setIsExporting(null);
    }
  };

  const onTrainSign = useCallback(
    (sign: Pick<Sign, "id" | "name" | "training_sample_count" | "video_count">): void => {
      navigate("/train", {
        state: {
          assignedSign: {
            id: sign.id,
            name: sign.name,
            trainingSampleCount: sign.training_sample_count,
            videoCount: sign.video_count
          }
        }
      });
    },
    [navigate]
  );

  return (
    <section className="space-y-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-heading text-2xl">Dictionary</h1>
        <div className="flex flex-wrap gap-2">
          <button className="touch-btn bg-primary text-white" onClick={() => setIsCreateOpen((current) => !current)}>
            {isCreateOpen ? "Close" : "New Sign"}
          </button>
          <label className="touch-btn cursor-pointer bg-slate-700 text-white">
            {isImporting ? "Importing..." : "Import"}
            <input
              type="file"
              className="hidden"
              accept=".zip,application/zip"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  void onImport(file);
                }
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>
      </header>

      {isCreateOpen && (
        <div className="card grid gap-3 p-4">
          <label className="flex flex-col gap-1 text-sm">
            Name
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={draft.name}
              onChange={(event) => setDraft((current) => ({ ...current, name: event.target.value }))}
              placeholder="Bonjour"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Category
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={draft.category}
              onChange={(event) => setDraft((current) => ({ ...current, category: event.target.value }))}
              placeholder="salutations"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Tags (comma separated)
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={draft.tags}
              onChange={(event) => setDraft((current) => ({ ...current, tags: event.target.value }))}
              placeholder="lsfb, v1"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Description
            <textarea
              className="h-24 rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={draft.description}
              onChange={(event) => setDraft((current) => ({ ...current, description: event.target.value }))}
            />
          </label>
          <button className="touch-btn bg-secondary text-slate-950" onClick={() => void onCreateSign()}>
            Create Sign
          </button>
        </div>
      )}

      {isOfflineFallback && (
        <div className="card border border-accent/50 bg-accent/10 p-3 text-sm text-accent">
          Offline fallback enabled: showing last cached dictionary data.
        </div>
      )}
      {statusMessage && <div className="card border border-secondary/40 bg-secondary/10 p-3 text-sm text-secondary">{statusMessage}</div>}
      {error && <div className="card border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">{error}</div>}

      <SearchBar value={query} onChange={setQuery} placeholder="Search signs..." />

      <div className="flex flex-wrap items-center gap-2">
        <button
          className={`touch-btn ${view === "graph" ? "bg-primary text-white" : "bg-slate-700 text-slate-200"}`}
          onClick={() => setView("graph")}
        >
          Graph
        </button>
        <button
          className={`touch-btn ${view === "cards" ? "bg-primary text-white" : "bg-slate-700 text-slate-200"}`}
          onClick={() => setView("cards")}
        >
          Cards
        </button>
        <div className="ml-auto flex flex-wrap gap-2">
          <button
            className="touch-btn bg-slate-700 text-white disabled:opacity-60"
            disabled={isExporting !== null}
            onClick={() => void onExport("json")}
          >
            {isExporting === "json" ? "Exporting..." : "Export JSON"}
          </button>
          <button
            className="touch-btn bg-slate-700 text-white disabled:opacity-60"
            disabled={isExporting !== null}
            onClick={() => void onExport("markdown")}
          >
            {isExporting === "markdown" ? "Exporting..." : "Export Markdown"}
          </button>
          <button
            className="touch-btn bg-slate-700 text-white disabled:opacity-60"
            disabled={isExporting !== null}
            onClick={() => void onExport("obsidian-vault")}
          >
            {isExporting === "obsidian-vault" ? "Exporting..." : "Export Obsidian"}
          </button>
        </div>
      </div>

      {view === "graph" ? (
        <GraphView
          graph={graph}
          onSelectNode={(nodeId) => {
            setSelectedSignId(nodeId);
          }}
        />
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {signs.map((sign) => (
            <SignCard
              key={sign.id}
              sign={sign}
              onSelect={(selected) => {
                setSelectedSignId(selected.id);
              }}
              onTrain={onTrainSign}
            />
          ))}
        </div>
      )}

      {effectiveSelectedSign && (
        <div className="grid gap-3 lg:grid-cols-[2fr_1fr]">
          <SignDetail
            sign={effectiveSelectedSign}
            allSigns={signs}
            onSelectSign={(signId) => setSelectedSignId(signId)}
            onTrain={onTrainSign}
            onUpdated={(updated) => {
              setSigns((current) => current.map((candidate) => (candidate.id === updated.id ? updated : candidate)));
              setSelectedSign(updated);
              void refreshGraph();
            }}
            onDeleted={(signId) => {
              setSigns((current) => current.filter((candidate) => candidate.id !== signId));
              setSelectedSignId(null);
              setSelectedSign(null);
              setBacklinks([]);
              void refreshGraph();
            }}
          />
          <BacklinksPanel backlinks={backlinks} onSelect={(signId) => setSelectedSignId(signId)} />
        </div>
      )}
    </section>
  );
}
