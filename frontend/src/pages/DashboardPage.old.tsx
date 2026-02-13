import { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { exportDictionary, importDictionary } from "../api/dictionary";
import { activateModel, exportModel, listModels, type ModelVersion } from "../api/models";
import { getAccuracyHistory, getOverviewStats, getSignsPerCategory, type AccuracyHistoryPoint, type CategoryCount, type OverviewStats } from "../api/stats";
import { listTrainingSessions, type TrainingSession } from "../api/training";

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

export function DashboardPage(): JSX.Element {
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<AccuracyHistoryPoint[]>([]);
  const [categoryCounts, setCategoryCounts] = useState<CategoryCount[]>([]);
  const [activatingModelId, setActivatingModelId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  const refresh = useCallback(async () => {
    const [overviewResult, modelResult, sessionResult, historyResult, categoriesResult] = await Promise.allSettled([
      getOverviewStats(),
      listModels(),
      listTrainingSessions(),
      getAccuracyHistory(60),
      getSignsPerCategory()
    ]);

    setOverview(overviewResult.status === "fulfilled" ? overviewResult.value : null);
    setModels(modelResult.status === "fulfilled" ? modelResult.value : []);
    setSessions(sessionResult.status === "fulfilled" ? sessionResult.value.slice(0, 8) : []);
    setAccuracyHistory(historyResult.status === "fulfilled" ? historyResult.value : []);
    setCategoryCounts(categoriesResult.status === "fulfilled" ? categoriesResult.value : []);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const activeModel = useMemo(() => models.find((model) => model.is_active) ?? null, [models]);

  const onActivateModel = async (modelId: string): Promise<void> => {
    setActivatingModelId(modelId);
    setError(null);
    try {
      await activateModel(modelId);
      setStatus("Model activated.");
      await refresh();
    } catch (activateError) {
      setError(activateError instanceof Error ? activateError.message : "Activation failed.");
    } finally {
      setActivatingModelId(null);
    }
  };

  const onExportDictionary = async (format: "json" | "markdown" | "obsidian-vault"): Promise<void> => {
    setIsBusy(true);
    setError(null);
    try {
      const blob = await exportDictionary(format);
      downloadBlob(`signflow-dictionary-${format}.zip`, blob);
      setStatus(`Dictionary exported as ${format}.`);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Dictionary export failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const onImportDictionary = async (file: File): Promise<void> => {
    setIsBusy(true);
    setError(null);
    try {
      const result = await importDictionary(file);
      setStatus(
        `Dictionary import: ${result.imported_signs} signs imported, ${result.imported_notes} notes imported, ${result.skipped} skipped.`
      );
      if (result.errors.length > 0) {
        setError(result.errors.join(" | "));
      }
      await refresh();
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Dictionary import failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const onExportModel = async (): Promise<void> => {
    if (!activeModel) {
      setError("No active model to export.");
      return;
    }

    setIsBusy(true);
    setError(null);
    try {
      const result = await exportModel(activeModel.id, "pt");
      setStatus(`Model export generated at: ${result.path}`);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Model export failed.");
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <section className="space-y-4">
      <h1 className="font-heading text-2xl">Dashboard</h1>

      {status && <p className="rounded-btn border border-secondary/40 bg-secondary/10 px-3 py-2 text-sm text-secondary">{status}</p>}
      {error && <p className="rounded-btn border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Signs" value={overview?.total_signs ?? 0} />
        <Kpi label="Videos" value={overview?.total_videos ?? 0} />
        <Kpi label="Accuracy" value={`${Math.round((overview?.model_accuracy ?? 0) * 100)}%`} />
        <Kpi label="Translations" value={overview?.total_translations ?? 0} />
      </div>

      <div className="card space-y-2 p-4">
        <h2 className="font-heading text-lg">Model Accuracy Over Time</h2>
        {accuracyHistory.length === 0 ? (
          <p className="text-sm text-slate-400">No model history yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={accuracyHistory}>
              <XAxis dataKey="version" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card space-y-2 p-4">
        <h2 className="font-heading text-lg">Signs Per Category</h2>
        {categoryCounts.length === 0 ? (
          <p className="text-sm text-slate-400">No category stats yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={categoryCounts}>
              <XAxis dataKey="category" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Bar dataKey="count" fill="#6366f1" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card p-4">
        <h2 className="font-heading text-lg">Model Versions</h2>
        <ul className="mt-3 space-y-2 text-sm">
          {models.map((model) => (
            <li key={model.id} className="flex items-center justify-between rounded-btn bg-slate-800/60 px-3 py-2">
              <span>
                {model.version} {model.is_active ? "(active)" : ""}
              </span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs">{Math.round(model.accuracy * 100)}%</span>
                {!model.is_active && (
                  <button
                    className="rounded-btn bg-primary px-2 py-1 text-xs text-white disabled:bg-slate-700"
                    disabled={activatingModelId === model.id}
                    onClick={() => {
                      void onActivateModel(model.id);
                    }}
                  >
                    {activatingModelId === model.id ? "..." : "Activate"}
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div className="card p-4">
        <h2 className="font-heading text-lg">Recent Trainings</h2>
        <ul className="mt-2 space-y-2 text-sm">
          {sessions.length === 0 ? (
            <li className="text-slate-400">No sessions yet.</li>
          ) : (
            sessions.map((session) => (
              <li key={session.id} className="flex items-center justify-between">
                <span className="truncate">{session.mode}</span>
                <span className="rounded-full bg-slate-700 px-2 py-1 text-xs">{session.status}</span>
              </li>
            ))
          )}
        </ul>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        <button className="touch-btn bg-slate-700 text-white disabled:opacity-60" disabled={isBusy} onClick={() => void onExportDictionary("json")}>
          Export Dict (JSON)
        </button>
        <button className="touch-btn bg-slate-700 text-white disabled:opacity-60" disabled={isBusy} onClick={() => void onExportDictionary("obsidian-vault")}>
          Export Dict (Obsidian)
        </button>
        <label className="touch-btn cursor-pointer bg-slate-700 text-white disabled:opacity-60">
          Import Dict
          <input
            type="file"
            className="hidden"
            accept=".zip,application/zip"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void onImportDictionary(file);
              }
              event.currentTarget.value = "";
            }}
          />
        </label>
        <button className="touch-btn bg-primary text-white disabled:opacity-60" disabled={isBusy || !activeModel} onClick={() => void onExportModel()}>
          Export Active Model
        </button>
      </div>
    </section>
  );
}

function Kpi({ label, value }: { label: string; value: string | number }): JSX.Element {
  return (
    <article className="card p-4">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="font-heading text-2xl">{value}</p>
    </article>
  );
}
