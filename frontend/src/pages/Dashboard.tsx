import { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { motion } from "framer-motion";
import { X, Languages } from "lucide-react";

import { exportDictionary, importDictionary } from "../api/dictionary";
import { activateModel, exportModel, listModels, type ModelVersion } from "../api/models";
import { getAccuracyHistory, getOverviewStats, getSignsPerCategory, type AccuracyHistoryPoint, type CategoryCount, type OverviewStats } from "../api/stats";
import { listTrainingSessions, type TrainingSession } from "../api/training";
import { useAuthStore } from "../stores/authStore";

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

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export default function Dashboard(): JSX.Element {
  const user = useAuthStore((state) => state.user);
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<AccuracyHistoryPoint[]>([]);
  const [categoryCounts, setCategoryCounts] = useState<CategoryCount[]>([]);
  const [activatingModelId, setActivatingModelId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [showTranslatePanel, setShowTranslatePanel] = useState(true);

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
      setStatus("Modèle activé avec succès.");
      await refresh();
    } catch (activateError) {
      setError(activateError instanceof Error ? activateError.message : "Échec de l'activation.");
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
      setStatus(`Dictionnaire exporté au format ${format}.`);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Échec de l'export du dictionnaire.");
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
        `Import : ${result.imported_signs} signes, ${result.imported_notes} notes, ${result.skipped} ignorés.`
      );
      if (result.errors.length > 0) {
        setError(result.errors.join(" | "));
      }
      await refresh();
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Échec de l'import du dictionnaire.");
    } finally {
      setIsBusy(false);
    }
  };

  const onExportModel = async (): Promise<void> => {
    if (!activeModel) {
      setError("Aucun modèle actif à exporter.");
      return;
    }

    setIsBusy(true);
    setError(null);
    try {
      const result = await exportModel(activeModel.id, "pt");
      setStatus(`Export du modèle généré : ${result.path}`);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Échec de l'export du modèle.");
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Main Dashboard Area */}
      <div className="flex-1 overflow-y-auto">
        <motion.section
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="relative space-y-6 pb-8 p-8"
        >
          {/* Animated background orbs */}
          <div className="pointer-events-none fixed inset-0 overflow-hidden opacity-20">
            <motion.div
              className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-primary blur-3xl"
              animate={{
                x: [0, 50, 0],
                y: [0, -30, 0],
                scale: [1, 1.1, 1]
              }}
              transition={{
                duration: 15,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            <motion.div
              className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-secondary blur-3xl"
              animate={{
                x: [0, -60, 0],
                y: [0, 40, 0],
                scale: [1, 1.2, 1]
              }}
              transition={{
                duration: 18,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          </div>

          {/* Header */}
          <motion.header variants={itemVariants} className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
              <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
                <svg className="h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight">
                <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  Dashboard
                </span>
              </h1>
              <p className="text-sm text-text-tertiary mt-1">
                Bienvenue, <span className="font-semibold text-text-secondary">{user?.username || "utilisateur"}</span> !
              </p>
            </div>
          </motion.header>

          {/* Status messages */}
          {status && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="card neon-border p-4"
            >
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 animate-pulse rounded-full bg-accent shadow-glow" />
                <p className="text-sm text-accent">{status}</p>
              </div>
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="card border-red-500/30 p-4"
            >
              <div className="flex items-center gap-3">
                <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-sm text-red-300">{error}</p>
              </div>
            </motion.div>
          )}

          {/* KPI Cards */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-2 gap-4 md:grid-cols-4"
          >
            <KpiCard
              icon={
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              }
              label="Signes"
              value={overview?.total_signs ?? 0}
              gradient="from-primary to-primary-dark"
            />
            <KpiCard
              icon={
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              }
              label="Vidéos"
              value={overview?.total_videos ?? 0}
              gradient="from-secondary to-secondary-dark"
            />
            <KpiCard
              icon={
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              }
              label="Précision"
              value={`${Math.round((overview?.model_accuracy ?? 0) * 100)}%`}
              gradient="from-accent to-accent-dark"
            />
            <KpiCard
              icon={
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              }
              label="Traductions"
              value={overview?.total_translations ?? 0}
              gradient="from-primary to-secondary"
            />
          </motion.div>

          {/* Accuracy History Chart */}
          <motion.div variants={itemVariants} className="card p-6">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-accent/20 to-accent-dark/20">
                <svg className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <h2 className="font-display text-xl font-semibold">Précision du modèle dans le temps</h2>
            </div>
            {accuracyHistory.length === 0 ? (
              <div className="flex h-64 items-center justify-center">
                <p className="text-sm text-text-muted">Aucun historique de modèle disponible.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={accuracyHistory}>
                  <defs>
                    <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10B981" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="version"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    stroke="#334155"
                    strokeWidth={1}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    stroke="#334155"
                    strokeWidth={1}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(15, 23, 42, 0.95)",
                      border: "1px solid rgba(14, 165, 233, 0.3)",
                      borderRadius: "12px",
                      backdropFilter: "blur(12px)"
                    }}
                    labelStyle={{ color: "#CBD5E1", fontWeight: 600 }}
                    itemStyle={{ color: "#10B981" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#10B981"
                    strokeWidth={3}
                    dot={{ fill: "#10B981", r: 5, strokeWidth: 2, stroke: "#020617" }}
                    activeDot={{ r: 7, strokeWidth: 2 }}
                    fill="url(#accuracyGradient)"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </motion.div>

          {/* Signs Per Category Chart */}
          <motion.div variants={itemVariants} className="card p-6">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20">
                <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h2 className="font-display text-xl font-semibold">Signes par catégorie</h2>
            </div>
            {categoryCounts.length === 0 ? (
              <div className="flex h-64 items-center justify-center">
                <p className="text-sm text-text-muted">Aucune statistique de catégorie disponible.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={categoryCounts}>
                  <defs>
                    <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#0EA5E9" />
                      <stop offset="100%" stopColor="#8B5CF6" />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="category"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    stroke="#334155"
                    strokeWidth={1}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    stroke="#334155"
                    strokeWidth={1}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(15, 23, 42, 0.95)",
                      border: "1px solid rgba(14, 165, 233, 0.3)",
                      borderRadius: "12px",
                      backdropFilter: "blur(12px)"
                    }}
                    labelStyle={{ color: "#CBD5E1", fontWeight: 600 }}
                    itemStyle={{ color: "#0EA5E9" }}
                  />
                  <Bar
                    dataKey="count"
                    fill="url(#barGradient)"
                    radius={[8, 8, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </motion.div>

          {/* Model Versions */}
          <motion.div variants={itemVariants} className="card p-6">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-secondary/20 to-primary/20">
                <svg className="h-5 w-5 text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <h2 className="font-display text-xl font-semibold">Versions du modèle</h2>
            </div>
            <ul className="space-y-3">
              {models.map((model, index) => (
                <motion.li
                  key={model.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="card flex items-center justify-between p-4 transition-all hover:scale-[1.01]"
                >
                  <div className="flex items-center gap-3">
                    <div className={`h-10 w-10 rounded-lg bg-gradient-to-br ${model.is_active ? 'from-accent to-accent-dark' : 'from-surface-secondary to-surface-tertiary'} flex items-center justify-center`}>
                      <span className="font-mono text-sm font-bold">{model.version}</span>
                    </div>
                    <div>
                      <p className="font-medium">
                        {model.version}
                        {model.is_active && (
                          <span className="ml-2 rounded-full bg-accent/20 px-2 py-0.5 text-xs font-medium text-accent">
                            ACTIF
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-text-tertiary">
                        Précision : <span className="font-mono text-primary">{Math.round(model.accuracy * 100)}%</span>
                      </p>
                    </div>
                  </div>
                  {!model.is_active && (
                    <button
                      className="touch-btn bg-gradient-to-br from-primary/30 to-secondary/30 text-primary backdrop-blur-sm disabled:opacity-50"
                      disabled={activatingModelId === model.id}
                      onClick={() => {
                        void onActivateModel(model.id);
                      }}
                    >
                      {activatingModelId === model.id ? "..." : "Activer"}
                    </button>
                  )}
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Recent Trainings */}
          <motion.div variants={itemVariants} className="card p-6">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-accent/20">
                <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="font-display text-xl font-semibold">Entraînements récents</h2>
            </div>
            <ul className="space-y-2">
              {sessions.length === 0 ? (
                <li className="py-4 text-center text-sm text-text-muted">Aucune session pour le moment.</li>
              ) : (
                sessions.map((session, index) => (
                  <motion.li
                    key={session.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className="flex items-center justify-between rounded-btn bg-surface-secondary/50 px-4 py-3"
                  >
                    <span className="truncate text-sm font-medium">{session.mode}</span>
                    <span className={`rounded-full px-3 py-1 text-xs font-medium ${
                      session.status === "completed"
                        ? "bg-accent/20 text-accent"
                        : session.status === "failed"
                        ? "bg-red-500/20 text-red-400"
                        : "bg-primary/20 text-primary"
                    }`}>
                      {session.status}
                    </span>
                  </motion.li>
                ))
              )}
            </ul>
          </motion.div>

          {/* Action Buttons */}
          <motion.div
            variants={itemVariants}
            className="grid gap-3 sm:grid-cols-2"
          >
            <button
              className="touch-btn bg-gradient-to-br from-primary/30 to-secondary/30 text-primary backdrop-blur-sm disabled:opacity-50"
              disabled={isBusy}
              onClick={() => void onExportDictionary("json")}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export Dict (JSON)
              </span>
            </button>

            <button
              className="touch-btn bg-gradient-to-br from-primary/30 to-secondary/30 text-primary backdrop-blur-sm disabled:opacity-50"
              disabled={isBusy}
              onClick={() => void onExportDictionary("obsidian-vault")}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export Dict (Obsidian)
              </span>
            </button>

            <label className="touch-btn cursor-pointer bg-gradient-to-br from-accent/30 to-accent-dark/30 text-accent backdrop-blur-sm disabled:opacity-50">
              <span className="flex items-center justify-center gap-2">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Import Dict
              </span>
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

            <button
              className="touch-btn bg-gradient-to-br from-primary to-secondary text-white disabled:opacity-50"
              disabled={isBusy || !activeModel}
              onClick={() => void onExportModel()}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                </svg>
                Export Modèle Actif
              </span>
            </button>
          </motion.div>
        </motion.section>
      </div>

      {/* Live Translation Assist Panel */}
      {showTranslatePanel && (
        <aside className="w-96 bg-background-card border-l border-surface-secondary flex flex-col h-screen overflow-hidden">
          <div className="p-6 border-b border-surface-secondary flex items-center justify-between">
            <h2 className="text-lg font-bold text-text-primary">Live Translation Assist</h2>
            <button
              onClick={() => setShowTranslatePanel(false)}
              className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-surface-secondary transition-colors"
            >
              <X className="w-5 h-5 text-text-tertiary" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Video Preview Placeholder */}
            <div className="aspect-video bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl overflow-hidden relative">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-48 h-48 border-4 border-accent rounded-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-2">
                      <div className="w-12 h-12 bg-accent rounded-full"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="absolute top-4 right-4 px-3 py-1 bg-accent rounded-full flex items-center gap-2">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="text-xs font-semibold text-white">LIVE</span>
              </div>
            </div>

            {/* Real-time Translation Result */}
            <div className="bg-gradient-to-br from-accent/10 to-accent-dark/10 rounded-xl p-4 border border-accent/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-text-secondary">Confiance</span>
                <span className="text-lg font-bold text-accent">
                  {overview ? `${Math.round(overview.model_accuracy * 100)}%` : "N/A"}
                </span>
              </div>
              <p className="text-2xl font-bold text-text-primary uppercase tracking-wide">
                {overview?.most_used_signs?.[0]?.sign || "En attente..."}
              </p>
            </div>

            {/* Most Used Signs */}
            <div>
              <h3 className="text-sm font-semibold text-text-secondary mb-3">Signes populaires</h3>
              <div className="grid grid-cols-3 gap-3">
                {overview?.most_used_signs?.slice(0, 3).map((sign, idx) => (
                  <button
                    key={idx}
                    className="aspect-square bg-surface-secondary hover:bg-surface-tertiary rounded-xl flex flex-col items-center justify-center transition-colors border border-surface-tertiary"
                  >
                    <span className="text-3xl mb-2">👋</span>
                    <span className="text-xs font-semibold text-text-secondary uppercase">{sign.sign}</span>
                    <span className="text-xs text-text-muted">({sign.count})</span>
                  </button>
                )) || (
                  <div className="col-span-3 text-center text-sm text-text-muted py-4">
                    Aucune donnée disponible
                  </div>
                )}
              </div>
            </div>

            {/* Action Button */}
            <button className="w-full bg-gradient-to-r from-primary to-secondary hover:from-primary-dark hover:to-secondary-dark text-white font-semibold py-3 rounded-xl transition-all shadow-lg">
              Aller à la traduction
            </button>
          </div>
        </aside>
      )}

      {/* Floating Button to Reopen Panel */}
      {!showTranslatePanel && (
        <button
          onClick={() => setShowTranslatePanel(true)}
          className="fixed bottom-8 right-8 w-14 h-14 bg-gradient-to-br from-primary to-secondary text-white rounded-full shadow-2xl hover:scale-110 transition-transform flex items-center justify-center"
        >
          <Languages className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}

interface KpiCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  gradient: string;
}

function KpiCard({ icon, label, value, gradient }: KpiCardProps): JSX.Element {
  return (
    <motion.article
      whileHover={{ scale: 1.03, y: -4 }}
      className="card relative overflow-hidden p-6"
    >
      {/* Background gradient blob */}
      <div className={`pointer-events-none absolute -top-10 -right-10 h-32 w-32 rounded-full bg-gradient-to-br ${gradient} opacity-20 blur-2xl`} />

      <div className="relative z-10">
        <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${gradient} p-[2px]`}>
          <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-card">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {icon}
            </svg>
          </div>
        </div>
        <p className="mb-1 text-xs font-medium uppercase tracking-wider text-text-tertiary">{label}</p>
        <p className="font-display text-3xl font-bold tracking-tight">{value}</p>
      </div>
    </motion.article>
  );
}
