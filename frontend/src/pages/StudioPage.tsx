/**
 * StudioPage — page principale du Studio d'annotation SignFlow.
 *
 * Sidebar: liste des sessions
 * Zone principale: tabs Vidéos / Annotations / Entraînement
 */

import { Plus, RefreshCw, Clapperboard } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { AnnotationSessionWithStats } from "../api/studio";
import { SessionCard } from "../components/studio/SessionCard";
import { useStudioStore } from "../stores/studioStore";

type TabKey = "sessions" | "stats";

export function StudioPage(): JSX.Element {
  const navigate = useNavigate();
  const { sessions, isLoadingSessions, error, fetchSessions, createSession } =
    useStudioStore();
  const [activeTab, setActiveTab] = useState<TabKey>("sessions");
  const [showNewSessionForm, setShowNewSessionForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    void fetchSessions();
  }, [fetchSessions]);

  const handleCreateSession = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const session = await createSession({
        name: newName.trim(),
        description: newDesc.trim() || undefined,
      });
      setShowNewSessionForm(false);
      setNewName("");
      setNewDesc("");
      navigate(`/studio/sessions/${session.id}`);
    } finally {
      setCreating(false);
    }
  };

  const filtered = activeTab === "sessions" ? sessions : sessions.filter((s) => s.status === "active");

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-800/80 px-6 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary/30 to-secondary/30 text-white ring-1 ring-primary/40">
          <Clapperboard className="h-5 w-5" />
        </div>
        <div>
          <h1 className="font-display text-xl font-semibold text-white">Studio d'annotation</h1>
          <p className="text-xs text-slate-400">Importez, annotez et entraînez des séquences LSFB</p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => void fetchSessions()}
            disabled={isLoadingSessions}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-sm text-slate-300 transition hover:bg-slate-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${isLoadingSessions ? "animate-spin" : ""}`} />
            Actualiser
          </button>
          <button
            onClick={() => setShowNewSessionForm(true)}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition hover:bg-primary/80"
          >
            <Plus className="h-4 w-4" />
            Nouvelle session
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* New session form */}
      {showNewSessionForm && (
        <div className="mx-6 mt-4 rounded-xl border border-primary/30 bg-slate-900/70 p-4">
          <h3 className="mb-3 font-semibold text-white">Nouvelle session</h3>
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">
                Nom <span className="text-red-400">*</span>
              </label>
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void handleCreateSession()}
                placeholder="Ex: Corpus LSFB — expressions faciales"
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:border-primary/60 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">
                Description
              </label>
              <textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Description optionnelle…"
                rows={2}
                className="w-full resize-none rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:border-primary/60 focus:outline-none"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => void handleCreateSession()}
                disabled={creating || !newName.trim()}
                className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-primary/80 disabled:opacity-50"
              >
                {creating ? "Création…" : "Créer"}
              </button>
              <button
                onClick={() => {
                  setShowNewSessionForm(false);
                  setNewName("");
                  setNewDesc("");
                }}
                className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 transition hover:bg-slate-700"
              >
                Annuler
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-800/60 px-6 pt-4">
        {(["sessions", "stats"] as TabKey[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-t-lg px-4 py-2 text-sm font-medium transition ${
              activeTab === tab
                ? "border-b-2 border-primary text-white"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {tab === "sessions" ? "Sessions" : "Statistiques"}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === "sessions" ? (
          <>
            {isLoadingSessions && sessions.length === 0 ? (
              <div className="flex items-center justify-center py-16 text-slate-500">
                Chargement…
              </div>
            ) : sessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-3 py-16 text-slate-500">
                <Clapperboard className="h-12 w-12 opacity-30" />
                <p className="text-sm">Aucune session d'annotation.</p>
                <button
                  onClick={() => setShowNewSessionForm(true)}
                  className="mt-2 text-sm text-primary hover:underline"
                >
                  Créer votre première session →
                </button>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {filtered.map((session) => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onClick={() => navigate(`/studio/sessions/${session.id}`)}
                  />
                ))}
              </div>
            )}
          </>
        ) : (
          <StudioStatsPanel sessions={sessions} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline stats panel
// ---------------------------------------------------------------------------

function StudioStatsPanel({
  sessions,
}: {
  sessions: AnnotationSessionWithStats[];
}): JSX.Element {
  const total = sessions.length;
  const active = sessions.filter((s) => s.status === "active").length;
  const totalAnnotations = sessions.reduce((s, x) => s + x.annotation_count, 0);
  const verifiedAnnotations = sessions.reduce((s, x) => s + x.verified_count, 0);
  const totalVideos = sessions.reduce((s, x) => s + x.video_count, 0);

  const stats = [
    { label: "Sessions totales", value: total },
    { label: "Sessions actives", value: active },
    { label: "Vidéos annotées", value: totalVideos },
    { label: "Annotations totales", value: totalAnnotations },
    { label: "Annotations vérifiées", value: verifiedAnnotations },
    {
      label: "Taux vérification",
      value: totalAnnotations > 0 ? `${Math.round((verifiedAnnotations / totalAnnotations) * 100)}%` : "—",
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-4"
        >
          <p className="text-xs text-slate-400">{stat.label}</p>
          <p className="mt-1 text-2xl font-bold text-white">{stat.value}</p>
        </div>
      ))}
    </div>
  );
}
