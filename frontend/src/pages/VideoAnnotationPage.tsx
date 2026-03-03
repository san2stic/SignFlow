/**
 * VideoAnnotationPage — éditeur d'annotation ELAN-like.
 *
 * Layout:
 * ┌─────────────────────────────────────────┐
 * │  Contrôles vidéo + timecode             │
 * ├─────────────────────────────────────────┤
 * │  Lecteur vidéo + canvas overlay         │
 * ├─────────────────────────────────────────┤
 * │  Timeline multi-piste                   │
 * ├─────────────────────────────────────────┤
 * │  Formulaire ajout annotation            │
 * └─────────────────────────────────────────┘
 */

import {
  ArrowLeft,
  CheckCircle,
  Download,
  Pause,
  Play,
  Plus,
  SkipBack,
  SkipForward,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import type { NMMTags } from "../api/studio";
import { exportSessionUrl } from "../api/studio";
import { AnnotationTimeline } from "../components/studio/AnnotationTimeline";
import { NMMIndicator } from "../components/studio/NMMIndicator";
import { SignSelector } from "../components/studio/SignSelector";
import { VideoPlayer } from "../components/studio/VideoPlayer";
import { useStudioStore } from "../stores/studioStore";

function msToTimecode(ms: number): string {
  const totalSeconds = ms / 1000;
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  const cs = Math.floor((ms % 1000) / 10);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

export function VideoAnnotationPage(): JSX.Element {
  const { sessionId, videoId } = useParams<{
    sessionId: string;
    videoId: string;
  }>();
  const navigate = useNavigate();
  const numSessionId = Number(sessionId);

  const {
    currentSession,
    sessionVideos,
    annotations,
    selectedAnnotation,
    currentTime,
    isPlaying,
    isLoadingAnnotations,
    selectSession,
    fetchAnnotations,
    addAnnotation,
    updateAnnotation,
    deleteAnnotation,
    selectAnnotation,
    autoSuggestAnnotations,
    setCurrentTime,
    setIsPlaying,
  } = useStudioStore();

  // Duration state (loaded from video element)
  const [durationMs, setDurationMs] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);

  // Annotation form state
  const [newSignLabel, setNewSignLabel] = useState("");
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [markEnd, setMarkEnd] = useState<number | null>(null);
  const [isAutoSuggesting, setIsAutoSuggesting] = useState(false);

  // Load session and annotations
  useEffect(() => {
    if (!numSessionId || !videoId) return;
    void selectSession(numSessionId);
    void fetchAnnotations(videoId, numSessionId);
  }, [numSessionId, videoId, selectSession, fetchAnnotations]);

  // Find current video in session
  const currentVideo =
    sessionVideos.find((v) => v.id === videoId) ?? null;

  const handleTimeUpdate = useCallback(
    (ms: number) => setCurrentTime(ms),
    [setCurrentTime]
  );
  const handlePlayPause = useCallback(
    (playing: boolean) => setIsPlaying(playing),
    [setIsPlaying]
  );
  const handleSeek = useCallback(
    (ms: number) => setCurrentTime(ms),
    [setCurrentTime]
  );

  const handleAnnotationUpdate = useCallback(
    (id: number, start_ms: number, end_ms: number) => {
      const ann = annotations.find((a) => a.id === id);
      if (!ann) return;
      const fps = currentVideo?.fps ?? 30;
      void updateAnnotation(id, {
        start_time_ms: start_ms,
        end_time_ms: end_ms,
        start_frame: Math.round((start_ms / 1000) * fps),
        end_frame: Math.round((end_ms / 1000) * fps),
      });
    },
    [annotations, currentVideo, updateAnnotation]
  );

  const handleAddAnnotation = () => {
    if (!newSignLabel.trim() || !videoId) return;
    const fps = currentVideo?.fps ?? 30;
    const start = markStart ?? Math.max(0, currentTime - 500);
    const end = markEnd ?? Math.min(durationMs, currentTime + 500);
    void addAnnotation(videoId, numSessionId, {
      sign_label: newSignLabel.toUpperCase(),
      start_time_ms: start,
      end_time_ms: end,
      start_frame: Math.round((start / 1000) * fps),
      end_frame: Math.round((end / 1000) * fps),
    });
    setMarkStart(null);
    setMarkEnd(null);
    setNewSignLabel("");
  };

  const handleAutoSuggest = async () => {
    if (!videoId) return;
    setIsAutoSuggesting(true);
    await autoSuggestAnnotations(videoId, numSessionId);
    setIsAutoSuggesting(false);
  };

  const handleVerifySelected = () => {
    if (!selectedAnnotation) return;
    void updateAnnotation(selectedAnnotation.id, {
      is_verified: !selectedAnnotation.is_verified,
    });
  };

  const handleDeleteSelected = () => {
    if (!selectedAnnotation) return;
    if (window.confirm(`Supprimer l'annotation "${selectedAnnotation.sign_label}" ?`)) {
      void deleteAnnotation(selectedAnnotation.id);
    }
  };

  const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];

  return (
    <div className="flex h-full flex-col overflow-hidden bg-slate-950">
      {/* Top bar */}
      <div className="flex items-center gap-3 border-b border-slate-800/80 px-4 py-2">
        <button
          onClick={() => navigate(`/studio/sessions/${sessionId}`)}
          className="flex h-7 w-7 items-center justify-center rounded border border-slate-700 bg-slate-800/60 text-slate-300 transition hover:bg-slate-700"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
        </button>
        <div className="flex-1 truncate">
          <span className="text-xs text-slate-500">
            {currentSession?.name ?? "Session"} /
          </span>
          <span className="ml-1 text-sm font-medium text-slate-200">
            {videoId ? videoId.slice(0, 8) + "…" : "—"}
          </span>
        </div>

        {/* Export */}
        <div className="flex items-center gap-1">
          {(["json", "csv", "elan"] as const).map((fmt) => (
            <a
              key={fmt}
              href={exportSessionUrl(numSessionId, fmt)}
              download
              className="flex items-center gap-1 rounded border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs text-slate-300 transition hover:bg-slate-700"
            >
              <Download className="h-3 w-3" />
              {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main annotation area */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Video controls */}
          <div className="flex items-center gap-2 border-b border-slate-800/60 bg-slate-900/60 px-4 py-2">
            <button
              onClick={() => setCurrentTime(0)}
              className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <SkipBack className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-white hover:bg-primary/80"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </button>
            <button
              onClick={() => setCurrentTime(durationMs)}
              className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <SkipForward className="h-4 w-4" />
            </button>

            {/* Timecode */}
            <span className="font-mono text-sm text-slate-300">
              {msToTimecode(currentTime)}{" "}
              <span className="text-slate-600">/</span>{" "}
              {msToTimecode(durationMs)}
            </span>

            {/* Mark start / end */}
            <button
              onClick={() => setMarkStart(currentTime)}
              className={`rounded px-2 py-1 text-xs font-medium transition ${
                markStart !== null
                  ? "bg-primary/30 text-primary"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
              title="Marquer le début (I)"
            >
              [{markStart !== null ? msToTimecode(markStart) : "I"}
            </button>
            <button
              onClick={() => setMarkEnd(currentTime)}
              className={`rounded px-2 py-1 text-xs font-medium transition ${
                markEnd !== null
                  ? "bg-primary/30 text-primary"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
              title="Marquer la fin (O)"
            >
              {markEnd !== null ? msToTimecode(markEnd) : "O"}]
            </button>

            {/* Playback rate */}
            <select
              value={playbackRate}
              onChange={(e) => setPlaybackRate(Number(e.target.value))}
              className="ml-auto rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 focus:outline-none"
            >
              {PLAYBACK_RATES.map((r) => (
                <option key={r} value={r}>
                  x{r}
                </option>
              ))}
            </select>

            {/* Landmark toggle */}
            <button
              onClick={() => setShowLandmarks((v) => !v)}
              className={`rounded px-2 py-1 text-xs transition ${
                showLandmarks
                  ? "bg-emerald-600/30 text-emerald-300 ring-1 ring-emerald-600/40"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              Landmarks
            </button>
          </div>

          {/* Video player */}
          <div className="shrink-0 px-4 pt-3">
            {currentVideo ? (
              <VideoPlayer
                videoId={currentVideo.id}
                filePath={currentVideo.file_path}
                currentTimeMs={currentTime}
                isPlaying={isPlaying}
                showLandmarks={showLandmarks}
                onTimeUpdate={handleTimeUpdate}
                onPlayPause={handlePlayPause}
                onDurationLoaded={setDurationMs}
              />
            ) : (
              <div className="flex aspect-video w-full items-center justify-center rounded-lg bg-slate-900 text-slate-600">
                {sessionVideos.length === 0 ? "Chargement de la vidéo…" : "Vidéo non trouvée"}
              </div>
            )}
          </div>

          {/* Timeline */}
          <div className="shrink-0 px-4 pt-2">
            <AnnotationTimeline
              annotations={annotations}
              durationMs={durationMs}
              currentTimeMs={currentTime}
              selectedAnnotationId={selectedAnnotation?.id ?? null}
              onSelectAnnotation={selectAnnotation}
              onUpdateAnnotation={handleAnnotationUpdate}
              onSeek={handleSeek}
            />
          </div>

          {/* Annotation form */}
          <div className="shrink-0 border-t border-slate-800/60 px-4 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="w-48">
                <SignSelector
                  value={newSignLabel}
                  onChange={setNewSignLabel}
                  placeholder="Signe…"
                />
              </div>
              <button
                onClick={handleAddAnnotation}
                disabled={!newSignLabel.trim()}
                className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white transition hover:bg-primary/80 disabled:opacity-50"
              >
                <Plus className="h-4 w-4" />
                Ajouter
              </button>
              <button
                onClick={() => void handleAutoSuggest()}
                disabled={isAutoSuggesting}
                className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm font-medium text-amber-300 transition hover:bg-amber-500/20 disabled:opacity-50"
              >
                <Sparkles className="h-4 w-4" />
                {isAutoSuggesting ? "Analyse…" : "Auto-suggest"}
              </button>

              {/* Selected annotation actions */}
              {selectedAnnotation && (
                <div className="ml-auto flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-1.5">
                  <span className="text-sm font-medium text-white">
                    {selectedAnnotation.sign_label}
                  </span>
                  <span className="text-xs text-slate-500">
                    {msToTimecode(selectedAnnotation.start_time_ms)} →{" "}
                    {msToTimecode(selectedAnnotation.end_time_ms)}
                  </span>
                  <button
                    onClick={handleVerifySelected}
                    title={
                      selectedAnnotation.is_verified
                        ? "Retirer la vérification"
                        : "Marquer comme vérifiée"
                    }
                    className={`rounded p-1 transition ${
                      selectedAnnotation.is_verified
                        ? "text-emerald-400 hover:text-slate-400"
                        : "text-slate-500 hover:text-emerald-400"
                    }`}
                  >
                    <CheckCircle className="h-4 w-4" />
                  </button>
                  <button
                    onClick={handleDeleteSelected}
                    className="rounded p-1 text-slate-500 transition hover:text-red-400"
                    title="Supprimer"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right sidebar — annotation list */}
        <AnnotationSidebar
          annotations={annotations}
          loading={isLoadingAnnotations}
          selectedId={selectedAnnotation?.id ?? null}
          onSelect={selectAnnotation}
          onVerify={(id) =>
            void updateAnnotation(id, {
              is_verified: !annotations.find((a) => a.id === id)?.is_verified,
            })
          }
          onDelete={(id) => {
            const ann = annotations.find((a) => a.id === id);
            if (ann && window.confirm(`Supprimer "${ann.sign_label}" ?`)) {
              void deleteAnnotation(id);
            }
          }}
          onSeek={(ms) => setCurrentTime(ms)}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Annotation sidebar
// ---------------------------------------------------------------------------

import type { VideoAnnotation } from "../api/studio";

function AnnotationSidebar({
  annotations,
  loading,
  selectedId,
  onSelect,
  onVerify,
  onDelete,
  onSeek,
}: {
  annotations: VideoAnnotation[];
  loading: boolean;
  selectedId: number | null;
  onSelect: (ann: VideoAnnotation) => void;
  onVerify: (id: number) => void;
  onDelete: (id: number) => void;
  onSeek: (ms: number) => void;
}): JSX.Element {
  return (
    <div className="flex w-64 shrink-0 flex-col overflow-hidden border-l border-slate-800/80 bg-slate-950/70">
      <div className="border-b border-slate-800/60 px-3 py-2">
        <p className="text-xs font-medium text-slate-400">
          Annotations ({annotations.length})
        </p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="px-3 py-6 text-center text-xs text-slate-600">
            Chargement…
          </div>
        ) : annotations.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-slate-600">
            Aucune annotation
          </div>
        ) : (
          annotations.map((ann) => (
            <div
              key={ann.id}
              onClick={() => {
                onSelect(ann);
                onSeek(ann.start_time_ms);
              }}
              className={`group flex cursor-pointer items-start gap-2 border-b border-slate-800/40 px-3 py-2 transition ${
                ann.id === selectedId
                  ? "bg-primary/10 ring-l-2 ring-primary"
                  : "hover:bg-slate-900/60"
              }`}
            >
              <div className="flex-1 min-w-0">
                <p className="truncate text-sm font-medium text-white">
                  {ann.sign_label}
                </p>
                <p className="text-[10px] text-slate-500">
                  {msToTimecode(ann.start_time_ms)} →{" "}
                  {msToTimecode(ann.end_time_ms)}
                </p>
                {ann.confidence !== null && ann.confidence !== undefined && (
                  <p className="text-[10px] text-slate-600">
                    conf: {(ann.confidence * 100).toFixed(0)}%
                  </p>
                )}
                {ann.nmm_tags && (
                  <NMMIndicator tags={ann.nmm_tags} compact />
                )}
              </div>
              <div className="flex shrink-0 flex-col items-center gap-1 opacity-0 transition group-hover:opacity-100">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onVerify(ann.id);
                  }}
                  className={`rounded p-0.5 transition ${
                    ann.is_verified
                      ? "text-emerald-400"
                      : "text-slate-600 hover:text-emerald-400"
                  }`}
                >
                  <CheckCircle className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(ann.id);
                  }}
                  className="rounded p-0.5 text-slate-600 transition hover:text-red-400"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
