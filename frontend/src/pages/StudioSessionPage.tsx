/**
 * StudioSessionPage — page d'une session d'annotation.
 *
 * - Header: nom + statut de la session
 * - Upload de vidéos (drag & drop)
 * - Liste des vidéos avec état d'annotation
 * - Bouton "Annoter" → VideoAnnotationPage
 * - Bouton "Lancer l'entraînement" → /training
 */

import {
  ArrowLeft,
  CheckCircle,
  Clapperboard,
  Film,
  GraduationCap,
  Pencil,
  Tag,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { apiBaseUrl } from "../api/client";
import type { VideoInSession } from "../api/studio";
import { uploadVideosToSession } from "../api/studio";
import { useStudioStore } from "../stores/studioStore";

export function StudioSessionPage(): JSX.Element {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const {
    currentSession,
    sessionVideos,
    isLoadingSessions,
    error,
    selectSession,
    fetchSessionVideos: loadVideos,
  } = useStudioStore();

  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const numId = Number(sessionId);

  useEffect(() => {
    if (numId) {
      void selectSession(numId);
    }
  }, [numId, selectSession]);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files).filter((f) => f.type.startsWith("video/"));
      if (!arr.length) return;
      setUploading(true);
      try {
        await uploadVideosToSession(numId, arr);
        await loadVideos(numId);
      } catch {
        // error visible via store
      } finally {
        setUploading(false);
      }
    },
    [numId, loadVideos]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    void handleFiles(e.dataTransfer.files);
  };

  const statusBadge =
    currentSession?.status === "active"
      ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40"
      : currentSession?.status === "completed"
      ? "bg-sky-500/20 text-sky-300 ring-1 ring-sky-500/40"
      : "bg-slate-500/20 text-slate-400 ring-1 ring-slate-500/40";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-800/80 px-6 py-4">
        <button
          onClick={() => navigate("/studio")}
          className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-700 bg-slate-800/60 text-slate-300 transition hover:bg-slate-700"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary/30 to-secondary/30 text-white ring-1 ring-primary/40">
          <Clapperboard className="h-5 w-5" />
        </div>
        <div className="flex-1">
          {isLoadingSessions ? (
            <div className="h-5 w-48 animate-pulse rounded bg-slate-800" />
          ) : (
            <div className="flex items-center gap-2">
              <h1 className="font-display text-xl font-semibold text-white">
                {currentSession?.name ?? "Session…"}
              </h1>
              {currentSession && (
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusBadge}`}
                >
                  {currentSession.status}
                </span>
              )}
            </div>
          )}
          {currentSession?.description && (
            <p className="text-xs text-slate-400">{currentSession.description}</p>
          )}
        </div>
        <button
          onClick={() => navigate("/training")}
          className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-primary to-secondary px-4 py-2 text-sm font-medium text-white transition hover:opacity-90"
        >
          <GraduationCap className="h-4 w-4" />
          Lancer l'entraînement
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-6 mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Drop zone */}
      <div className="p-6">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed py-8 transition ${
            dragOver
              ? "border-primary bg-primary/10"
              : "border-slate-700 bg-slate-900/40 hover:border-primary/50"
          }`}
        >
          <Upload
            className={`h-8 w-8 transition ${dragOver ? "text-primary" : "text-slate-500"}`}
          />
          <div className="text-center">
            <p className="text-sm font-medium text-slate-300">
              Déposer des vidéos ici
            </p>
            <p className="text-xs text-slate-500">
              ou cliquer pour choisir des fichiers MP4/WebM/MOV
            </p>
          </div>
          {uploading && (
            <p className="text-sm text-primary">Envoi en cours…</p>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && void handleFiles(e.target.files)}
          />
        </div>
      </div>

      {/* Video list */}
      <div className="flex-1 overflow-y-auto px-6 pb-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-slate-200">
            Vidéos ({sessionVideos.length})
          </h2>
        </div>

        {sessionVideos.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-slate-500">
            <Film className="h-10 w-10 opacity-30" />
            <p className="text-sm">Aucune vidéo dans cette session.</p>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {sessionVideos.map((video) => (
              <VideoCard
                key={video.id}
                video={video}
                sessionId={numId}
                onAnnotate={() =>
                  navigate(
                    `/studio/sessions/${numId}/videos/${video.id}/annotate`
                  )
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VideoCard inline component
// ---------------------------------------------------------------------------

function VideoCard({
  video,
  sessionId,
  onAnnotate,
}: {
  video: VideoInSession;
  sessionId: number;
  onAnnotate: () => void;
}): JSX.Element {
  const base = apiBaseUrl();
  const thumbSrc = video.thumbnail_path
    ? video.thumbnail_path.startsWith("http")
      ? video.thumbnail_path
      : `${base}${video.thumbnail_path}`
    : null;

  const progressPercent =
    video.annotation_count > 0
      ? Math.round((video.verified_count / video.annotation_count) * 100)
      : 0;

  return (
    <div className="overflow-hidden rounded-xl border border-slate-700/70 bg-slate-900/60 transition hover:border-primary/40">
      {/* Thumbnail */}
      <div className="relative aspect-video w-full overflow-hidden bg-slate-800">
        {thumbSrc ? (
          <img src={thumbSrc} alt="thumbnail" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <Film className="h-8 w-8 text-slate-600" />
          </div>
        )}
        {video.landmarks_extracted && (
          <span className="absolute right-1 top-1 rounded bg-emerald-600/80 px-1.5 py-0.5 text-[10px] font-medium text-white">
            Landmarks
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-3">
        <p className="truncate text-sm font-medium text-slate-200">
          {video.file_path.split("/").pop()}
        </p>
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-slate-500">
          <span>{(video.duration_ms / 1000).toFixed(1)}s</span>
          <span>{video.fps} fps</span>
          <span>{video.resolution}</span>
        </div>

        {/* Annotation stats */}
        <div className="mt-2 flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1 text-slate-400">
            <Tag className="h-3 w-3" />
            {video.annotation_count} annotation{video.annotation_count !== 1 ? "s" : ""}
          </span>
          {video.verified_count > 0 && (
            <span className="flex items-center gap-1 text-emerald-400">
              <CheckCircle className="h-3 w-3" />
              {video.verified_count} vérifiée{video.verified_count !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {/* Progress bar */}
        {video.annotation_count > 0 && (
          <div className="mt-2">
            <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary to-secondary"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="mt-3 flex gap-2">
          <button
            onClick={onAnnotate}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary/20 px-3 py-1.5 text-xs font-medium text-primary ring-1 ring-primary/30 transition hover:bg-primary/30"
          >
            <Pencil className="h-3.5 w-3.5" />
            Annoter
          </button>
        </div>
      </div>
    </div>
  );
}
