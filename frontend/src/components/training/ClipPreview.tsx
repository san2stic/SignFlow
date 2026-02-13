import { useEffect, useRef, useState } from "react";
import type { LandmarkFrame } from "../../lib/mediapipe";
import { useMediaPipe } from "../../hooks/useMediaPipe";
import { LandmarkOverlay } from "../camera/LandmarkOverlay";

interface ClipPreviewProps {
  url: string;
  durationMs: number;
  quality: "valid" | "low";
  qualityReasons: string[];
  onRemove: () => void;
}

export function ClipPreview({ url, durationMs, quality, qualityReasons, onRemove }: ClipPreviewProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frame, setFrame] = useState<LandmarkFrame | null>(null);

  // Activer MediaPipe uniquement quand la vidéo est en lecture
  const { frame: detectedFrame } = useMediaPipe({
    videoRef,
    enabled: isPlaying,
    modelComplexity: 0, // Mode lite pour performances
    targetFps: 30
  });

  // Mettre à jour le frame détecté
  useEffect(() => {
    if (detectedFrame) {
      setFrame(detectedFrame);
    }
  }, [detectedFrame]);

  // Écouter les événements de lecture
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      setFrame(null);
    };

    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);
    video.addEventListener("ended", handleEnded);

    return () => {
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
      video.removeEventListener("ended", handleEnded);
    };
  }, []);

  return (
    <article className="rounded-btn border border-slate-700 bg-slate-900/60 p-2">
      <div className="relative mb-2 h-24 w-full overflow-hidden rounded-btn bg-black">
        <video
          ref={videoRef}
          src={url}
          controls
          className="h-full w-full object-cover"
          crossOrigin="anonymous"
        />
        {isPlaying && (
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ mixBlendMode: "screen" }}
          />
        )}
        {isPlaying && frame && <LandmarkOverlay frame={frame} showConfidenceIndicator={false} />}
      </div>

      <p className="text-xs text-slate-300">{(durationMs / 1000).toFixed(1)}s</p>
      <p className={`text-xs ${quality === "valid" ? "text-secondary" : "text-accent"}`}>
        {quality === "valid" ? "Quality: OK" : "Quality: low"}
      </p>

      {quality === "low" && qualityReasons.length > 0 && (
        <p className="mt-1 text-[11px] text-slate-400">{qualityReasons.join(" · ")}</p>
      )}

      <button className="mt-2 w-full rounded-btn bg-slate-700 px-2 py-1 text-xs hover:bg-slate-600" onClick={onRemove}>
        Remove
      </button>
    </article>
  );
}
