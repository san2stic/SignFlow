/**
 * VideoPlayer — lecteur HTML5 avec canvas overlay pour les landmarks.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { apiBaseUrl } from "../../api/client";

interface VideoPlayerProps {
  videoId: string;
  filePath: string;
  currentTimeMs: number;
  isPlaying: boolean;
  showLandmarks?: boolean;
  onTimeUpdate: (timeMs: number) => void;
  onPlayPause: (playing: boolean) => void;
  onDurationLoaded: (durationMs: number) => void;
}

export function VideoPlayer({
  videoId: _videoId,
  filePath,
  currentTimeMs,
  isPlaying,
  showLandmarks = false,
  onTimeUpdate,
  onPlayPause,
  onDurationLoaded,
}: VideoPlayerProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);

  // Build video URL from file path or streaming endpoint
  const base = apiBaseUrl();
  const videoSrc = filePath.startsWith("http")
    ? filePath
    : filePath.startsWith("/")
    ? `${base}${filePath}`
    : `${base}/api/v1/media/${_videoId}/stream`;

  // Sync play/pause state
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.play().catch(() => onPlayPause(false));
    } else {
      video.pause();
    }
  }, [isPlaying, onPlayPause]);

  // Sync seek from external source
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const diff = Math.abs(video.currentTime * 1000 - currentTimeMs);
    if (diff > 150) {
      video.currentTime = currentTimeMs / 1000;
    }
  }, [currentTimeMs]);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    onTimeUpdate(video.currentTime * 1000);
  }, [onTimeUpdate]);

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    onDurationLoaded(video.duration * 1000);
    setVideoWidth(video.videoWidth);
    setVideoHeight(video.videoHeight);
  }, [onDurationLoaded]);

  // Draw landmark overlay (placeholder — would need actual landmark data)
  useEffect(() => {
    if (!showLandmarks) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // TODO: draw actual landmarks when landmark data is loaded
  }, [showLandmarks, currentTimeMs, videoWidth, videoHeight]);

  return (
    <div
      className="relative overflow-hidden rounded-lg bg-black"
      style={{ aspectRatio: videoWidth && videoHeight ? `${videoWidth}/${videoHeight}` : "16/9" }}
    >
      <video
        ref={videoRef}
        src={videoSrc}
        className="h-full w-full object-contain"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => onPlayPause(true)}
        onPause={() => onPlayPause(false)}
        onEnded={() => onPlayPause(false)}
        preload="metadata"
        playsInline
      />
      {showLandmarks && (
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
          width={videoWidth}
          height={videoHeight}
        />
      )}
    </div>
  );
}
