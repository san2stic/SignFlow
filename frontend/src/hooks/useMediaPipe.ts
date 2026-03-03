import { useEffect, useRef, useState, type RefObject } from "react";

import { Holistic } from "@mediapipe/holistic";

import { frameFromHolisticResult, type LandmarkFrame } from "../lib/mediapipe";

interface UseMediaPipeOptions {
  videoRef: RefObject<HTMLVideoElement>;
  enabled: boolean;
  targetFps?: number;
  includeFace?: boolean;
  modelComplexity?: 0 | 1 | 2; // 0=lite, 1=full, 2=heavy (most accurate)
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
}

export function useMediaPipe({
  videoRef,
  enabled,
  targetFps = 30, // Increased from 12 to 30 for smoother sign language capture
  includeFace = false,
  modelComplexity = 1, // Fix: abaissé de 2 → 1 (~30-40 ms/frame vs ~60-80 ms)
  minDetectionConfidence = 0.7,
  minTrackingConfidence = 0.7
}: UseMediaPipeOptions): { frame: LandmarkFrame | null; ready: boolean } {
  const [frame, setFrame] = useState<LandmarkFrame | null>(null);
  const [ready, setReady] = useState(false);
  const frameIndex = useRef(0);

  useEffect(() => {
    if (!enabled) {
      setFrame(null);
      setReady(false);
      return;
    }

    const video = videoRef.current;
    if (!video) {
      setReady(false);
      return;
    }

    let disposed = false;
    let rafId = 0;
    let lastTick = 0;
    let processing = false;

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
      modelComplexity: modelComplexity, // Fix: défaut 1 (full) au lieu de 2 (heavy)
      smoothLandmarks: true, // Keep landmark smoothing for temporal consistency
      enableSegmentation: false, // Not needed for sign language
      refineFaceLandmarks: false, // Fix: false → évite ~10-15 ms de surcoût inutile
      minDetectionConfidence, // Configurable by context (training can be more permissive)
      minTrackingConfidence // Configurable by context (training can be more permissive)
    });

    holistic.onResults((results) => {
      if (disposed) return;
      frameIndex.current += 1;
      setFrame(frameFromHolisticResult(results, frameIndex.current, includeFace));
      setReady(true);
    });

    const targetFrameDelay = 1000 / targetFps;

    const tick = async () => {
      if (disposed) return;

      const now = performance.now();
      const videoReady = video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;

      if (videoReady && !processing && now - lastTick >= targetFrameDelay) {
        processing = true;
        lastTick = now;
        try {
          await holistic.send({ image: video });
        } catch {
          // Swallow transient MediaPipe errors while camera stream stabilizes.
        } finally {
          processing = false;
        }
      }

      rafId = window.requestAnimationFrame(() => {
        void tick();
      });
    };

    rafId = window.requestAnimationFrame(() => {
      void tick();
    });

    return () => {
      disposed = true;
      window.cancelAnimationFrame(rafId);
      void holistic.close();
    };
  }, [enabled, includeFace, minDetectionConfidence, minTrackingConfidence, modelComplexity, targetFps, videoRef]);

  return { frame, ready };
}

export type { LandmarkFrame };
