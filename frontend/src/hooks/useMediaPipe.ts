import { useEffect, useRef, useState, type RefObject } from "react";

import { Holistic } from "@mediapipe/holistic";

import { frameFromHolisticResult, type LandmarkFrame } from "../lib/mediapipe";

interface UseMediaPipeOptions {
  videoRef: RefObject<HTMLVideoElement>;
  enabled: boolean;
  targetFps?: number;
  includeFace?: boolean;
  modelComplexity?: 0 | 1 | 2; // 0=lite, 1=full, 2=heavy (most accurate)
}

export function useMediaPipe({
  videoRef,
  enabled,
  targetFps = 30, // Increased from 12 to 30 for smoother sign language capture
  includeFace = false,
  modelComplexity = 2 // Use highest quality model by default for best accuracy
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
      modelComplexity: modelComplexity, // Use configurable model complexity (default: 2 = highest)
      smoothLandmarks: true, // Keep landmark smoothing for temporal consistency
      enableSegmentation: false, // Not needed for sign language
      refineFaceLandmarks: true, // Enable for better facial expression detection
      minDetectionConfidence: 0.7, // Increased from 0.5 to 0.7 for higher precision
      minTrackingConfidence: 0.7 // Increased from 0.5 to 0.7 for better tracking
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
  }, [enabled, includeFace, targetFps, videoRef]);

  return { frame, ready };
}

export type { LandmarkFrame };
