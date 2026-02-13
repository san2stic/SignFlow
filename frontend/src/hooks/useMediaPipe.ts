import { useEffect, useRef, useState, type RefObject } from "react";

import { Holistic } from "@mediapipe/holistic";

import { frameFromHolisticResult, type LandmarkFrame } from "../lib/mediapipe";

interface UseMediaPipeOptions {
  videoRef: RefObject<HTMLVideoElement>;
  enabled: boolean;
  targetFps?: number;
  includeFace?: boolean;
}

export function useMediaPipe({
  videoRef,
  enabled,
  targetFps = 12,
  includeFace = false
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
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
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
