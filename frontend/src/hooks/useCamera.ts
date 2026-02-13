import { useCallback, useEffect, useRef, useState } from "react";

interface BufferedChunk {
  blob: Blob;
  timestamp: number;
}

interface ZoomCapability {
  min: number;
  max: number;
  step?: number;
}

function preferredRecorderMimeType(): string | null {
  const candidates = [
    "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];

  for (const candidate of candidates) {
    if (MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return null;
}

async function bindStreamToVideo(videoElement: HTMLVideoElement, stream: MediaStream): Promise<void> {
  if (videoElement.srcObject !== stream) {
    videoElement.srcObject = stream;
  }

  if (videoElement.paused) {
    await videoElement.play();
  }
}

async function getCameraStream(facingMode: "user" | "environment"): Promise<MediaStream> {
  const preferredConstraints: MediaStreamConstraints = {
    audio: false,
    video: {
      facingMode: { exact: facingMode },
      width: { ideal: 1280, min: 640 },
      height: { ideal: 960, min: 480 },
      aspectRatio: { ideal: 4 / 3 },
      frameRate: { ideal: 30, max: 30 }
    }
  };

  const fallbackConstraints: MediaStreamConstraints = {
    audio: false,
    video: {
      facingMode,
      width: 640,
      height: 480,
      frameRate: 30
    }
  };

  try {
    return await navigator.mediaDevices.getUserMedia(preferredConstraints);
  } catch {
    return navigator.mediaDevices.getUserMedia(fallbackConstraints);
  }
}

async function applyWidestAvailableZoom(stream: MediaStream): Promise<void> {
  const videoTrack = stream.getVideoTracks()[0];
  if (!videoTrack || typeof videoTrack.getCapabilities !== "function") {
    return;
  }

  const capabilities = videoTrack.getCapabilities() as MediaTrackCapabilities & { zoom?: ZoomCapability };
  const zoomCapability = capabilities.zoom;
  if (!zoomCapability || typeof zoomCapability.min !== "number") {
    return;
  }

  try {
    await videoTrack.applyConstraints({
      advanced: [{ zoom: zoomCapability.min } as MediaTrackConstraintSet]
    });
  } catch {
    // Ignore if the browser/device does not allow runtime zoom changes.
  }
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [isReady, setIsReady] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const preRollRecorderRef = useRef<MediaRecorder | null>(null);
  const preRollChunksRef = useRef<BufferedChunk[]>([]);

  const attachVideoRef = useCallback((node: HTMLVideoElement | null): void => {
    videoRef.current = node;
    if (!node || !streamRef.current) {
      return;
    }

    void bindStreamToVideo(node, streamRef.current).catch(() => {
      // Ignore transient autoplay errors until browser/user interaction stabilizes.
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function start(): Promise<void> {
      setIsReady(false);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (preRollRecorderRef.current && preRollRecorderRef.current.state !== "inactive") {
        preRollRecorderRef.current.stop();
      }
      preRollRecorderRef.current = null;
      preRollChunksRef.current = [];

      try {
        const stream = await getCameraStream(facingMode);

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        await applyWidestAvailableZoom(stream);
        streamRef.current = stream;

        if (videoRef.current) {
          await bindStreamToVideo(videoRef.current, stream);
        }

        if (typeof MediaRecorder !== "undefined") {
          const preferredMimeType = preferredRecorderMimeType();
          const recorder = preferredMimeType ? new MediaRecorder(stream, { mimeType: preferredMimeType }) : new MediaRecorder(stream);
          recorder.ondataavailable = (event) => {
            if (!event.data || event.data.size === 0) return;
            preRollChunksRef.current.push({ blob: event.data, timestamp: Date.now() });

            const cutoff = Date.now() - 10_000;
            preRollChunksRef.current = preRollChunksRef.current.filter((chunk) => chunk.timestamp >= cutoff);
          };
          recorder.start(250);
          preRollRecorderRef.current = recorder;
        }

        setIsReady(true);
      } catch {
        setIsReady(false);
      }
    }

    void start();

    return () => {
      cancelled = true;
      if (preRollRecorderRef.current && preRollRecorderRef.current.state !== "inactive") {
        preRollRecorderRef.current.stop();
      }
      preRollRecorderRef.current = null;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [facingMode]);

  const toggleFacing = (): void => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  const capturePreRollClip = (durationSec = 4): Blob | null => {
    const cutoff = Date.now() - durationSec * 1000;
    const recentChunks = preRollChunksRef.current.filter((chunk) => chunk.timestamp >= cutoff).map((chunk) => chunk.blob);
    if (recentChunks.length === 0) {
      return null;
    }

    const mimeType = recentChunks[0].type || "video/webm";
    return new Blob(recentChunks, { type: mimeType });
  };

  return { videoRef, attachVideoRef, isReady, facingMode, toggleFacing, capturePreRollClip };
}
