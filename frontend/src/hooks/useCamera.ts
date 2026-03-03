import { useCallback, useEffect, useRef, useState } from "react";

interface BufferedChunk {
  blob: Blob;
  timestamp: number;
}

interface UseCameraOptions {
  /** Largeur du canvas pré-alloué pour l'inférence (défaut : 640) */
  inferenceWidth?: number;
  /** Hauteur du canvas pré-alloué pour l'inférence (défaut : 480) */
  inferenceHeight?: number;
}

interface ZoomCapability {
  min: number;
  max: number;
  step?: number;
}

export type CameraErrorCode =
  | "insecure-context"
  | "unsupported"
  | "permission-denied"
  | "device-not-found"
  | "device-busy"
  | "unknown";

export interface CameraError {
  code: CameraErrorCode;
  message: string;
}

function toCameraError(error: unknown): CameraError {
  const fallback: CameraError = {
    code: "unknown",
    message: "Impossible de demarrer la camera. Verifiez vos permissions navigateur."
  };

  if (typeof DOMException === "undefined" || !(error instanceof DOMException)) {
    return fallback;
  }

  if (error.name === "NotAllowedError" || error.name === "SecurityError") {
    return {
      code: "permission-denied",
      message: "Acces camera refuse. Autorisez la camera dans le navigateur et rechargez la page."
    };
  }

  if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
    return {
      code: "device-not-found",
      message: "Aucune camera compatible detectee sur cet appareil."
    };
  }

  if (error.name === "NotReadableError" || error.name === "TrackStartError") {
    return {
      code: "device-busy",
      message: "Camera indisponible. Fermez les autres apps qui utilisent deja la camera."
    };
  }

  return fallback;
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

export function useCamera({ inferenceWidth = 640, inferenceHeight = 480 }: UseCameraOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<CameraError | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const preRollRecorderRef = useRef<MediaRecorder | null>(null);
  const preRollChunksRef = useRef<BufferedChunk[]>([]);

  /**
   * Canvas pré-alloué pour l'inférence (640×480 par défaut).
   * Évite les allocations GC dans la boucle de traitement des frames.
   */
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  /** Initialise (ou réutilise) le canvas d'inférence pré-alloué */
  const getInferenceCanvas = useCallback((): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null => {
    if (!inferenceCanvasRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = inferenceWidth;
      canvas.height = inferenceHeight;
      inferenceCanvasRef.current = canvas;
      inferenceCtxRef.current = canvas.getContext("2d");
    }
    if (!inferenceCtxRef.current) return null;
    return { canvas: inferenceCanvasRef.current, ctx: inferenceCtxRef.current };
  }, [inferenceWidth, inferenceHeight]);

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
      setError(null);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (preRollRecorderRef.current && preRollRecorderRef.current.state !== "inactive") {
        preRollRecorderRef.current.stop();
      }
      preRollRecorderRef.current = null;
      preRollChunksRef.current = [];

      if (!window.isSecureContext) {
        setError({
          code: "insecure-context",
          message: "Camera bloquee: utilisez HTTPS (obligatoire hors localhost)."
        });
        return;
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        setError({
          code: "unsupported",
          message: "Camera non supportee sur ce navigateur/appareil."
        });
        return;
      }

      try {
        const stream = await getCameraStream(facingMode);

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        await applyWidestAvailableZoom(stream);
        streamRef.current = stream;

        if (videoRef.current) {
          await bindStreamToVideo(videoRef.current, stream).catch(() => {
            // Keep camera active even if autoplay handshake is delayed.
          });
        }

        if (typeof MediaRecorder !== "undefined") {
          try {
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
          } catch {
            preRollRecorderRef.current = null;
          }
        }

        setIsReady(true);
      } catch (cameraError) {
        setError(toCameraError(cameraError));
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

  /**
   * Dessine la frame courante du flux caméra dans le canvas pré-alloué
   * (inferenceWidth × inferenceHeight, défaut 640×480) et retourne l'ImageData.
   *
   * Usage recommandé : passer cette ImageData au Worker MediaPipe au lieu des
   * 1.2 Mpx du flux HD, ce qui réduit la charge de transfert vers le Worker.
   *
   * @returns ImageData redimensionnée, ou null si la vidéo n'est pas prête.
   */
  const captureInferenceFrame = useCallback((): ImageData | null => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const result = getInferenceCanvas();
    if (!result) return null;

    const { canvas, ctx } = result;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }, [getInferenceCanvas]);

  return { videoRef, attachVideoRef, isReady, error, facingMode, toggleFacing, capturePreRollClip, captureInferenceFrame };
}
