import { useEffect, useRef, useState, type Dispatch, type Ref, type RefObject, type SetStateAction } from "react";

import type { LandmarkFrame } from "../../lib/mediapipe";
import { CameraFeed } from "../camera/CameraFeed";
import { LandmarkOverlay } from "../camera/LandmarkOverlay";
import { RecordButton } from "../camera/RecordButton";
import { ClipCounter } from "./ClipCounter";
import { ClipPreview } from "./ClipPreview";
import { QualityIndicator } from "./QualityIndicator";
import { SignGuideOverlay } from "./SignGuideOverlay";

export interface RecordedClip {
  id: string;
  file: File;
  url: string;
  durationMs: number;
  quality: "valid" | "low";
  qualityReasons: string[];
  source: "recorded" | "pre-roll";
}

interface ClipRecorderProps {
  videoRef: RefObject<HTMLVideoElement>;
  cameraRef?: Ref<HTMLVideoElement>;
  clips: RecordedClip[];
  setClips: Dispatch<SetStateAction<RecordedClip[]>>;
  visibleHands?: number;
  frame?: LandmarkFrame | null;
}

const MIN_HAND_PRESENCE_RATIO = 0.35;
const MIN_TRACKED_FRAMES = 8;

function pickRecorderMimeType(): string | null {
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

function hasVisibleHandPoints(points: number[][]): boolean {
  return points.some((point) => point[0] !== 0 || point[1] !== 0 || point[2] !== 0);
}

function countVisibleHandsInFrame(frame: LandmarkFrame | null): number {
  if (!frame) return 0;
  let count = 0;
  if (hasVisibleHandPoints(frame.hands.left)) {
    count += 1;
  }
  if (hasVisibleHandPoints(frame.hands.right)) {
    count += 1;
  }
  return count;
}

export function ClipRecorder({
  videoRef,
  cameraRef,
  clips,
  setClips,
  visibleHands = 0,
  frame = null
}: ClipRecorderProps): JSX.Element {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);
  const autoStopRef = useRef<number | null>(null);
  const qualitySampleRef = useRef<number | null>(null);
  const startedAtRef = useRef<number>(0);
  const sampledFrameCountRef = useRef(0);
  const trackedFrameCountRef = useRef(0);
  const lastTrackedFrameIdxRef = useRef<number | null>(null);
  const handDetectedFrameCountRef = useRef(0);
  const brightnessTotalRef = useRef(0);
  const brightnessSamplesRef = useRef(0);
  const visibleHandsRef = useRef(visibleHands);
  const frameRef = useRef<LandmarkFrame | null>(frame);
  const brightnessCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    visibleHandsRef.current = visibleHands;
  }, [visibleHands]);

  useEffect(() => {
    frameRef.current = frame;
  }, [frame]);

  const measureBrightness = (): number | null => {
    const video = videoRef.current;
    if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return null;
    }

    const canvas = brightnessCanvasRef.current ?? document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 48;
    brightnessCanvasRef.current = canvas;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let luminance = 0;
    for (let idx = 0; idx < data.length; idx += 4) {
      luminance += 0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2];
    }
    return luminance / (data.length / 4);
  };

  const sampleQualityFrame = (): void => {
    sampledFrameCountRef.current += 1;
    const currentFrame = frameRef.current;
    if (currentFrame && currentFrame.frame_idx !== lastTrackedFrameIdxRef.current) {
      lastTrackedFrameIdxRef.current = currentFrame.frame_idx;
      trackedFrameCountRef.current += 1;
      if (countVisibleHandsInFrame(currentFrame) > 0 || visibleHandsRef.current > 0) {
        handDetectedFrameCountRef.current += 1;
      }
    }

    const brightness = measureBrightness();
    if (brightness !== null) {
      brightnessTotalRef.current += brightness;
      brightnessSamplesRef.current += 1;
    }
  };

  const cleanupTimers = (): void => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (autoStopRef.current) {
      window.clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
    if (qualitySampleRef.current) {
      window.clearInterval(qualitySampleRef.current);
      qualitySampleRef.current = null;
    }
  };

  const handleStopRecording = (): void => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      cleanupTimers();
      setIsRecording(false);
      setSeconds(0);
      return;
    }

    recorder.stop();
  };

  const handleStartRecording = (): void => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    if (!stream || typeof MediaRecorder === "undefined") {
      return;
    }

    chunksRef.current = [];
    startedAtRef.current = Date.now();
    sampledFrameCountRef.current = 0;
    trackedFrameCountRef.current = 0;
    lastTrackedFrameIdxRef.current = null;
    handDetectedFrameCountRef.current = 0;
    brightnessTotalRef.current = 0;
    brightnessSamplesRef.current = 0;
    sampleQualityFrame();

    const preferredMimeType = pickRecorderMimeType();
    const recorder = preferredMimeType ? new MediaRecorder(stream, { mimeType: preferredMimeType }) : new MediaRecorder(stream);

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      cleanupTimers();

      const durationMs = Math.max(0, Date.now() - startedAtRef.current);
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
      const extension = blob.type.includes("mp4") ? "mp4" : "webm";
      const file = new File([blob], `clip-${Date.now()}.${extension}`, { type: blob.type || "video/webm" });
      const url = URL.createObjectURL(file);

      const hasSufficientDuration = durationMs >= 2000;
      const hasEnoughData = file.size >= 20_000;
      const sampledFrames = sampledFrameCountRef.current;
      const trackedFrames = trackedFrameCountRef.current;
      const framesWithHands = handDetectedFrameCountRef.current;
      const handPresenceRatio = trackedFrames > 0 ? framesWithHands / trackedFrames : 0;
      const averageBrightness =
        brightnessSamplesRef.current > 0 ? brightnessTotalRef.current / brightnessSamplesRef.current : 0;
      const hasEnoughTrackingData = trackedFrames >= MIN_TRACKED_FRAMES || sampledFrames <= MIN_TRACKED_FRAMES;
      const hasVisibleHands = handPresenceRatio >= MIN_HAND_PRESENCE_RATIO;
      const hasEnoughLight = averageBrightness >= 45;
      const quality: "valid" | "low" =
        hasSufficientDuration &&
        hasEnoughData &&
        hasEnoughTrackingData &&
        hasVisibleHands &&
        hasEnoughLight
          ? "valid"
          : "low";
      const qualityReasons: string[] = [];
      if (!hasSufficientDuration) {
        qualityReasons.push("Clip too short (<2s)");
      }
      if (!hasEnoughData) {
        qualityReasons.push("Video quality too low");
      }
      if (!hasEnoughTrackingData) {
        qualityReasons.push("Tracking data is unstable, keep hands in frame a bit longer");
      }
      if (!hasVisibleHands) {
        qualityReasons.push(`At least one hand must be visible in >=${Math.round(MIN_HAND_PRESENCE_RATIO * 100)}% of frames`);
      }
      if (!hasEnoughLight) {
        qualityReasons.push("Lighting is too low");
      }

      setClips((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          file,
          url,
          durationMs,
          quality,
          qualityReasons,
          source: "recorded"
        }
      ]);

      mediaRecorderRef.current = null;
      setIsRecording(false);
      setSeconds(0);
    };

    recorder.start();
    mediaRecorderRef.current = recorder;
    setIsRecording(true);

    timerRef.current = window.setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);
    qualitySampleRef.current = window.setInterval(() => {
      sampleQualityFrame();
    }, 120);

    autoStopRef.current = window.setTimeout(() => {
      handleStopRecording();
    }, 3000);
  };

  const onRecordToggle = (): void => {
    if (isRecording) {
      handleStopRecording();
      return;
    }
    handleStartRecording();
  };

  const removeClip = (clipId: string): void => {
    setClips((current) => {
      const target = current.find((clip) => clip.id === clipId);
      if (target) {
        URL.revokeObjectURL(target.url);
      }
      return current.filter((clip) => clip.id !== clipId);
    });
  };

  return (
    <div className="space-y-3">
      <div className="relative mx-auto h-[44vh] min-h-[18rem] w-full max-w-5xl overflow-hidden rounded-card border border-slate-700 sm:h-[50vh] sm:min-h-[22rem] md:h-[58vh] md:min-h-[26rem] md:max-h-[42rem]">
        <CameraFeed ref={cameraRef ?? videoRef} fit="contain" />
        <LandmarkOverlay frame={frame} showConfidenceIndicator={false} videoRef={videoRef} fit="contain" />
        <SignGuideOverlay />
      </div>
      <QualityIndicator visibleHands={visibleHands} />
      <div className="flex items-center justify-between">
        <RecordButton isRecording={isRecording} onClick={onRecordToggle} />
        <span className="font-mono text-sm text-slate-300">{seconds.toFixed(1)}s</span>
      </div>
      <ClipCounter clips={clips} minClips={5} />

      {clips.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {clips.map((clip) => (
            <ClipPreview
              key={clip.id}
              url={clip.url}
              durationMs={clip.durationMs}
              quality={clip.quality}
              qualityReasons={clip.qualityReasons}
              onRemove={() => removeClip(clip.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
