import { useEffect, useRef, useState, type Dispatch, type Ref, type RefObject, type SetStateAction } from "react";

import type { LandmarkFrame } from "../../lib/mediapipe";
import { CameraFeed } from "../camera/CameraFeed";
import { RecordButton } from "../camera/RecordButton";
import { ClipCounter } from "./ClipCounter";
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

function isCenteredFrame(frame: LandmarkFrame | null): boolean {
  if (!frame) return false;
  const points = [...frame.hands.left, ...frame.hands.right].filter((point) => point[0] !== 0 || point[1] !== 0);
  if (points.length === 0) return false;
  const centerX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const centerY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  return centerX >= 0.2 && centerX <= 0.8 && centerY >= 0.15 && centerY <= 0.85;
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
  const handDetectedFrameCountRef = useRef(0);
  const centeredFrameCountRef = useRef(0);
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
    if (visibleHandsRef.current > 0) {
      handDetectedFrameCountRef.current += 1;
    }
    if (isCenteredFrame(frameRef.current)) {
      centeredFrameCountRef.current += 1;
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
    handDetectedFrameCountRef.current = 0;
    centeredFrameCountRef.current = 0;
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
      const framesWithHands = handDetectedFrameCountRef.current;
      const centeredFrames = centeredFrameCountRef.current;
      const handPresenceRatio = sampledFrames > 0 ? framesWithHands / sampledFrames : 0;
      const centeredRatio = sampledFrames > 0 ? centeredFrames / sampledFrames : 0;
      const averageBrightness =
        brightnessSamplesRef.current > 0 ? brightnessTotalRef.current / brightnessSamplesRef.current : 0;
      const hasVisibleHands = handPresenceRatio >= 0.8;
      const isCentered = centeredRatio >= 0.7;
      const hasEnoughLight = averageBrightness >= 45;
      const quality: "valid" | "low" =
        hasSufficientDuration && hasEnoughData && hasVisibleHands && isCentered && hasEnoughLight ? "valid" : "low";
      const qualityReasons: string[] = [];
      if (!hasSufficientDuration) {
        qualityReasons.push("Clip too short (<2s)");
      }
      if (!hasEnoughData) {
        qualityReasons.push("Video quality too low");
      }
      if (!hasVisibleHands) {
        qualityReasons.push("Hands visibility too low (<80% frames)");
      }
      if (!isCentered) {
        qualityReasons.push("Gesture too often outside center frame");
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
      <div className="relative h-[44vh] min-h-[18rem] max-h-[34rem] overflow-hidden rounded-card border border-slate-700 sm:h-[50vh] sm:min-h-[22rem] md:h-[58vh] md:min-h-[26rem] md:max-h-[42rem]">
        <CameraFeed ref={cameraRef ?? videoRef} />
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
            <article key={clip.id} className="rounded-btn border border-slate-700 bg-slate-900/60 p-2">
              <video src={clip.url} controls className="mb-2 h-24 w-full rounded-btn bg-black object-cover" />
              <p className="text-xs text-slate-300">{(clip.durationMs / 1000).toFixed(1)}s</p>
              <p className={`text-xs ${clip.quality === "valid" ? "text-secondary" : "text-accent"}`}>
                {clip.quality === "valid" ? "Quality: OK" : "Quality: low"}
              </p>
              {clip.quality === "low" && clip.qualityReasons.length > 0 && (
                <p className="mt-1 text-[11px] text-slate-400">{clip.qualityReasons.join(" · ")}</p>
              )}
              <button className="mt-2 w-full rounded-btn bg-slate-700 px-2 py-1 text-xs" onClick={() => removeClip(clip.id)}>
                Remove
              </button>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
