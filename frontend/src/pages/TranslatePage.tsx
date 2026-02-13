import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

import { listSigns, type Sign } from "../api/signs";
import { CameraFeed } from "../components/camera/CameraFeed";
import { LandmarkOverlay } from "../components/camera/LandmarkOverlay";
import { ConfidenceBadge } from "../components/common/ConfidenceBadge";
import { useCamera } from "../hooks/useCamera";
import { useMediaPipe } from "../hooks/useMediaPipe";
import { useWebSocket } from "../hooks/useWebSocket";
import { serializeLandmarkFrame, type LandmarkFrame } from "../lib/mediapipe";
import { speak } from "../lib/speech";
import { useSettingsStore } from "../stores/settingsStore";
import { useTrainingStore } from "../stores/trainingStore";
import { useTranslateStore } from "../stores/translateStore";

interface StreamPayload {
  prediction: string;
  confidence: number;
  alternatives: Array<{ sign: string; confidence: number }>;
  sentence_buffer: string;
  is_sentence_complete: boolean;
}

type UnknownPromptMode = "decision" | "assign";

export function TranslatePage(): JSX.Element {
  const navigate = useNavigate();
  const { videoRef, attachVideoRef, toggleFacing, capturePreRollClip } = useCamera();
  const { frame } = useMediaPipe({ videoRef, enabled: true, targetFps: 12, includeFace: false });

  const speakOnDetect = useSettingsStore((state) => state.speakOnDetect);
  const spellingMode = useSettingsStore((state) => state.spellingMode);
  const unknownThreshold = useSettingsStore((state) => state.unknownThreshold);
  const unknownFrameWindow = useSettingsStore((state) => state.unknownFrameWindow);

  const live = useTranslateStore((state) => state.live);
  const history = useTranslateStore((state) => state.history);
  const setLive = useTranslateStore((state) => state.setLive);
  const reset = useTranslateStore((state) => state.reset);

  const setPendingClip = useTrainingStore((state) => state.setPendingClip);

  const [showUnknownPrompt, setShowUnknownPrompt] = useState(false);
  const [promptCooldown, setPromptCooldown] = useState(false);
  const [cooldownSeconds, setCooldownSeconds] = useState(0);
  const [handoffStatus, setHandoffStatus] = useState<string | null>(null);
  const [unknownPromptMode, setUnknownPromptMode] = useState<UnknownPromptMode>("decision");
  const [assignQuery, setAssignQuery] = useState("");
  const [assignCandidates, setAssignCandidates] = useState<Sign[]>([]);
  const [isLoadingCandidates, setIsLoadingCandidates] = useState(false);
  const [assignError, setAssignError] = useState<string | null>(null);
  const lowConfidenceFrames = useRef(0);

  const ws = useWebSocket<LandmarkFrame, StreamPayload>({
    path: "/translate/stream",
    onMessage: (payload) => {
      setLive({
        prediction: payload.prediction,
        confidence: payload.confidence,
        alternatives: payload.alternatives,
        sentenceBuffer: payload.sentence_buffer
      });

      if (speakOnDetect && payload.prediction !== "NONE" && payload.confidence >= 0.7) {
        speak(payload.prediction);
      }

      const isWarmup =
        payload.prediction === "NONE" && payload.confidence === 0 && payload.alternatives.length === 0;
      const isLowConfidence = !isWarmup && (payload.prediction === "NONE" || payload.confidence < unknownThreshold);
      lowConfidenceFrames.current = isLowConfidence ? lowConfidenceFrames.current + 1 : 0;

      if (!promptCooldown && lowConfidenceFrames.current >= unknownFrameWindow) {
        setShowUnknownPrompt(true);
        setPromptCooldown(true);
        lowConfidenceFrames.current = 0;
      }
    }
  });

  useEffect(() => {
    if (!frame || !ws.connected) return;
    ws.send(serializeLandmarkFrame(frame));
  }, [frame, ws.connected, ws.send]);

  useEffect(() => {
    if (!promptCooldown) {
      setCooldownSeconds(0);
      return;
    }

    setCooldownSeconds(3);
    const intervalId = window.setInterval(() => {
      setCooldownSeconds((current) => {
        if (current <= 1) {
          window.clearInterval(intervalId);
          return 0;
        }
        return current - 1;
      });
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [promptCooldown]);

  useEffect(() => {
    if (!showUnknownPrompt || unknownPromptMode !== "assign") {
      setAssignCandidates([]);
      setIsLoadingCandidates(false);
      setAssignError(null);
      return;
    }

    let cancelled = false;
    const timerId = window.setTimeout(() => {
      setIsLoadingCandidates(true);
      setAssignError(null);

      void listSigns(assignQuery.trim())
        .then((response) => {
          if (cancelled) return;
          setAssignCandidates(response.items.slice(0, 8));
        })
        .catch(() => {
          if (cancelled) return;
          setAssignCandidates([]);
          setAssignError("Impossible de charger la liste des signes.");
        })
        .finally(() => {
          if (!cancelled) {
            setIsLoadingCandidates(false);
          }
        });
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [assignQuery, showUnknownPrompt, unknownPromptMode]);

  const resetUnknownPromptState = (): void => {
    setUnknownPromptMode("decision");
    setAssignQuery("");
    setAssignCandidates([]);
    setAssignError(null);
  };

  const dismissUnknownPrompt = (): void => {
    setShowUnknownPrompt(false);
    resetUnknownPromptState();
    window.setTimeout(() => setPromptCooldown(false), 3000);
  };

  const handoffToTraining = (options?: { assignedSign?: Pick<Sign, "id" | "name"> }): void => {
    const preRoll = capturePreRollClip(4);
    if (preRoll && preRoll.size > 0) {
      const extension = preRoll.type.includes("mp4") ? "mp4" : "webm";
      const file = new File([preRoll], `unknown-${Date.now()}.${extension}`, { type: preRoll.type || "video/webm" });
      setPendingClip({
        file,
        suggestedName:
          options?.assignedSign?.name ??
          (live.prediction !== "NONE" ? live.prediction : live.alternatives[0]?.sign ?? ""),
        assignedSign: options?.assignedSign
          ? {
              signId: options.assignedSign.id,
              signName: options.assignedSign.name
            }
          : undefined,
        createdAt: Date.now()
      });
      setHandoffStatus(
        options?.assignedSign
          ? `Clip attaché. Le fine-tuning ciblera le signe existant "${options.assignedSign.name}".`
          : "Pre-roll clip attached to training session."
      );
    } else {
      setHandoffStatus("No pre-roll clip available. Continue with manual recording.");
    }

    setShowUnknownPrompt(false);
    resetUnknownPromptState();
    navigate("/train");
  };

  const addUnknownToTraining = (): void => {
    handoffToTraining();
  };

  const startAssignToExisting = (): void => {
    setUnknownPromptMode("assign");
    setAssignQuery(live.prediction !== "NONE" ? live.prediction : live.alternatives[0]?.sign ?? "");
  };

  const assignToExistingSign = (sign: Pick<Sign, "id" | "name">): void => {
    handoffToTraining({ assignedSign: sign });
  };

  const displayedPrediction =
    spellingMode && live.prediction !== "NONE"
      ? live.prediction
          .replace(/^lsfb_/i, "")
          .split("")
          .map((char) => char.toUpperCase())
          .join(" ")
      : live.prediction;

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between">
        <h1 className="font-heading text-2xl">Translate Live</h1>
        <button onClick={toggleFacing} className="touch-btn bg-slate-700 text-white">
          Switch Camera
        </button>
      </header>

      <div className="relative h-[55vh] min-h-80 overflow-hidden rounded-card border border-slate-700/70">
        <CameraFeed ref={attachVideoRef} />
        <LandmarkOverlay frame={frame} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="card flex items-center justify-between gap-4 p-4"
      >
        <div>
          <p className="text-xs text-slate-400">Current sign</p>
          <p className="font-heading text-3xl">{displayedPrediction}</p>
        </div>
        <div className="w-40">
          <ConfidenceBadge confidence={live.confidence} />
        </div>
      </motion.div>

      <div className="card p-4">
        <p className="text-xs text-slate-400">Sentence buffer</p>
        <p className="mt-1 text-lg">{live.sentenceBuffer || "..."}</p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <button className="touch-btn bg-secondary text-slate-950" onClick={() => speak(live.sentenceBuffer)}>
          Lire
        </button>
        <button
          className="touch-btn bg-slate-700 text-white"
          onClick={async () => navigator.clipboard.writeText(live.sentenceBuffer)}
        >
          Copier
        </button>
        <button className="touch-btn bg-red-500 text-white" onClick={reset}>
          Reset
        </button>
        <button className="touch-btn bg-primary text-white" onClick={() => navigate("/train")}>
          Ajouter
        </button>
      </div>

      <div className="card p-4">
        <h2 className="font-heading text-lg">Recent predictions</h2>
        <div className="mt-2 max-h-32 overflow-auto text-sm text-slate-300">
          {history.length === 0 ? "No predictions yet." : history.join(" · ")}
        </div>
      </div>

      {(promptCooldown || handoffStatus) && (
        <div className="card p-3 text-xs text-slate-300">
          {handoffStatus && <p>{handoffStatus}</p>}
          {promptCooldown && <p>Unknown-sign prompt cooldown: {cooldownSeconds}s</p>}
        </div>
      )}

      {showUnknownPrompt && (
        <div className="fixed inset-0 z-[60] flex items-end justify-center bg-black/50 p-4 sm:items-center">
          <div className="w-full max-w-md rounded-card border border-slate-700 bg-surface p-4">
            <h3 className="font-heading text-xl">Signe inconnu détecté</h3>
            {unknownPromptMode === "decision" ? (
              <>
                <p className="mt-2 text-sm text-slate-300">
                  La confiance est restée faible. Voulez-vous créer un nouveau signe ou l&apos;assigner à un signe existant avant
                  entraînement ?
                </p>
                <div className="mt-4 grid gap-2">
                  <button className="touch-btn bg-primary text-white" onClick={addUnknownToTraining}>
                    Ajouter un nouveau signe
                  </button>
                  <button className="touch-btn bg-secondary text-slate-950" onClick={startAssignToExisting}>
                    Assigner à un signe existant
                  </button>
                  <button className="touch-btn bg-slate-700 text-white" onClick={dismissUnknownPrompt}>
                    Ignorer
                  </button>
                </div>
              </>
            ) : (
              <div className="mt-3 space-y-3">
                <label className="flex flex-col gap-1 text-sm">
                  Choisir un signe existant
                  <input
                    className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
                    value={assignQuery}
                    onChange={(event) => setAssignQuery(event.target.value)}
                    placeholder="Rechercher un signe..."
                  />
                </label>

                {isLoadingCandidates && <p className="text-xs text-slate-400">Chargement…</p>}
                {assignError && <p className="text-xs text-red-300">{assignError}</p>}

                {!isLoadingCandidates && assignCandidates.length === 0 && (
                  <p className="text-xs text-slate-400">Aucun signe trouvé. Vous pouvez ajouter un nouveau signe.</p>
                )}

                <div className="max-h-56 space-y-2 overflow-auto">
                  {assignCandidates.map((candidate) => (
                    <button
                      key={candidate.id}
                      className="w-full rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2 text-left text-sm"
                      onClick={() => assignToExistingSign(candidate)}
                    >
                      <p className="font-medium text-slate-100">{candidate.name}</p>
                      <p className="text-xs text-slate-400">{candidate.category ?? "Uncategorized"}</p>
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button className="touch-btn bg-slate-700 text-white" onClick={() => setUnknownPromptMode("decision")}>
                    Retour
                  </button>
                  <button className="touch-btn bg-primary text-white" onClick={addUnknownToTraining}>
                    Nouveau signe
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
