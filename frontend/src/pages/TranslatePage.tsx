import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
import type { CameraError } from "../hooks/useCamera";

interface StreamPayload {
  prediction: string;
  confidence: number;
  alternatives: Array<{ sign: string; confidence: number }>;
  sentence_buffer: string;
  is_sentence_complete: boolean;
}

type UnknownPromptMode = "decision" | "assign";

const UNKNOWN_PROMPT_GRACE_MS = 700;
const UNKNOWN_PROMPT_COOLDOWN_MS = 3000;
const UNKNOWN_PROMPT_RECOVERY_CONFIDENCE = 0.7;

function cameraHelpText(error: CameraError | null): string {
  if (!error) {
    return "Initialisation de la camera...";
  }

  if (error.code === "insecure-context") {
    return "Acces refuse: en distant la camera fonctionne uniquement via HTTPS.";
  }

  return error.message;
}

export function TranslatePage(): JSX.Element {
  const navigate = useNavigate();
  const { videoRef, attachVideoRef, isReady: cameraReady, error: cameraError, toggleFacing, capturePreRollClip } = useCamera();
  const { frame, ready } = useMediaPipe({
    videoRef,
    enabled: true, // ✅ FIX: Toujours activer MediaPipe (ne pas dépendre de cameraReady)
    targetFps: 30,
    includeFace: false,
    modelComplexity: 2
  });

  const speakOnDetect = useSettingsStore((state) => state.speakOnDetect);
  const spellingMode = useSettingsStore((state) => state.spellingMode);
  const unknownThreshold = useSettingsStore((state) => state.unknownThreshold);
  const unknownFrameWindow = useSettingsStore((state) => state.unknownFrameWindow);

  const live = useTranslateStore((state) => state.live);
  const displayedPredictionRaw = useTranslateStore((state) => state.displayedPrediction);
  const displayedConfidence = useTranslateStore((state) => state.displayedConfidence);
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
  const unknownPromptTimer = useRef<number | null>(null);
  const cooldownResetTimer = useRef<number | null>(null);
  const promptCooldownRef = useRef(false);
  const showUnknownPromptRef = useRef(false);

  const clearUnknownPromptTimer = (): void => {
    if (unknownPromptTimer.current !== null) {
      window.clearTimeout(unknownPromptTimer.current);
      unknownPromptTimer.current = null;
    }
  };

  const startPromptCooldown = (): void => {
    if (cooldownResetTimer.current !== null) {
      window.clearTimeout(cooldownResetTimer.current);
      cooldownResetTimer.current = null;
    }

    promptCooldownRef.current = true;
    setPromptCooldown(true);
    cooldownResetTimer.current = window.setTimeout(() => {
      promptCooldownRef.current = false;
      setPromptCooldown(false);
      cooldownResetTimer.current = null;
    }, UNKNOWN_PROMPT_COOLDOWN_MS);
  };

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

      const isConfidentPrediction =
        payload.prediction !== "NONE" &&
        payload.prediction !== "RECORDING" &&
        payload.confidence >= UNKNOWN_PROMPT_RECOVERY_CONFIDENCE;
      if (isConfidentPrediction) {
        lowConfidenceFrames.current = 0;
        clearUnknownPromptTimer();
        if (showUnknownPromptRef.current) {
          showUnknownPromptRef.current = false;
          setShowUnknownPrompt(false);
          resetUnknownPromptState();
        }
        return;
      }

      const isWarmup =
        payload.prediction === "NONE" && payload.confidence === 0 && payload.alternatives.length === 0;
      const isLowConfidence =
        !isWarmup &&
        payload.prediction !== "RECORDING" &&
        (payload.prediction === "NONE" || payload.confidence < unknownThreshold);
      if (!isLowConfidence) {
        lowConfidenceFrames.current = 0;
        clearUnknownPromptTimer();
        return;
      }

      lowConfidenceFrames.current += 1;
      if (promptCooldownRef.current || showUnknownPromptRef.current) {
        return;
      }

      if (lowConfidenceFrames.current >= unknownFrameWindow && unknownPromptTimer.current === null) {
        unknownPromptTimer.current = window.setTimeout(() => {
          unknownPromptTimer.current = null;

          if (promptCooldownRef.current || showUnknownPromptRef.current) {
            return;
          }
          if (lowConfidenceFrames.current < unknownFrameWindow) {
            return;
          }

          showUnknownPromptRef.current = true;
          setShowUnknownPrompt(true);
          startPromptCooldown();
          lowConfidenceFrames.current = 0;
        }, UNKNOWN_PROMPT_GRACE_MS);
      }
    }
  });

  useEffect(() => {
    if (!frame || !ws.connected) return;

    // ✅ FIX: Vérifier que la frame contient des landmarks valides avant l'envoi
    const hasValidLandmarks =
      (frame.hands.left.length > 0 && frame.hands.left.some(point => point[0] !== 0 || point[1] !== 0 || point[2] !== 0)) ||
      (frame.hands.right.length > 0 && frame.hands.right.some(point => point[0] !== 0 || point[1] !== 0 || point[2] !== 0)) ||
      (frame.pose.length > 0 && frame.pose.some(point => point[0] !== 0 || point[1] !== 0 || point[2] !== 0));

    if (!hasValidLandmarks) {
      // Console log pour débogage (peut être retiré en production)
      console.debug('[TranslatePage] Frame sans landmarks valides ignorée');
      return;
    }

    ws.send(serializeLandmarkFrame(frame));
  }, [frame, ws.connected, ws.send]);

  useEffect(() => {
    promptCooldownRef.current = promptCooldown;
  }, [promptCooldown]);

  useEffect(() => {
    showUnknownPromptRef.current = showUnknownPrompt;
  }, [showUnknownPrompt]);

  useEffect(() => {
    return () => {
      clearUnknownPromptTimer();
      if (cooldownResetTimer.current !== null) {
        window.clearTimeout(cooldownResetTimer.current);
      }
    };
  }, []);

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
    clearUnknownPromptTimer();
    lowConfidenceFrames.current = 0;
    showUnknownPromptRef.current = false;
    setShowUnknownPrompt(false);
    resetUnknownPromptState();
    startPromptCooldown();
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
          (displayedPredictionRaw !== "NONE"
            ? displayedPredictionRaw
            : live.prediction !== "NONE" && live.prediction !== "RECORDING"
              ? live.prediction
              : live.alternatives[0]?.sign ?? ""),
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
    showUnknownPromptRef.current = false;
    clearUnknownPromptTimer();
    resetUnknownPromptState();
    navigate("/training");
  };

  const addUnknownToTraining = (): void => {
    handoffToTraining();
  };

  const startAssignToExisting = (): void => {
    setUnknownPromptMode("assign");
    setAssignQuery(
      displayedPredictionRaw !== "NONE"
        ? displayedPredictionRaw
        : live.prediction !== "NONE" && live.prediction !== "RECORDING"
          ? live.prediction
          : live.alternatives[0]?.sign ?? ""
    );
  };

  const assignToExistingSign = (sign: Pick<Sign, "id" | "name">): void => {
    handoffToTraining({ assignedSign: sign });
  };

  const displayedPrediction =
    spellingMode && displayedPredictionRaw !== "NONE"
      ? displayedPredictionRaw
          .replace(/^lsfb_/i, "")
          .split("")
          .map((char) => char.toUpperCase())
          .join(" ")
      : displayedPredictionRaw;

  return (
    <section className="relative space-y-6 pb-8">
      {/* Animated background orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden opacity-30">
        <motion.div
          className="absolute -top-32 -left-32 h-64 w-64 rounded-full bg-primary blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute top-1/2 -right-32 h-64 w-64 rounded-full bg-secondary blur-3xl"
          animate={{
            x: [0, -80, 0],
            y: [0, -60, 0],
            scale: [1, 1.3, 1]
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      {/* Header with biomechanical styling */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
            <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
              <svg className="h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
          </div>
          <h1 className="font-display text-3xl font-bold tracking-tight">
            <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              SignFlow Live
            </span>
          </h1>
        </div>
        <button
          onClick={toggleFacing}
          className="touch-btn group relative overflow-hidden bg-gradient-to-br from-primary/20 to-secondary/20 text-primary backdrop-blur-sm"
        >
          <span className="relative z-10 flex items-center gap-2">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Retourner
          </span>
        </button>
      </motion.header>

      {/* Camera viewport with enhanced styling */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="relative overflow-hidden rounded-card neon-border"
      >
        <div className="relative h-[60vh] min-h-[500px] overflow-hidden rounded-card bg-black/40 backdrop-blur-sm">
          <CameraFeed ref={attachVideoRef} />
          <LandmarkOverlay frame={frame} showConfidenceIndicator={true} />

          {/* Cyber grid overlay */}
          <div className="pointer-events-none absolute inset-0 cyber-grid opacity-20" />

          {/* Status indicators */}
          <div className="absolute top-4 left-4 flex gap-2">
            {ready ? (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="flex items-center gap-2 rounded-full bg-accent/20 px-3 py-1.5 backdrop-blur-sm"
              >
                <div className="h-2 w-2 animate-pulse rounded-full bg-accent shadow-glow" />
                <span className="text-xs font-medium text-accent">LIVE</span>
              </motion.div>
            ) : (
              <div className="shimmer flex items-center gap-2 rounded-full px-3 py-1.5 backdrop-blur-sm">
                <div className={`h-2 w-2 animate-pulse rounded-full ${cameraError ? "bg-red-400" : "bg-text-tertiary"}`} />
                <span className={`text-xs font-medium ${cameraError ? "text-red-200" : "text-text-tertiary"}`}>
                  {cameraError ? "Camera indisponible" : "Initialisation..."}
                </span>
              </div>
            )}

            {ws.connected && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="flex items-center gap-2 rounded-full bg-primary/20 px-3 py-1.5 backdrop-blur-sm"
              >
                <div className="h-2 w-2 rounded-full bg-primary shadow-glow" />
                <span className="text-xs font-medium text-primary">IA Connected</span>
              </motion.div>
            )}
          </div>

          {/* FPS counter */}
          {ready && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute top-4 right-4 rounded-full bg-surface/60 px-3 py-1.5 font-mono text-xs text-text-tertiary backdrop-blur-sm"
            >
              30 FPS
            </motion.div>
          )}

          {/* Not ready overlay */}
          <AnimatePresence>
            {!ready && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm"
              >
                <div className="text-center">
                  <div className="shimmer mx-auto mb-4 h-16 w-16 rounded-full" />
                  <p className="font-display text-lg text-text-secondary">
                    {cameraError ? "Camera indisponible" : cameraReady ? "Preparation de MediaPipe..." : "Demarrage camera..."}
                  </p>
                  <p className={`mt-2 max-w-md text-sm ${cameraError ? "text-red-200" : "text-text-muted"}`}>
                    {cameraHelpText(cameraError)}
                  </p>
                  {cameraError?.code === "insecure-context" && (
                    <p className="mt-2 text-xs text-text-muted">
                      Ouvre SignFlow en HTTPS (ex: via Caddy ou un tunnel SSL) puis recharge la page.
                    </p>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Prediction display with fluid morphing background */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card relative overflow-hidden p-6"
      >
        {/* Morphing background blob */}
        <div className="morph-shape pointer-events-none absolute -top-20 -right-20 h-64 w-64 bg-gradient-to-br from-primary/20 to-secondary/20 blur-3xl" />

        <div className="relative z-10 flex items-center justify-between gap-6">
          <div className="flex-1">
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-tertiary">Signe détecté</p>
            <motion.p
              key={displayedPrediction}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="font-display text-4xl font-bold tracking-tight"
            >
              <span className={displayedPrediction !== "NONE" ? "glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent" : "text-text-muted"}>
                {displayedPrediction}
              </span>
            </motion.p>
          </div>
          <div className="w-48">
            <ConfidenceBadge confidence={displayedConfidence} />
          </div>
        </div>
      </motion.div>

      {/* Sentence buffer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
        className="card p-6"
      >
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-tertiary">Phrase en cours</p>
        <motion.p
          key={live.sentenceBuffer}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="min-h-[2rem] font-body text-xl leading-relaxed text-text-secondary"
        >
          {live.sentenceBuffer || <span className="text-text-muted">En attente de signes...</span>}
        </motion.p>
      </motion.div>

      {/* Action buttons with enhanced styling */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-2 gap-3 md:grid-cols-4"
      >
        <button
          className="touch-btn group relative overflow-hidden bg-gradient-to-br from-accent to-accent-dark text-white"
          onClick={() => speak(live.sentenceBuffer)}
        >
          <span className="relative z-10 flex items-center justify-center gap-2">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
            </svg>
            Lire
          </span>
        </button>

        <button
          className="touch-btn relative overflow-hidden bg-gradient-to-br from-primary/30 to-secondary/30 text-primary backdrop-blur-sm"
          onClick={async () => navigator.clipboard.writeText(live.sentenceBuffer)}
        >
          <span className="relative z-10 flex items-center justify-center gap-2">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Copier
          </span>
        </button>

        <button
          className="touch-btn relative overflow-hidden bg-gradient-to-br from-red-500/80 to-red-600 text-white"
          onClick={reset}
        >
          <span className="relative z-10 flex items-center justify-center gap-2">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Reset
          </span>
        </button>

        <button
          className="touch-btn relative overflow-hidden bg-gradient-to-br from-primary to-secondary text-white"
          onClick={() => navigate("/training")}
        >
          <span className="relative z-10 flex items-center justify-center gap-2">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Ajouter
          </span>
        </button>
      </motion.div>

      {/* Recent predictions with scrolling animation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
        className="card p-6"
      >
        <h2 className="mb-4 font-display text-lg font-semibold">Prédictions récentes</h2>
        <div className="max-h-32 overflow-auto rounded-btn bg-surface-secondary/50 p-3">
          {history.length === 0 ? (
            <p className="text-sm text-text-muted">Aucune prédiction pour le moment.</p>
          ) : (
            <p className="font-mono text-sm text-text-secondary">{history.join(" · ")}</p>
          )}
        </div>
      </motion.div>

      {/* Status indicators */}
      <AnimatePresence>
        {(promptCooldown || handoffStatus) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="card p-4"
          >
            {handoffStatus && (
              <p className="mb-2 text-sm text-accent">{handoffStatus}</p>
            )}
            {promptCooldown && (
              <p className="text-sm text-text-tertiary">
                Cooldown prompt signe inconnu : <span className="font-mono text-primary">{cooldownSeconds}s</span>
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Unknown sign prompt modal */}
      {showUnknownPrompt && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 z-[60] flex items-end justify-center bg-black/60 p-4 backdrop-blur-sm sm:items-center"
        >
          <motion.div
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            className="card neon-border w-full max-w-md p-6"
          >
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-secondary/20">
                <svg className="h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-display text-xl font-semibold">Signe inconnu détecté</h3>
            </div>

            {unknownPromptMode === "decision" ? (
              <>
                <p className="mb-6 text-sm text-text-secondary">
                  La confiance est restée faible. Voulez-vous créer un nouveau signe ou l&apos;assigner à un signe existant avant
                  entraînement ?
                </p>
                <div className="grid gap-3">
                  <button
                    className="touch-btn bg-gradient-to-br from-primary to-secondary text-white"
                    onClick={addUnknownToTraining}
                  >
                    Ajouter un nouveau signe
                  </button>
                  <button
                    className="touch-btn bg-gradient-to-br from-accent/30 to-accent-dark/30 text-accent backdrop-blur-sm"
                    onClick={startAssignToExisting}
                  >
                    Assigner à un signe existant
                  </button>
                  <button
                    className="touch-btn bg-surface-secondary text-text-secondary"
                    onClick={dismissUnknownPrompt}
                  >
                    Ignorer
                  </button>
                </div>
              </>
            ) : (
              <div className="space-y-4">
                <label className="flex flex-col gap-2 text-sm">
                  <span className="font-medium text-text-secondary">Choisir un signe existant</span>
                  <input
                    className="rounded-btn border border-primary/30 bg-surface-secondary/80 px-4 py-3 text-base backdrop-blur-sm transition-all focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
                    value={assignQuery}
                    onChange={(event) => setAssignQuery(event.target.value)}
                    placeholder="Rechercher un signe..."
                    autoFocus
                  />
                </label>

                {isLoadingCandidates && (
                  <div className="shimmer h-8 rounded-btn" />
                )}
                {assignError && (
                  <p className="text-xs text-red-400">{assignError}</p>
                )}

                {!isLoadingCandidates && assignCandidates.length === 0 && assignQuery.trim() && (
                  <p className="text-xs text-text-tertiary">Aucun signe trouvé. Vous pouvez ajouter un nouveau signe.</p>
                )}

                <div className="max-h-64 space-y-2 overflow-auto">
                  {assignCandidates.map((candidate) => (
                    <motion.button
                      key={candidate.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="card w-full p-4 text-left transition-all hover:scale-[1.02]"
                      onClick={() => assignToExistingSign(candidate)}
                    >
                      <p className="font-medium text-text">{candidate.name}</p>
                      <p className="text-xs text-text-tertiary">{candidate.category ?? "Sans catégorie"}</p>
                    </motion.button>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <button
                    className="touch-btn bg-surface-secondary text-text-secondary"
                    onClick={() => setUnknownPromptMode("decision")}
                  >
                    Retour
                  </button>
                  <button
                    className="touch-btn bg-gradient-to-br from-primary to-secondary text-white"
                    onClick={addUnknownToTraining}
                  >
                    Nouveau signe
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </section>
  );
}
