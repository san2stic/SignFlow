import { motion } from "framer-motion";
import { useEffect, useRef, useState, type Ref, type RefObject } from "react";

import { useMediaPipe } from "../../hooks/useMediaPipe";
import { useWebSocket } from "../../hooks/useWebSocket";
import { serializeLandmarkFrame, type LandmarkFrame } from "../../lib/mediapipe";
import { CameraFeed } from "../camera/CameraFeed";
import { LandmarkOverlay } from "../camera/LandmarkOverlay";
import { DeploymentReadinessBadge } from "./DeploymentReadinessBadge";
import { RecommendedActionMessage } from "./RecommendedActionMessage";

interface StreamPayload {
  prediction: string;
  confidence: number;
  alternatives: Array<{ sign: string; confidence: number }>;
  sentence_buffer: string;
  is_sentence_complete: boolean;
}

interface ValidationTestProps {
  prediction: string;
  confidence: number;
  deploymentReady: boolean;
  deployThreshold: number;
  recommendedAction: "deploy" | "collect_more_examples" | "wait" | "review_error";
  onDeploy: () => Promise<void>;
  onCollectMore: () => void;
  isDeploying: boolean;
  deployError: string | null;
  videoRef: RefObject<HTMLVideoElement>;
  cameraRef?: Ref<HTMLVideoElement>;
}

export function ValidationTest({
  prediction,
  confidence,
  deploymentReady,
  deployThreshold,
  recommendedAction,
  onDeploy,
  onCollectMore,
  isDeploying,
  deployError,
  videoRef,
  cameraRef
}: ValidationTestProps): JSX.Element {
  const [deploySuccess, setDeploySuccess] = useState(false);
  const [livePrediction, setLivePrediction] = useState<StreamPayload | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const hasLiveHitRef = useRef(false);

  const { frame } = useMediaPipe({
    videoRef,
    enabled: true,
    targetFps: 10,
    includeFace: false
  });

  const ws = useWebSocket<LandmarkFrame, StreamPayload>({
    path: "/translate/stream",
    onMessage: (payload) => {
      setLivePrediction(payload);
      hasLiveHitRef.current = hasLiveHitRef.current || payload.prediction !== "NONE";
      setWsError(null);
    }
  });

  useEffect(() => {
    if (!frame || !ws.connected) return;
    ws.send(serializeLandmarkFrame(frame));
  }, [frame, ws.connected, ws.send]);

  useEffect(() => {
    if (ws.connected) return;
    setWsError("Live validation stream is disconnected.");
  }, [ws.connected]);

  useEffect(() => {
    if (!deploySuccess) return;

    const timer = window.setTimeout(() => {
      window.location.assign("/translate");
    }, 2000);

    return () => window.clearTimeout(timer);
  }, [deploySuccess]);

  const handleDeploy = async (): Promise<void> => {
    try {
      await onDeploy();
      setDeploySuccess(true);
    } catch {
      setDeploySuccess(false);
    }
  };

  const displayedPrediction = livePrediction?.prediction ?? prediction;
  const displayedConfidence = livePrediction?.confidence ?? confidence;

  return (
    <div className="card space-y-3 p-4">
      <h3 className="font-heading text-lg">Validation Live Test</h3>
      <p className="text-xs text-slate-400">Perform the sign in camera. The live WS translation is used for final check.</p>

      <div className="relative h-[44vh] min-h-[18rem] max-h-[34rem] overflow-hidden rounded-card border border-slate-700 sm:h-[50vh] sm:min-h-[22rem] md:h-[58vh] md:min-h-[26rem] md:max-h-[42rem]">
        <CameraFeed ref={cameraRef ?? videoRef} />
        <LandmarkOverlay frame={frame} />
      </div>

      {!ws.connected && (
        <p className="rounded-btn bg-accent/20 px-3 py-2 text-xs text-accent">
          {wsError ?? "Waiting for live websocket..."}
        </p>
      )}
      {hasLiveHitRef.current && (
        <p className="rounded-btn bg-secondary/15 px-3 py-2 text-xs text-secondary">Live prediction is active.</p>
      )}

      <div className="rounded-btn border border-slate-700 bg-slate-900/50 p-3">
        <p className="text-xs uppercase tracking-wide text-slate-400">Predicted sign</p>
        <p className="mt-1 text-xl font-semibold text-white">{displayedPrediction}</p>
        <p className="mt-1 text-sm text-slate-300">Confidence: {(displayedConfidence * 100).toFixed(1)}%</p>
      </div>
      <DeploymentReadinessBadge ready={deploymentReady} accuracy={displayedConfidence} threshold={deployThreshold} />
      <RecommendedActionMessage action={recommendedAction} />

      {deployError && <p className="rounded-btn bg-red-600/20 px-3 py-2 text-xs text-red-200">{deployError}</p>}

      {!deploySuccess && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {recommendedAction === "deploy" ? (
            <button
              className="touch-btn bg-secondary text-slate-950 disabled:bg-slate-700 disabled:text-slate-400"
              disabled={!deploymentReady || isDeploying}
              onClick={() => {
                void handleDeploy();
              }}
            >
              {isDeploying ? "Deploying..." : "Deploy model"}
            </button>
          ) : (
            <button className="touch-btn bg-accent text-slate-950" onClick={onCollectMore}>
              Record more clips
            </button>
          )}
          <button className="touch-btn bg-slate-700 text-white" onClick={onCollectMore}>
            Back to recording
          </button>
        </div>
      )}

      {deploySuccess && (
        <motion.div
          className="rounded-btn border border-secondary/40 bg-secondary/15 p-5 text-center"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.25 }}
        >
          <motion.p className="text-2xl font-bold text-secondary" animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 0.5 }}>
            Deployment complete
          </motion.p>
          <p className="mt-1 text-sm text-slate-300">Redirecting to live translation...</p>
        </motion.div>
      )}
    </div>
  );
}
