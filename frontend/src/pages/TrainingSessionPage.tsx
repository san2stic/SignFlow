import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { deployTrainingSession, getTrainingSession, stopTraining } from "../api/training";
import { useWebSocket } from "../hooks/useWebSocket";
import { useTrainingStore, type TrainingProgress } from "../stores/trainingStore";
import { TrainingProgress as TrainingProgressUI } from "../components/training/TrainingProgress";

interface LiveTrainingPayload {
  status: string;
  progress: number;
  estimated_remaining?: string;
  error_message?: string | null;
  metrics: {
    loss?: number;
    accuracy?: number;
    val_accuracy?: number;
    current_epoch?: number;
  };
  deployment_ready?: boolean;
  deploy_threshold?: number;
  final_val_accuracy?: number | null;
  recommended_next_action?: "deploy" | "collect_more_examples" | "wait" | "review_error";
}

export function TrainingSessionPage(): JSX.Element {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const setSession = useTrainingStore((state) => state.setSession);
  const setProgress = useTrainingStore((state) => state.setProgress);
  const progress = useTrainingStore((state) => state.progress);
  const [error, setError] = useState<string | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [isDeploying, setIsDeploying] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Load initial session data via REST
  useEffect(() => {
    if (!sessionId) return;
    setSession(sessionId);

    let cancelled = false;
    void getTrainingSession(sessionId)
      .then((session) => {
        if (cancelled) return;
        setLoaded(true);
        const p: TrainingProgress = {
          status: session.status,
          progress: session.progress,
          metrics: {
            loss: session.metrics?.loss ?? 0,
            accuracy: session.metrics?.accuracy ?? 0,
            val_accuracy: session.metrics?.val_accuracy ?? 0,
            current_epoch: session.metrics?.current_epoch ?? 0,
          },
          deployment_ready: session.deployment_ready,
          deploy_threshold: session.deploy_threshold,
          final_val_accuracy: session.final_val_accuracy,
          recommended_next_action: session.recommended_next_action,
        };
        setProgress(p);
      })
      .catch(() => {
        if (!cancelled) {
          setError("Training session not found.");
          setLoaded(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId, setSession, setProgress]);

  // Live WebSocket updates
  const wsPath = sessionId
    ? `/training/sessions/${sessionId}/live`
    : "/training/sessions/unknown/live";

  useWebSocket<{ noop: true }, LiveTrainingPayload>({
    path: wsPath,
    enabled: Boolean(sessionId) && loaded && progress.status !== "completed" && progress.status !== "failed",
    onMessage: (payload) => {
      if (payload.metrics) {
        setProgress({
          status: payload.status,
          progress: payload.progress,
          metrics: {
            loss: Number(payload.metrics.loss ?? 0),
            accuracy: Number(payload.metrics.accuracy ?? 0),
            val_accuracy: Number(payload.metrics.val_accuracy ?? 0),
            current_epoch: Number(payload.metrics.current_epoch ?? 0),
          },
          deployment_ready: Boolean(payload.deployment_ready),
          deploy_threshold: Number(payload.deploy_threshold ?? 0.85),
          final_val_accuracy:
            payload.final_val_accuracy === null || payload.final_val_accuracy === undefined
              ? null
              : Number(payload.final_val_accuracy),
          recommended_next_action: payload.recommended_next_action ?? "wait",
        });
      }
    },
  });

  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelConfirm, setCancelConfirm] = useState(false);

  const handleStop = useCallback(async () => {
    if (!sessionId) return;
    setIsCancelling(true);
    setCancelConfirm(false);
    try {
      await stopTraining(sessionId);
    } catch {
      setError("Impossible d'annuler l'entraînement. Réessayez.");
    } finally {
      setIsCancelling(false);
    }
  }, [sessionId]);

  const handleDeploy = useCallback(async () => {
    if (!sessionId) return;
    setDeployError(null);
    setIsDeploying(true);
    try {
      await deployTrainingSession(sessionId);
    } catch (err) {
      setDeployError(err instanceof Error ? err.message : "Deployment failed.");
    } finally {
      setIsDeploying(false);
    }
  }, [sessionId]);

  if (!sessionId) {
    return (
      <section className="space-y-5">
        <div className="card p-5">
          <p className="text-red-300">No session ID provided.</p>
          <button className="touch-btn mt-3 bg-primary text-white" onClick={() => navigate("/training")}>
            Back to Training
          </button>
        </div>
      </section>
    );
  }

  if (!loaded) {
    return (
      <section className="space-y-5">
        <div className="card flex min-h-[360px] items-center justify-center p-5">
          <div className="text-center">
            <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-primary" />
            <p className="text-sm text-slate-300">Loading training session...</p>
          </div>
        </div>
      </section>
    );
  }

  if (error && !loaded) {
    return (
      <section className="space-y-5">
        <div className="card p-5">
          <p className="text-red-300">{error}</p>
          <button className="touch-btn mt-3 bg-primary text-white" onClick={() => navigate("/training")}>
            Back to Training
          </button>
        </div>
      </section>
    );
  }

  const ACTIVE_STATUSES = new Set(["queued", "preprocessing", "training", "validating"]);
  const isRunning = ACTIVE_STATUSES.has(progress.status);
  const isCompleted = progress.status === "completed";
  const isFailed = progress.status === "failed";

  return (
    <section className="space-y-5">
      <header className="card p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Training Session</p>
            <h1 className="mt-2 font-display text-2xl font-semibold text-white">
              Session {sessionId.slice(0, 8)}...
            </h1>
            <p className="mt-1 text-sm text-text-secondary">
              Status:{" "}
              <span
                className={
                  isRunning
                    ? "text-yellow-400"
                    : isCompleted
                      ? "text-green-400"
                      : isFailed
                        ? "text-red-400"
                        : "text-slate-400"
                }
              >
                {progress.status}
              </span>
            </p>
          </div>
          <button
            className="touch-btn bg-slate-700 text-sm text-white hover:bg-slate-600"
            onClick={() => navigate("/training")}
          >
            ← Back
          </button>
        </div>
      </header>

      {error && <p className="rounded-btn bg-red-600/20 px-3 py-2 text-sm text-red-200">{error}</p>}

      <TrainingProgressUI />

      <div className="card space-y-3 p-4">
        {isRunning && !cancelConfirm && (
          <button
            className="touch-btn border border-red-500/60 bg-red-600/20 text-red-300 hover:bg-red-600/30 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isCancelling}
            onClick={() => setCancelConfirm(true)}
          >
            Annuler l'entraînement
          </button>
        )}

        {cancelConfirm && (
          <div className="rounded-btn border border-red-500/40 bg-red-900/20 p-4">
            <p className="mb-1 text-sm font-semibold text-red-200">Confirmer l'annulation</p>
            <p className="mb-3 text-xs text-red-300/80">
              L'entraînement en cours sera interrompu. Cette action est irréversible.
            </p>
            <div className="flex gap-2">
              <button
                className="touch-btn border border-red-500 bg-red-600 text-sm text-white hover:bg-red-700 disabled:opacity-60"
                disabled={isCancelling}
                onClick={() => { void handleStop(); }}
              >
                {isCancelling ? "Annulation…" : "Oui, annuler"}
              </button>
              <button
                className="touch-btn bg-slate-700 text-sm text-white hover:bg-slate-600"
                disabled={isCancelling}
                onClick={() => setCancelConfirm(false)}
              >
                Non, continuer
              </button>
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-3">
        {isCompleted && progress.deployment_ready && (
          <button
            className="touch-btn bg-green-600 text-white hover:bg-green-500 disabled:opacity-50"
            disabled={isDeploying}
            onClick={handleDeploy}
          >
            {isDeploying ? "Deploying..." : "Deploy Model"}
          </button>
        )}

        {deployError && <p className="text-sm text-red-300">{deployError}</p>}

        {isFailed && (
          <button
            className="touch-btn bg-primary text-white"
            onClick={() => navigate("/training")}
          >
            Start New Training
          </button>
        )}
        </div>
      </div>

      <div className="card space-y-2 p-4 text-xs text-slate-400">
        <p>
          <strong>Session ID:</strong> <code className="text-slate-300">{sessionId}</code>
        </p>
        <p>
          <strong>Shareable URL:</strong>{" "}
          <code className="text-slate-300">{window.location.href}</code>
        </p>
      </div>
    </section>
  );
}
