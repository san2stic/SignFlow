import { useEffect, useMemo, useState, type Ref, type RefObject } from "react";

import { createSign, listSignVideos, listSigns, uploadSignVideo, type Sign as ApiSign } from "../../api/signs";
import { useMediaPipe } from "../../hooks/useMediaPipe";
import { useTraining } from "../../hooks/useTraining";
import { useWebSocket } from "../../hooks/useWebSocket";
import { countVisibleHands } from "../../lib/landmarks";
import { useTrainingStore } from "../../stores/trainingStore";
import { TagInput } from "../common/TagInput";
import { ClipRecorder, type RecordedClip } from "./ClipRecorder";
import { TrainingProgress } from "./TrainingProgress";
import { ValidationTest } from "./ValidationTest";

interface TrainingWizardProps {
  videoRef: RefObject<HTMLVideoElement>;
  cameraRef?: Ref<HTMLVideoElement>;
  initialAssignedSign?: AssignedSignTarget;
}

interface LiveTrainingPayload {
  status: string;
  progress: number;
  estimated_remaining?: string;
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

interface AssignedSignTarget {
  id: string;
  name: string;
  trainingSampleCount?: number;
  videoCount?: number;
}

function normalizeSignName(rawName: string): string {
  const compact = rawName.trim().replace(/\s+/g, "_").toLowerCase();
  if (!compact) {
    return "";
  }
  return compact.startsWith("lsfb_") ? compact : `lsfb_${compact}`;
}

export function TrainingWizard({ videoRef, cameraRef, initialAssignedSign }: TrainingWizardProps): JSX.Element {
  const [step, setStep] = useState(1);
  const [name, setName] = useState("");
  const [category, setCategory] = useState("lsfb-v1");
  const [tags, setTags] = useState<string[]>(["lsfb", "v1"]);
  const [description, setDescription] = useState("");
  const [nameSuggestions, setNameSuggestions] = useState<string[]>([]);
  const [clips, setClips] = useState<RecordedClip[]>([]);
  const [prediction, setPrediction] = useState("NONE");
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [assignedSignTarget, setAssignedSignTarget] = useState<AssignedSignTarget | null>(null);
  const [existingTrainingClipCount, setExistingTrainingClipCount] = useState(0);
  const [isLoadingExistingTrainingClipCount, setIsLoadingExistingTrainingClipCount] = useState(false);

  const activeSessionId = useTrainingStore((state) => state.activeSessionId);
  const pendingClip = useTrainingStore((state) => state.pendingClip);
  const clearPendingClip = useTrainingStore((state) => state.clearPendingClip);
  const setProgress = useTrainingStore((state) => state.setProgress);
  const resetProgress = useTrainingStore((state) => state.resetProgress);
  const progressState = useTrainingStore((state) => state.progress);
  const { startFewShot, stop, deploy } = useTraining();

  const { frame } = useMediaPipe({
    videoRef,
    enabled: step === 2,
    targetFps: 8,
    includeFace: false
  });

  const visibleHands = countVisibleHands(frame);

  useEffect(() => {
    const query = name.trim();
    if (query.length < 2) {
      setNameSuggestions([]);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      void listSigns(query)
        .then((response) => {
          if (cancelled) return;
          setNameSuggestions(
            response.items
              .map((item) => item.name)
              .filter((candidate, index, all) => all.indexOf(candidate) === index)
              .slice(0, 6)
          );
        })
        .catch(() => {
          if (!cancelled) {
            setNameSuggestions([]);
          }
        });
    }, 200);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [name]);

  useEffect(() => {
    if (!initialAssignedSign) return;

    setAssignedSignTarget((current) => current ?? initialAssignedSign);
    setName((current) => (current ? current : normalizeSignName(initialAssignedSign.name)));
  }, [initialAssignedSign]);

  useEffect(() => {
    if (!assignedSignTarget?.id) {
      setExistingTrainingClipCount(0);
      setIsLoadingExistingTrainingClipCount(false);
      return;
    }

    let cancelled = false;
    setIsLoadingExistingTrainingClipCount(true);

    void listSignVideos(assignedSignTarget.id)
      .then((videos) => {
        if (cancelled) return;
        const eligibleVideos = videos.filter((video) => video.landmarks_extracted).length;
        setExistingTrainingClipCount(eligibleVideos);
      })
      .catch(() => {
        if (!cancelled) {
          // Fallback to sign metadata when per-video list is unavailable.
          setExistingTrainingClipCount(
            Math.max(0, assignedSignTarget.videoCount ?? assignedSignTarget.trainingSampleCount ?? 0)
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingExistingTrainingClipCount(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [assignedSignTarget?.id, assignedSignTarget?.trainingSampleCount]);

  useEffect(() => {
    if (!pendingClip) return;

    if (pendingClip.assignedSign) {
      setAssignedSignTarget({
        id: pendingClip.assignedSign.signId,
        name: pendingClip.assignedSign.signName
      });
    }

    if (pendingClip.suggestedName && !name) {
      setName(normalizeSignName(pendingClip.suggestedName));
    }

    setClips((current) => {
      const exists = current.some((clip) => clip.file.name === pendingClip.file.name && clip.source === "pre-roll");
      if (exists) return current;

      return [
        ...current,
        {
          id: crypto.randomUUID(),
          file: pendingClip.file,
          url: URL.createObjectURL(pendingClip.file),
          durationMs: 4000,
          quality: "valid",
          qualityReasons: [],
          source: "pre-roll"
        }
      ];
    });

    setStep(2);
    clearPendingClip();
  }, [clearPendingClip, name, pendingClip]);

  const wsPath = useMemo(
    () => (activeSessionId ? `/training/sessions/${activeSessionId}/live` : "/training/sessions/unknown/live"),
    [activeSessionId]
  );

  useWebSocket<{ noop: true }, LiveTrainingPayload>({
    path: wsPath,
    enabled: Boolean(activeSessionId),
    onMessage: (payload) => {
      if (payload.metrics) {
        setProgress({
          status: payload.status,
          progress: payload.progress,
          metrics: {
            loss: Number(payload.metrics.loss ?? 0),
            accuracy: Number(payload.metrics.accuracy ?? 0),
            val_accuracy: Number(payload.metrics.val_accuracy ?? 0),
            current_epoch: Number(payload.metrics.current_epoch ?? 0)
          },
          deployment_ready: Boolean(payload.deployment_ready),
          deploy_threshold: Number(payload.deploy_threshold ?? 0.85),
          final_val_accuracy:
            payload.final_val_accuracy === null || payload.final_val_accuracy === undefined
              ? null
              : Number(payload.final_val_accuracy),
          recommended_next_action: payload.recommended_next_action ?? "wait"
        });
      }

      if (payload.status === "completed") {
        setPrediction(name || "new_sign");
        setConfidence(Number(payload.final_val_accuracy ?? payload.metrics?.val_accuracy ?? 0));
      }

      if (payload.status === "failed") {
        setError("Training failed. Please add more high-quality clips and retry.");
      }
    }
  });

  const validClips = useMemo(() => clips.filter((clip) => clip.quality === "valid"), [clips]);
  const minRequiredLocalClips = assignedSignTarget ? 0 : 5;
  const totalAvailableTrainingClips = validClips.length + existingTrainingClipCount;
  const canStartTraining =
    !isLoadingExistingTrainingClipCount &&
    !isSubmitting &&
    (assignedSignTarget ? totalAvailableTrainingClips > 0 : validClips.length >= minRequiredLocalClips);

  const onStepOneNext = (): void => {
    const normalized = normalizeSignName(name);
    if (!normalized && !assignedSignTarget) {
      setError("Sign name is required.");
      return;
    }

    setError(null);
    if (normalized) {
      setName(normalized);
    } else if (assignedSignTarget) {
      setName(normalizeSignName(assignedSignTarget.name));
    }
    setStep(2);
  };

  const onStartTraining = async (): Promise<void> => {
    setError(null);

    const trimmedName = normalizeSignName(name);
    if (!trimmedName && !assignedSignTarget) {
      setError("Sign name is required.");
      return;
    }
    if (!assignedSignTarget && validClips.length < minRequiredLocalClips) {
      setError("At least 5 valid clips are required.");
      return;
    }
    if (assignedSignTarget && totalAvailableTrainingClips < 1) {
      setError("No existing videos found for this sign. Record at least one clip.");
      return;
    }

    setIsSubmitting(true);
    try {
      if (trimmedName) {
        setName(trimmedName);
      }

      let sign: Pick<ApiSign, "id" | "name">;
      if (assignedSignTarget) {
        sign = assignedSignTarget;
      } else {
        try {
          sign = await createSign({
            name: trimmedName,
            category: category || "lsfb-v1",
            tags,
            description: description || `LSFB custom sign: ${trimmedName}`,
            variants: [],
            related_signs: [],
            notes: ""
          });
        } catch {
          const existing = await listSigns(trimmedName);
          const exactMatch = existing.items.find((item) => item.name.toLowerCase() === trimmedName.toLowerCase());
          if (!exactMatch) {
            throw new Error("Unable to create or resolve sign entry.");
          }
          sign = exactMatch;
        }
      }

      for (const clip of validClips) {
        await uploadSignVideo(sign.id, clip.file, {
          type: "training",
          durationMs: clip.durationMs,
          fps: 30,
          resolution: "640x480"
        });
      }

      resetProgress();
      await startFewShot(sign.id, 0.85);
      setStep(3);
    } catch (trainingError) {
      setError(trainingError instanceof Error ? trainingError.message : "Failed to start training.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const trainingCompleted = progressState.status === "completed";
  const recommendation = progressState.recommended_next_action ?? "wait";

  const onDeployModel = async (): Promise<void> => {
    if (!activeSessionId) {
      setDeployError("Missing active training session.");
      throw new Error("Missing active training session.");
    }

    setDeployError(null);
    setIsDeploying(true);
    try {
      await deploy(activeSessionId);
    } catch (deployErr) {
      const message = deployErr instanceof Error ? deployErr.message : "Deployment failed.";
      setDeployError(message);
      throw new Error(message);
    } finally {
      setIsDeploying(false);
    }
  };

  return (
    <section className="space-y-4">
      <header>
        <h1 className="font-heading text-2xl">New Sign Training</h1>
        <p className="text-sm text-slate-400">Step {step}/4</p>
      </header>

      {error && <p className="rounded-btn bg-red-600/20 px-3 py-2 text-sm text-red-200">{error}</p>}

      {step === 1 && (
        <div className="card space-y-3 p-4">
          {assignedSignTarget && (
            <div className="rounded-btn border border-emerald-500/40 bg-emerald-500/10 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-emerald-300">Assigned existing sign</p>
              <p className="font-medium text-emerald-100">{assignedSignTarget.name}</p>
              <button
                className="mt-2 text-xs text-emerald-200 underline"
                onClick={() => setAssignedSignTarget(null)}
              >
                Remove assignment
              </button>
            </div>
          )}
          <label className="flex flex-col gap-1 text-sm">
            Sign Name
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={name}
              onChange={(event) => setName(event.target.value)}
              onBlur={() => setName((current) => normalizeSignName(current))}
              list="existing-signs"
              placeholder="Bonjour"
            />
          </label>
          <datalist id="existing-signs">
            {nameSuggestions.map((suggestion) => (
              <option key={suggestion} value={suggestion} />
            ))}
          </datalist>
          <p className="text-xs text-slate-400">
            {assignedSignTarget
              ? "Sign will be trained on the assigned existing dictionary entry."
              : "Prefix `lsfb_` is added automatically."}
          </p>
          <label className="flex flex-col gap-1 text-sm">
            Category
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
              placeholder="lsfb-v1"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Description
            <textarea
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Describe the sign meaning and movement."
            />
          </label>
          <div>
            <p className="mb-1 text-sm">Tags</p>
            <TagInput tags={tags} onChange={setTags} />
          </div>
          <button className="touch-btn bg-primary text-white" onClick={onStepOneNext}>
            Next
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="card space-y-4 p-4">
          {assignedSignTarget && (
            <div className="rounded-btn border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-100">
              <p>Fine-tuning target: {assignedSignTarget.name}</p>
              <p className="mt-1 text-emerald-200/90">
                {isLoadingExistingTrainingClipCount
                  ? "Loading existing sign videos..."
                  : `Training will use all existing videos (${existingTrainingClipCount}) plus any new valid clips.`}
              </p>
            </div>
          )}
          <ClipRecorder
            videoRef={videoRef}
            cameraRef={cameraRef ?? videoRef}
            clips={clips}
            setClips={setClips}
            visibleHands={visibleHands}
            frame={frame}
          />
          <div className="flex gap-2">
            <button className="touch-btn bg-slate-700 text-white" onClick={() => setStep(1)}>
              Back
            </button>
            <button
              className="touch-btn bg-primary text-white"
              disabled={!canStartTraining}
              onClick={() => {
                void onStartTraining();
              }}
            >
              {isSubmitting ? "Uploading..." : "Start Training"}
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="space-y-3">
          <TrainingProgress />
          <div className="flex gap-2">
            <button
              className="touch-btn bg-accent text-slate-950"
              onClick={() => {
                if (activeSessionId) {
                  void stop(activeSessionId);
                }
              }}
            >
              Stop
            </button>
            <button
              className="touch-btn bg-primary text-white disabled:bg-slate-700 disabled:text-slate-400"
              disabled={!trainingCompleted}
              onClick={() => {
                if (trainingCompleted) {
                  setStep(4);
                }
              }}
            >
              Validate
            </button>
          </div>
          {!trainingCompleted && (
            <p className="text-xs text-slate-400">Validation unlocks automatically when training is completed.</p>
          )}
        </div>
      )}

      {step === 4 && (
        <ValidationTest
          prediction={prediction}
          confidence={confidence}
          deploymentReady={Boolean(progressState.deployment_ready)}
          deployThreshold={progressState.deploy_threshold ?? 0.85}
          recommendedAction={recommendation}
          onDeploy={onDeployModel}
          onCollectMore={() => setStep(2)}
          isDeploying={isDeploying}
          deployError={deployError}
          videoRef={videoRef}
          cameraRef={cameraRef ?? videoRef}
        />
      )}
    </section>
  );
}
