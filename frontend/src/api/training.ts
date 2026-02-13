import { apiFetch } from "./client";

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  val_accuracy: number;
  current_epoch?: number;
}

export interface TrainingSession {
  id: string;
  status: string;
  progress: number;
  mode: "few-shot" | "full-retrain";
  metrics?: TrainingMetrics;
  model_version_produced?: string | null;
  deployment_ready: boolean;
  deploy_threshold: number;
  final_val_accuracy: number | null;
  recommended_next_action: "deploy" | "collect_more_examples" | "wait" | "review_error";
}

type TrainingStartPayload =
  | {
      sign_id: string;
      mode: "few-shot";
      config?: {
        epochs?: number;
        learning_rate?: number;
        augmentation?: boolean;
        min_deploy_accuracy?: number;
      };
    }
  | {
      sign_id?: undefined;
      mode: "full-retrain";
      config?: {
        epochs?: number;
        learning_rate?: number;
        augmentation?: boolean;
        min_deploy_accuracy?: number;
      };
    };

export async function startTraining(payload: TrainingStartPayload): Promise<TrainingSession> {
  return apiFetch<TrainingSession>("/training/sessions", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function stopTraining(sessionId: string): Promise<{ status: string }> {
  return apiFetch<{ status: string }>(`/training/sessions/${sessionId}/stop`, {
    method: "POST"
  });
}

export async function getTrainingSession(sessionId: string): Promise<TrainingSession> {
  return apiFetch<TrainingSession>(`/training/sessions/${sessionId}`);
}

export async function listTrainingSessions(): Promise<TrainingSession[]> {
  return apiFetch<TrainingSession[]>("/training/sessions");
}

export async function deployTrainingSession(sessionId: string): Promise<{
  status: "deployed";
  session_id: string;
  active_model_id: string;
  version: string;
}> {
  return apiFetch<{
    status: "deployed";
    session_id: string;
    active_model_id: string;
    version: string;
  }>(`/training/sessions/${sessionId}/deploy`, {
    method: "POST"
  });
}
