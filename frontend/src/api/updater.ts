import { apiFetch } from "./client";

// Types alignés avec les schémas Pydantic backend
export type DeploymentStatus =
  | "idle"
  | "fetching"
  | "pulling"
  | "building"
  | "deploying"
  | "success"
  | "error"
  | "rolled_back";

export interface DeploymentHistory {
  id: number;
  status: DeploymentStatus;
  commit_hash: string | null;
  previous_commit_hash: string | null;
  commit_message: string | null;
  commit_author: string | null;
  build_log: string | null;
  error_message: string | null;
  build_duration_s: number | null;
  deploy_duration_s: number | null;
  total_duration_s: number | null;
  triggered_by: "auto" | "manual";
  rollback_of_id: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string;
}

export interface UpdaterStatus {
  state: DeploymentStatus;
  current_deployment_id: number | null;
  last_deployment: DeploymentHistory | null;
  git_remote_url: string;
  git_branch: string;
  local_commit: string | null;
  remote_commit: string | null;
  last_check_at: string | null;
  poll_interval_s: number;
  auto_update_enabled: boolean;
}

export interface TriggerResponse {
  deployment_id: number;
  message: string;
}

export async function getStatus(): Promise<UpdaterStatus> {
  return apiFetch<UpdaterStatus>("/updater/status");
}

export async function getHistory(limit: number = 20): Promise<DeploymentHistory[]> {
  return apiFetch<DeploymentHistory[]>(`/updater/history?limit=${limit}`);
}

export async function triggerUpdate(): Promise<TriggerResponse> {
  return apiFetch<TriggerResponse>("/updater/trigger", {
    method: "POST"
  });
}

export async function rollbackTo(id: number): Promise<TriggerResponse> {
  return apiFetch<TriggerResponse>(`/updater/rollback/${id}`, {
    method: "POST"
  });
}
