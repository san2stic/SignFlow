import { apiFetch } from "./client";

export interface ModelVersion {
  id: string;
  version: string;
  is_active: boolean;
  accuracy: number;
  created_at: string;
}

export function listModels(): Promise<ModelVersion[]> {
  return apiFetch<ModelVersion[]>("/models");
}

export function getActiveModel(): Promise<ModelVersion | null> {
  return apiFetch<ModelVersion | null>("/models/active");
}

export function activateModel(modelId: string): Promise<{ active_model_id: string; version: string }> {
  return apiFetch<{ active_model_id: string; version: string }>(`/models/${modelId}/activate`, {
    method: "POST"
  });
}

export interface ModelExportResponse {
  model_id: string;
  version: string;
  format: string;
  path: string;
}

export function exportModel(modelId: string, format: "pt" | "onnx" = "pt"): Promise<ModelExportResponse> {
  return apiFetch<ModelExportResponse>(`/models/${modelId}/export?format=${encodeURIComponent(format)}`);
}
