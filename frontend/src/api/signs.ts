import { apiBaseUrl, apiFetch } from "./client";

export interface Sign {
  id: string;
  name: string;
  slug: string;
  description?: string;
  category?: string;
  tags: string[];
  variants: string[];
  related_signs: string[];
  video_count: number;
  training_sample_count: number;
  accuracy?: number;
  usage_count: number;
  notes?: string;
  created_at?: string;
  updated_at?: string;
}

export interface SignListResponse {
  items: Sign[];
  page: number;
  per_page: number;
  total: number;
}

export interface SignVideo {
  id: string;
  sign_id: string;
  file_path: string;
  thumbnail_path?: string;
  duration_ms: number;
  fps: number;
  resolution: string;
  type: "training" | "reference" | "example";
  landmarks_extracted: boolean;
  landmarks_path?: string;
  detection_rate: number;
  quality_score: number;
  is_trainable: boolean;
  landmark_feature_dim: number;
  created_at: string;
}

export async function listSigns(query = ""): Promise<SignListResponse> {
  const suffix = query ? `?search=${encodeURIComponent(query)}` : "";
  return apiFetch<SignListResponse>(`/signs${suffix}`);
}

export async function getSign(signId: string): Promise<Sign> {
  return apiFetch<Sign>(`/signs/${signId}`);
}

export async function createSign(payload: {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
  variants?: string[];
  related_signs?: string[];
  notes?: string;
}): Promise<Sign> {
  return apiFetch<Sign>("/signs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function updateSign(
  signId: string,
  payload: Partial<{
    name: string;
    description: string;
    category: string;
    tags: string[];
    variants: string[];
    related_signs: string[];
    notes: string;
  }>
): Promise<Sign> {
  return apiFetch<Sign>(`/signs/${signId}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export async function deleteSign(signId: string): Promise<void> {
  const response = await fetch(`${apiBaseUrl()}/api/v1/signs/${signId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error ${response.status}`);
  }
}

export interface Backlink {
  id: string;
  name: string;
  slug: string;
}

export async function getSignBacklinks(signId: string): Promise<{ sign_id: string; backlinks: Backlink[] }> {
  return apiFetch<{ sign_id: string; backlinks: Backlink[] }>(`/signs/${signId}/backlinks`);
}

export async function uploadSignVideo(
  signId: string,
  file: File,
  options?: {
    type?: "training" | "reference" | "example";
    durationMs?: number;
    fps?: number;
    resolution?: string;
  }
): Promise<SignVideo> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("type", options?.type ?? "training");
  form.append(
    "metadata",
    JSON.stringify({
      duration_ms: options?.durationMs ?? 0,
      fps: options?.fps ?? 30,
      resolution: options?.resolution ?? "640x480"
    })
  );

  const response = await fetch(`${apiBaseUrl()}/api/v1/signs/${signId}/videos`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    let message = `API error ${response.status}`;
    const contentType = response.headers.get("content-type") ?? "";

    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as { detail?: string };
      if (payload?.detail) {
        message = payload.detail;
      }
    } else {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }
    throw new Error(message);
  }

  return (await response.json()) as SignVideo;
}

export async function listSignVideos(signId: string): Promise<SignVideo[]> {
  return apiFetch<SignVideo[]>(`/signs/${signId}/videos`);
}
