/**
 * Typed API client for all /studio/* endpoints.
 */

import { apiFetch } from "./client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AnnotationSession {
  id: number;
  name: string;
  description: string | null;
  status: "active" | "completed" | "archived";
  created_at: string;
  updated_at: string;
}

export interface AnnotationSessionWithStats extends AnnotationSession {
  video_count: number;
  annotation_count: number;
  verified_count: number;
  coverage_percent: number;
}

export interface CreateSessionData {
  name: string;
  description?: string;
  status?: "active" | "completed" | "archived";
}

export interface UpdateSessionData {
  name?: string;
  description?: string;
  status?: "active" | "completed" | "archived";
}

export interface NMMTags {
  polar_question?: boolean;
  wh_question?: boolean;
  negation?: boolean;
  eyebrow_raise?: boolean;
  eyebrow_furrow?: boolean;
  head_nod?: boolean;
  head_shake?: boolean;
  mouth_gesture?: string;
}

export interface VideoAnnotation {
  id: number;
  video_id: string;
  session_id: number;
  sign_label: string;
  start_frame: number;
  end_frame: number;
  start_time_ms: number;
  end_time_ms: number;
  confidence: number | null;
  is_verified: boolean;
  annotator_notes: string | null;
  nmm_tags: NMMTags | null;
  created_at: string;
}

export interface CreateAnnotationData {
  sign_label: string;
  start_frame: number;
  end_frame: number;
  start_time_ms: number;
  end_time_ms: number;
  confidence?: number;
  is_verified?: boolean;
  annotator_notes?: string;
  nmm_tags?: NMMTags;
}

export interface UpdateAnnotationData {
  sign_label?: string;
  start_frame?: number;
  end_frame?: number;
  start_time_ms?: number;
  end_time_ms?: number;
  confidence?: number;
  is_verified?: boolean;
  annotator_notes?: string;
  nmm_tags?: NMMTags;
}

export interface SignSequenceItem {
  label: string;
  start: number;
  end: number;
  start_ms?: number;
  end_ms?: number;
}

export interface GrammarAnnotation {
  id: number;
  session_id: number;
  video_id: string;
  sign_sequence: SignSequenceItem[];
  french_translation: string;
  grammar_tags: Record<string, unknown> | null;
  annotator_id: number | null;
  created_at: string;
}

export interface CreateGrammarAnnotationData {
  session_id: number;
  video_id: string;
  sign_sequence?: SignSequenceItem[];
  french_translation: string;
  grammar_tags?: Record<string, unknown>;
  annotator_id?: number;
}

export interface UpdateGrammarAnnotationData {
  sign_sequence?: SignSequenceItem[];
  french_translation?: string;
  grammar_tags?: Record<string, unknown>;
}

export interface VideoInSession {
  id: string;
  file_path: string;
  thumbnail_path: string | null;
  duration_ms: number;
  fps: number;
  resolution: string;
  landmarks_extracted: boolean;
  detection_rate: number;
  quality_score: number;
  sign_id: string | null;
  annotation_count: number;
  verified_count: number;
}

export interface StudioStats {
  total_sessions: number;
  active_sessions: number;
  total_videos_annotated: number;
  total_annotations: number;
  verified_annotations: number;
  total_grammar_annotations: number;
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export async function fetchSessions(
  statusFilter?: string
): Promise<AnnotationSessionWithStats[]> {
  const qs = statusFilter ? `?status=${encodeURIComponent(statusFilter)}` : "";
  return apiFetch<AnnotationSessionWithStats[]>(`/studio/sessions${qs}`);
}

export async function createSession(
  data: CreateSessionData
): Promise<AnnotationSession> {
  return apiFetch<AnnotationSession>("/studio/sessions", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function fetchSession(
  sessionId: number
): Promise<AnnotationSessionWithStats> {
  return apiFetch<AnnotationSessionWithStats>(`/studio/sessions/${sessionId}`);
}

export async function updateSession(
  sessionId: number,
  data: UpdateSessionData
): Promise<AnnotationSession> {
  return apiFetch<AnnotationSession>(`/studio/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function deleteSession(sessionId: number): Promise<void> {
  await apiFetch<void>(`/studio/sessions/${sessionId}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Videos inside a session
// ---------------------------------------------------------------------------

export async function fetchSessionVideos(
  sessionId: number
): Promise<VideoInSession[]> {
  return apiFetch<VideoInSession[]>(`/studio/sessions/${sessionId}/videos`);
}

export async function uploadVideosToSession(
  sessionId: number,
  files: File[]
): Promise<VideoInSession[]> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const RAW_API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.trim() ?? "";
  const base = RAW_API_URL.replace(/\/+$/, "");
  const response = await fetch(
    `${base}/api/v1/studio/sessions/${sessionId}/videos`,
    {
      method: "POST",
      body: formData,
    }
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Upload error ${response.status}`);
  }
  return response.json() as Promise<VideoInSession[]>;
}

// ---------------------------------------------------------------------------
// Video annotations
// ---------------------------------------------------------------------------

export async function fetchVideoAnnotations(
  videoId: string,
  sessionId?: number
): Promise<VideoAnnotation[]> {
  const qs = sessionId !== undefined ? `?session_id=${sessionId}` : "";
  return apiFetch<VideoAnnotation[]>(
    `/studio/videos/${videoId}/annotations${qs}`
  );
}

export async function createAnnotation(
  videoId: string,
  sessionId: number,
  data: CreateAnnotationData
): Promise<VideoAnnotation> {
  return apiFetch<VideoAnnotation>(
    `/studio/videos/${videoId}/annotations?session_id=${sessionId}`,
    {
      method: "POST",
      body: JSON.stringify(data),
    }
  );
}

export async function updateAnnotation(
  annotationId: number,
  data: UpdateAnnotationData
): Promise<VideoAnnotation> {
  return apiFetch<VideoAnnotation>(`/studio/annotations/${annotationId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteAnnotation(annotationId: number): Promise<void> {
  await apiFetch<void>(`/studio/annotations/${annotationId}`, {
    method: "DELETE",
  });
}

export async function bulkCreateAnnotations(
  videoId: string,
  sessionId: number,
  annotations: CreateAnnotationData[]
): Promise<VideoAnnotation[]> {
  return apiFetch<VideoAnnotation[]>(
    `/studio/videos/${videoId}/annotations/bulk?session_id=${sessionId}`,
    {
      method: "POST",
      body: JSON.stringify({ annotations }),
    }
  );
}

export async function autoSuggestAnnotations(
  videoId: string,
  sessionId: number
): Promise<VideoAnnotation[]> {
  return apiFetch<VideoAnnotation[]>(
    `/studio/videos/${videoId}/annotations/auto-suggest?session_id=${sessionId}`,
    { method: "POST" }
  );
}

// ---------------------------------------------------------------------------
// Grammar annotations
// ---------------------------------------------------------------------------

export async function fetchGrammarAnnotations(
  sessionId: number
): Promise<GrammarAnnotation[]> {
  return apiFetch<GrammarAnnotation[]>(`/studio/sessions/${sessionId}/grammar`);
}

export async function createGrammarAnnotation(
  data: CreateGrammarAnnotationData
): Promise<GrammarAnnotation> {
  return apiFetch<GrammarAnnotation>("/studio/grammar", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateGrammarAnnotation(
  grammarId: number,
  data: UpdateGrammarAnnotationData
): Promise<GrammarAnnotation> {
  return apiFetch<GrammarAnnotation>(`/studio/grammar/${grammarId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export function exportSessionUrl(
  sessionId: number,
  format: "json" | "csv" | "elan"
): string {
  const RAW_API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.trim() ?? "";
  const base = RAW_API_URL.replace(/\/+$/, "");
  return `${base}/api/v1/studio/sessions/${sessionId}/export?format=${format}`;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

export async function fetchStudioStats(): Promise<StudioStats> {
  return apiFetch<StudioStats>("/studio/stats");
}
