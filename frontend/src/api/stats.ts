import { apiFetch } from "./client";

export interface OverviewStats {
  total_signs: number;
  total_videos: number;
  model_accuracy: number;
  total_translations: number;
  most_used_signs: Array<{ sign: string; count: number }>;
  recent_activity: Array<{ action: string; timestamp: string }>;
}

export interface AccuracyHistoryPoint {
  id: string;
  version: string;
  accuracy: number;
  created_at: string;
  is_active: boolean;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export function getOverviewStats(): Promise<OverviewStats> {
  return apiFetch<OverviewStats>("/stats/overview");
}

export function getAccuracyHistory(limit = 50): Promise<AccuracyHistoryPoint[]> {
  return apiFetch<AccuracyHistoryPoint[]>(`/stats/accuracy-history?limit=${limit}`);
}

export function getSignsPerCategory(): Promise<CategoryCount[]> {
  return apiFetch<CategoryCount[]>("/stats/signs-per-category");
}
