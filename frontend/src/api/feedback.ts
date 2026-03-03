import { apiFetch } from "./client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FeedbackStatsEntry {
  predicted_sign: string;
  corrected_sign: string;
  count: number;
}

export interface FeedbackStats {
  total: number;
  by_sign: FeedbackStatsEntry[];
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * GET /api/v1/feedback/stats
 * Récupère les statistiques de corrections par signe.
 */
export async function fetchFeedbackStats(): Promise<FeedbackStats> {
  return apiFetch<FeedbackStats>("/feedback/stats");
}

/**
 * DELETE /api/v1/feedback/corrections/{id}
 * Ignore (supprime) une correction existante.
 */
export async function deleteFeedbackCorrection(id: number): Promise<void> {
  await apiFetch<void>(`/feedback/corrections/${id}`, { method: "DELETE" });
}
