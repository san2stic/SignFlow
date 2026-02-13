import { apiFetch } from './client';

export interface UnlabeledVideo {
  id: string;
  file_path: string;
  thumbnail_path?: string;
  duration_ms: number;
  fps: number;
  resolution: string;
  landmarks_extracted: boolean;
  landmarks_path?: string;
  created_at: string;
  sign_id: null;
}

export interface SuggestedVideo extends Omit<UnlabeledVideo, 'sign_id'> {
  similarity_score: number;
}

export async function getUnlabeledVideos(): Promise<UnlabeledVideo[]> {
  const response = await apiFetch<{ items: UnlabeledVideo[] }>(
    '/videos/unlabeled'
  );
  return response.items;
}

export async function labelVideo(
  videoId: string,
  signId: string
): Promise<void> {
  await apiFetch<void>(`/videos/${videoId}/label`, {
    method: 'PATCH',
    body: JSON.stringify({ sign_id: signId })
  });
}

export async function getSuggestions(
  videoId: string,
  threshold: number = 0.75
): Promise<{ suggestions: SuggestedVideo[] }> {
  const query = `?threshold=${encodeURIComponent(String(threshold))}`;
  return apiFetch<{ suggestions: SuggestedVideo[] }>(
    `/videos/${videoId}/suggestions${query}`,
    {
      method: 'POST'
    }
  );
}

export async function bulkLabelVideos(
  videoIds: string[],
  signId: string
): Promise<void> {
  await apiFetch<void>('/videos/bulk-label', {
    method: 'PATCH',
    body: JSON.stringify({
      video_ids: videoIds,
      sign_id: signId
    })
  });
}
