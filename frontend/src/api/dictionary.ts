import { apiBaseUrl, apiFetch } from "./client";

export interface GraphPayload {
  nodes: Array<{
    id: string;
    label: string;
    category?: string;
    video_count?: number;
    usage_count?: number;
    accuracy?: number | null;
    thumbnail_url?: string | null;
  }>;
  edges: Array<{ source: string; target: string; relation_type: string; weight: number }>;
}

export function getGraph(): Promise<GraphPayload> {
  return apiFetch<GraphPayload>("/dictionary/graph");
}

export function searchDictionary(query: string): Promise<Array<{ id: string; name: string; tags: string[] }>> {
  return apiFetch<Array<{ id: string; name: string; tags: string[] }>>(`/dictionary/search?q=${encodeURIComponent(query)}`);
}

export async function exportDictionary(format: "json" | "markdown" | "obsidian-vault"): Promise<Blob> {
  const form = new FormData();
  form.append("format", format);
  const response = await fetch(`${apiBaseUrl()}/api/v1/dictionary/export`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error ${response.status}`);
  }
  return response.blob();
}

export async function importDictionary(file: File): Promise<{
  imported_signs: number;
  imported_notes: number;
  skipped: number;
  errors: string[];
}> {
  const form = new FormData();
  form.append("archive", file, file.name);

  const response = await fetch(`${apiBaseUrl()}/api/v1/dictionary/import`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error ${response.status}`);
  }
  return (await response.json()) as {
    imported_signs: number;
    imported_notes: number;
    skipped: number;
    errors: string[];
  };
}
