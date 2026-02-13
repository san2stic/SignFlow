const RAW_API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.trim();

function trimTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, "");
}

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

const API_URL = RAW_API_URL ? trimTrailingSlashes(RAW_API_URL) : "";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}/api/v1${normalizePath(path)}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error ${response.status}`);
  }

  return (await response.json()) as T;
}

export function apiBaseUrl(): string {
  return API_URL;
}
