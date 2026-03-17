const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Grid = number[][];

export interface WriteResponse {
  id: number;
  fidelity: number;
}

export interface CompleteResponse {
  completed: Grid;
  confidence: number[][];
  fidelity: number;
  original: Grid;
}

export interface BlendResponse {
  grid: Grid;
  confidence: number[][];
  fidelity: number;
}

export interface SurpriseResponse {
  novelty: number;
  fidelity: number;
}

export interface Pattern {
  id: number;
  name: string;
  grid: Grid;
  created_at: string;
}

export interface StatsResponse {
  pattern_count: number;
  heather_stats: Record<string, unknown>;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  write: (grid: Grid, name = "") =>
    request<WriteResponse>("/api/write", {
      method: "POST",
      body: JSON.stringify({ grid, name }),
    }),

  complete: (grid: Grid) =>
    request<CompleteResponse>("/api/complete", {
      method: "POST",
      body: JSON.stringify({ grid }),
    }),

  blend: (patternIds: number[]) =>
    request<BlendResponse>("/api/blend", {
      method: "POST",
      body: JSON.stringify({ pattern_ids: patternIds }),
    }),

  surprise: (grid: Grid) =>
    request<SurpriseResponse>("/api/surprise", {
      method: "POST",
      body: JSON.stringify({ grid }),
    }),

  patterns: () => request<{ patterns: Pattern[] }>("/api/patterns"),

  deletePattern: (id: number) =>
    request<{ deleted: boolean }>(`/api/patterns/${id}`, { method: "DELETE" }),

  stats: () => request<StatsResponse>("/api/stats"),
};
