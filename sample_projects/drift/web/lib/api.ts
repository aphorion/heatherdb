const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChainStep {
  step: number;
  concept: string;
  similarity: number;
  drift: number;
  distance_from_start: number;
  nearby: { concept: string; similarity: number }[];
  converged: boolean;
}

export interface LandscapePoint {
  name: string;
  x: number;
  y: number;
}

export interface TrajectoryPoint {
  x: number;
  y: number;
}

export interface ThinkResponse {
  chain: ChainStep[];
  landscape: {
    concepts: LandscapePoint[];
    trajectory: TrajectoryPoint[];
  };
  stats: {
    concepts: number;
    pairs_written: number;
  };
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
  seed: (text: string) =>
    request<{ concepts: number; pairs_written: number }>("/api/seed", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  think: (
    stimulus: string,
    params: { steps?: number; alpha?: number; beta?: number; gamma?: number } = {}
  ) =>
    request<ThinkResponse>("/api/think", {
      method: "POST",
      body: JSON.stringify({ stimulus, ...params }),
    }),

  loadStarter: () =>
    request<{ concepts: number; pairs_written: number }>("/api/load-starter", {
      method: "POST",
    }),

  reset: () => request<{ cleared: boolean }>("/api/reset", { method: "POST" }),

  concepts: () => request<{ concepts: string[]; count: number }>("/api/concepts"),

  stats: () =>
    request<{ concepts: number; pairs_written: number; heather: Record<string, unknown> }>(
      "/api/stats"
    ),
};
