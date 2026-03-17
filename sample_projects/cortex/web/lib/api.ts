const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface NoteResult {
  id: number;
  text: string;
  similarity: number;
}

export interface WriteResponse {
  id: number;
  fidelity_hint: number;
}

export interface LateralInsight {
  source: NoteResult;
  bridge: string;
}

export interface LateralResponse {
  problem: string;
  fidelity: number;
  laterals: LateralInsight[];
  direct: NoteResult[];
  llm_answer: string | null;
}

export interface EchoResponse {
  note: NoteResult;
  fidelity: number;
  interfering: NoteResult[];
}

export interface SurpriseResponse {
  novelty: number;
  familiar_notes: NoteResult[];
}

export interface DreamHop {
  text: string;
  similarity: number;
}

export interface DreamResponse {
  hops: DreamHop[];
  theme: string[];
}

export interface Basin {
  label: string;
  fidelity: number;
  notes: NoteResult[];
}

export interface LandscapeResponse {
  basins: Basin[];
}

export interface Note {
  id: number;
  text: string;
  created_at: string;
}

export interface StatsResponse {
  note_count: number;
  heather_stats: Record<string, unknown>;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

export const api = {
  write: (text: string) =>
    request<WriteResponse>("/api/write", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  lateral: (problem: string, k = 5) =>
    request<LateralResponse>("/api/lateral", {
      method: "POST",
      body: JSON.stringify({ problem, k }),
    }),

  llmStatus: () => request<{ enabled: boolean }>("/api/llm-status"),

  echo: (noteId: number) =>
    request<EchoResponse>(`/api/echo/${noteId}`),

  surprise: (text: string) =>
    request<SurpriseResponse>("/api/surprise", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  dream: (hops = 5) =>
    request<DreamResponse>("/api/dream", {
      method: "POST",
      body: JSON.stringify({ hops }),
    }),

  landscape: (probes = 20) =>
    request<LandscapeResponse>(`/api/landscape?probes=${probes}`),

  notes: () => request<{ notes: Note[] }>("/api/notes"),

  load: (texts: string[]) =>
    request<{ count: number }>("/api/load", {
      method: "POST",
      body: JSON.stringify({ texts }),
    }),

  stats: () => request<StatsResponse>("/api/stats"),

  deleteNote: (id: number) =>
    request<{ deleted: boolean }>(`/api/notes/${id}`, { method: "DELETE" }),
};
