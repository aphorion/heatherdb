const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Prediction {
  word: string;
  confidence: number;
}

export interface TypeResponse {
  predictions: Prediction[];
  stats: {
    vocab_size: number;
    patterns: number;
    words: number;
  };
}

export interface FeedResponse {
  words_processed: number;
  patterns_written: number;
  predictions: Prediction[];
  stats: {
    vocab_size: number;
    patterns: number;
    words: number;
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
  type: (word: string, context: string[]) =>
    request<TypeResponse>("/api/type", {
      method: "POST",
      body: JSON.stringify({ word, context }),
    }),

  feed: (text: string) =>
    request<FeedResponse>("/api/feed", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  reset: () =>
    request<{ cleared: boolean }>("/api/reset", { method: "POST" }),

  stats: () =>
    request<{ vocab_size: number; patterns: number; words: number }>(
      "/api/stats"
    ),
};
