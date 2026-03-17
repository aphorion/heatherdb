const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface RecalledMemory {
  id: number;
  role: "user" | "assistant";
  content: string;
  similarity: number;
  memory_type: "vivid" | "clear" | "vague";
  timestamp: string;
}

export interface ChatResponse {
  response: string;
  recalled_memories: RecalledMemory[];
  fidelity: number;
  recent_tokens: number;
  recalled_tokens: number;
}

export interface Stats {
  message_count: number;
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
  chat: (message: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  messages: (limit = 50) =>
    request<{ messages: Message[] }>(`/api/messages?limit=${limit}`),

  stats: () => request<Stats>("/api/stats"),
};
