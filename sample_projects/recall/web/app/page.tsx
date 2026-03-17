"use client";

import { useState, useEffect, useCallback } from "react";
import { api, type ChatResponse, type Stats } from "@/lib/api";
import ChatInterface from "@/components/ChatInterface";
import MemoryPanel from "@/components/MemoryPanel";

export default function Home() {
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  const refreshStats = useCallback(() => {
    api.stats().then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  const handleResponse = useCallback(
    (res: ChatResponse) => {
      setLastResponse(res);
      refreshStats();
    },
    [refreshStats]
  );

  const msgCount = stats?.message_count ?? 0;
  const hardLocations =
    (stats?.heather_stats as Record<string, number>)?.hard_locations ?? "—";

  return (
    <div className="h-screen flex flex-col">
      {/* header */}
      <header className="border-b border-zinc-800 px-6 py-3 flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-sm font-bold tracking-tight">Recall</h1>
          <p className="text-[10px] text-zinc-600">
            SDM long-term memory for conversations
          </p>
        </div>
        <div className="flex gap-4 text-[10px] text-zinc-600 font-[family-name:var(--font-geist-mono)]">
          <span>{msgCount} messages</span>
          <span>{hardLocations} locations</span>
        </div>
      </header>

      {/* main */}
      <div className="flex-1 flex min-h-0">
        <ChatInterface onResponse={handleResponse} />
        <MemoryPanel lastResponse={lastResponse} />
      </div>
    </div>
  );
}
