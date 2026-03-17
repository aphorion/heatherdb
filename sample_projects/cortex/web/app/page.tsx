"use client";

import { useState, useEffect } from "react";
import NoteInput from "@/components/NoteInput";
import LateralPanel from "@/components/LateralPanel";
import EchoPanel from "@/components/EchoPanel";
import SurprisePanel from "@/components/SurprisePanel";
import DreamPanel from "@/components/DreamPanel";
import LandscapePanel from "@/components/LandscapePanel";
import NotesList from "@/components/NotesList";
import { api, type StatsResponse } from "@/lib/api";

type Tab = "write" | "lateral" | "echo" | "surprise" | "dream" | "landscape" | "notes";

const tabs: { id: Tab; label: string; icon: string }[] = [
  { id: "write", label: "Write", icon: "M" },
  { id: "lateral", label: "Lateral", icon: "/" },
  { id: "echo", label: "Echo", icon: ")" },
  { id: "surprise", label: "Surprise", icon: "!" },
  { id: "dream", label: "Dream", icon: "*" },
  { id: "landscape", label: "Landscape", icon: "#" },
  { id: "notes", label: "Notes", icon: "=" },
];

export default function Home() {
  const [active, setActive] = useState<Tab>("write");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    api.stats().then(setStats).catch(() => {});
  }, [refreshKey]);

  function handleNoteWritten() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <nav className="w-56 border-r border-zinc-800 bg-zinc-950 flex flex-col shrink-0">
        <div className="p-5 border-b border-zinc-800">
          <h1 className="text-base font-semibold text-zinc-100 tracking-tight">
            Cortex
          </h1>
          <p className="text-[11px] text-zinc-600 mt-0.5">
            lateral thinking engine
          </p>
        </div>

        <div className="flex-1 py-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className={`w-full flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                active === tab.id
                  ? "text-zinc-100 bg-zinc-800/50"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50"
              }`}
            >
              <span
                className={`w-5 h-5 flex items-center justify-center text-xs font-mono rounded ${
                  active === tab.id
                    ? "bg-zinc-700 text-zinc-200"
                    : "bg-zinc-900 text-zinc-600"
                }`}
              >
                {tab.icon}
              </span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Stats */}
        <div className="p-4 border-t border-zinc-800 space-y-1.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-zinc-600">Notes</span>
            <span className="font-mono text-zinc-400">
              {stats?.note_count ?? "..."}
            </span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-zinc-600">HeatherDB</span>
            <span className="font-mono text-zinc-400">
              {stats?.heather_stats
                ? `${(stats.heather_stats as Record<string, number>).hard_locations ?? "?"} locs`
                : "..."}
            </span>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto p-8">
          {active === "write" && <NoteInput onNoteWritten={handleNoteWritten} />}
          {active === "lateral" && <LateralPanel />}
          {active === "echo" && <EchoPanel />}
          {active === "surprise" && <SurprisePanel />}
          {active === "dream" && <DreamPanel />}
          {active === "landscape" && <LandscapePanel />}
          {active === "notes" && <NotesList refreshKey={refreshKey} />}
        </div>
      </main>
    </div>
  );
}
