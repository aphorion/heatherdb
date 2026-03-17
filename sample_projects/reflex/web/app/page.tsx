"use client";

import { useState } from "react";
import DrawPanel from "@/components/DrawPanel";
import CompletePanel from "@/components/CompletePanel";
import BlendPanel from "@/components/BlendPanel";
import SurprisePanel from "@/components/SurprisePanel";
import PatternsPanel from "@/components/PatternsPanel";

const tabs = [
  { id: "draw", label: "Draw" },
  { id: "complete", label: "Complete" },
  { id: "blend", label: "Blend" },
  { id: "surprise", label: "Surprise" },
  { id: "patterns", label: "Patterns" },
] as const;

type Tab = (typeof tabs)[number]["id"];

export default function Home() {
  const [tab, setTab] = useState<Tab>("draw");
  const [refreshKey, setRefreshKey] = useState(0);

  function handlePatternWritten() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">Reflex</h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              SDM as sole intelligence — no neural networks, no LLMs
            </p>
          </div>
          <div className="text-xs text-zinc-600 font-mono">64 dims</div>
        </div>
      </header>

      <nav className="border-b border-zinc-800 px-6">
        <div className="max-w-3xl mx-auto flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium transition-colors relative ${
                tab === t.id
                  ? "text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t.label}
              {tab === t.id && (
                <div className="absolute bottom-0 left-2 right-2 h-0.5 bg-zinc-100 rounded-full" />
              )}
            </button>
          ))}
        </div>
      </nav>

      <main className="flex-1 px-6 py-8">
        <div className="max-w-3xl mx-auto">
          {tab === "draw" && (
            <DrawPanel onPatternWritten={handlePatternWritten} />
          )}
          {tab === "complete" && <CompletePanel />}
          {tab === "blend" && <BlendPanel key={refreshKey} />}
          {tab === "surprise" && <SurprisePanel />}
          {tab === "patterns" && <PatternsPanel key={refreshKey} />}
        </div>
      </main>
    </div>
  );
}
