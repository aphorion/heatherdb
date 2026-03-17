"use client";

import { useState } from "react";
import { api, type SurpriseResponse } from "@/lib/api";
import ResultCard from "./ResultCard";

function noveltyColor(v: number): string {
  if (v >= 0.6) return "text-emerald-400";
  if (v >= 0.3) return "text-amber-400";
  return "text-rose-400";
}

function noveltyLabel(v: number): string {
  if (v >= 0.7) return "Highly novel";
  if (v >= 0.5) return "Mostly new";
  if (v >= 0.3) return "Partially familiar";
  return "Already well-covered";
}

function noveltyBg(v: number): string {
  if (v >= 0.6) return "border-emerald-500/30 bg-emerald-500/5";
  if (v >= 0.3) return "border-amber-500/30 bg-amber-500/5";
  return "border-rose-500/30 bg-rose-500/5";
}

export default function SurprisePanel() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SurpriseResponse | null>(null);

  async function handleSurprise() {
    if (!text.trim() || loading) return;
    setLoading(true);
    try {
      const res = await api.surprise(text.trim());
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? Math.round(result.novelty * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Surprise</h2>
        <p className="text-sm text-zinc-500">
          Test how novel an idea is to your notebook. Low fidelity = high novelty.
        </p>
      </div>

      <div className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write an observation to test its novelty..."
          className="w-full h-24 px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 resize-none transition-colors"
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.metaKey) handleSurprise();
          }}
        />
        <button
          onClick={handleSurprise}
          disabled={!text.trim() || loading}
          className="px-5 py-2.5 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Analyzing..." : "Test novelty"}
        </button>
      </div>

      {result && (
        <div className="space-y-4 animate-in fade-in duration-300">
          <div className={`p-5 border rounded-lg ${noveltyBg(result.novelty)}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-zinc-400">Novelty</span>
              <span className={`text-3xl font-mono font-bold ${noveltyColor(result.novelty)}`}>
                {pct}%
              </span>
            </div>
            <div className="w-full h-3 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-3 rounded-full transition-all duration-700 ${
                  result.novelty >= 0.6
                    ? "bg-emerald-500"
                    : result.novelty >= 0.3
                    ? "bg-amber-500"
                    : "bg-rose-500"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-zinc-400">{noveltyLabel(result.novelty)}</p>
          </div>

          {result.familiar_notes.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                Reminds me of...
              </h3>
              {result.familiar_notes.map((r) => (
                <ResultCard key={r.id} result={r} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
