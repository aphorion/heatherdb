"use client";

import { useState } from "react";
import { api, type DreamResponse } from "@/lib/api";

export default function DreamPanel() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DreamResponse | null>(null);
  const [hops, setHops] = useState(5);

  async function handleDream() {
    setLoading(true);
    try {
      const res = await api.dream(hops);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Dream</h2>
        <p className="text-sm text-zinc-500">
          Free association from random noise. The database wanders through its own
          knowledge.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleDream}
          disabled={loading}
          className="px-5 py-2.5 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Dreaming..." : "Dream"}
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500">Hops</label>
          <input
            type="number"
            value={hops}
            onChange={(e) => setHops(Math.max(2, Math.min(10, parseInt(e.target.value) || 5)))}
            min={2}
            max={10}
            className="w-16 px-2 py-1.5 bg-zinc-900 border border-zinc-800 rounded text-sm text-zinc-300 text-center focus:outline-none focus:border-zinc-600"
          />
        </div>
      </div>

      {result && (
        <div className="space-y-5 animate-in fade-in duration-300">
          {/* Theme */}
          {result.theme.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-zinc-500 uppercase tracking-wider mr-1">
                Theme
              </span>
              {result.theme.map((w, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 bg-indigo-500/10 text-indigo-400 text-xs rounded border border-indigo-500/20"
                >
                  {w}
                </span>
              ))}
            </div>
          )}

          {/* Chain */}
          <div className="relative">
            {result.hops.map((hop, i) => (
              <div key={i} className="flex gap-4 mb-1">
                {/* Vertical line + dot */}
                <div className="flex flex-col items-center w-6 shrink-0">
                  <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 border-2 border-zinc-950 z-10" />
                  {i < result.hops.length - 1 && (
                    <div className="w-px flex-1 bg-gradient-to-b from-indigo-500/50 to-indigo-500/10" />
                  )}
                </div>
                {/* Content */}
                <div className="pb-5 flex-1 min-w-0">
                  <div className="p-3.5 bg-zinc-900/50 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors">
                    <p className="text-sm text-zinc-200 leading-relaxed">
                      {hop.text}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <span className="text-[10px] text-zinc-600 uppercase tracking-wider">
                        Hop {i + 1}
                      </span>
                      <span className="text-[11px] font-mono text-zinc-500">
                        {Math.round(hop.similarity * 100)}% association
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={handleDream}
            disabled={loading}
            className="px-4 py-2 text-sm text-zinc-400 border border-zinc-800 rounded-lg hover:border-zinc-600 hover:text-zinc-300 transition-colors"
          >
            Dream again
          </button>
        </div>
      )}
    </div>
  );
}
