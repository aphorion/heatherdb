"use client";

import { useState, useEffect, useCallback } from "react";
import { api, type Pattern, type StatsResponse } from "@/lib/api";

export default function PatternsPanel() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  const load = useCallback(async () => {
    const [p, s] = await Promise.all([api.patterns(), api.stats()]);
    setPatterns(p.patterns);
    setStats(s);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleDelete(id: number) {
    await api.deletePattern(id);
    load();
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Patterns</h2>
        <p className="text-sm text-zinc-500">
          All patterns stored in the EAM.
        </p>
      </div>

      {stats && (
        <div className="flex gap-4">
          <div className="px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg flex-1">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">
              Patterns
            </p>
            <p className="text-2xl font-mono font-bold text-zinc-100 mt-1">
              {stats.pattern_count}
            </p>
          </div>
          <div className="px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg flex-1">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">
              Hard locations
            </p>
            <p className="text-2xl font-mono font-bold text-zinc-100 mt-1">
              {(stats.heather_stats as Record<string, number>).hard_locations ??
                "—"}
            </p>
          </div>
        </div>
      )}

      {patterns.length === 0 ? (
        <p className="text-sm text-zinc-600 italic">
          No patterns stored yet. Go to Draw to add some.
        </p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {patterns.map((p) => (
            <div
              key={p.id}
              className="p-3 bg-zinc-900 border border-zinc-800 rounded-lg group"
            >
              <MiniGrid grid={p.grid} />
              <div className="flex items-center justify-between mt-2">
                <p className="text-xs text-zinc-400 truncate">
                  {p.name || `#${p.id}`}
                </p>
                <button
                  onClick={() => handleDelete(p.id)}
                  className="text-xs text-zinc-700 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-all"
                >
                  delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MiniGrid({ grid }: { grid: number[][] }) {
  return (
    <div
      className="inline-grid mx-auto"
      style={{
        gridTemplateColumns: `repeat(8, 10px)`,
        gap: "1px",
      }}
    >
      {grid.map((row, r) =>
        row.map((cell, c) => (
          <div
            key={`${r}-${c}`}
            className={`rounded-sm ${
              cell === 1 ? "bg-zinc-200" : "bg-zinc-800"
            }`}
            style={{ width: 10, height: 10 }}
          />
        ))
      )}
    </div>
  );
}
