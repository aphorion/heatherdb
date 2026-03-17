"use client";

import { useState } from "react";
import Grid, { emptyGrid } from "./Grid";
import { api, type Grid as GridType } from "@/lib/api";

export default function SurprisePanel() {
  const [grid, setGrid] = useState<GridType>(emptyGrid());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    novelty: number;
    fidelity: number;
  } | null>(null);

  const hasContent = grid.some((row) => row.some((c) => c === 1));

  async function handleSurprise() {
    if (!hasContent || loading) return;
    setLoading(true);
    try {
      const res = await api.surprise(grid);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setGrid(emptyGrid());
    setResult(null);
  }

  const noveltyPct = result ? Math.round(result.novelty * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Surprise</h2>
        <p className="text-sm text-zinc-500">
          Draw a pattern. The EAM tells you how novel it is — high surprise
          means the engine has never seen anything like it.
        </p>
      </div>

      <div className="flex justify-center">
        <Grid grid={grid} onChange={result ? undefined : setGrid} readOnly={!!result} />
      </div>

      <div className="flex items-center gap-3 justify-center">
        {!result ? (
          <>
            <button
              onClick={handleSurprise}
              disabled={!hasContent || loading}
              className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Analyzing..." : "How surprising?"}
            </button>
            <button
              onClick={handleClear}
              className="px-4 py-2 text-sm text-zinc-500 border border-zinc-800 rounded-lg hover:border-zinc-600 hover:text-zinc-300 transition-colors"
            >
              Clear
            </button>
          </>
        ) : (
          <button
            onClick={handleClear}
            className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white transition-colors"
          >
            Try another
          </button>
        )}
      </div>

      {result && (
        <div className="text-center space-y-3 animate-in fade-in duration-300">
          <div className="relative w-48 mx-auto">
            <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${noveltyPct}%`,
                  backgroundColor:
                    noveltyPct >= 70
                      ? "#f43f5e"
                      : noveltyPct >= 40
                      ? "#f59e0b"
                      : "#10b981",
                }}
              />
            </div>
          </div>
          <p className="text-sm text-zinc-400">
            Novelty:{" "}
            <span
              className={`font-mono font-bold ${
                noveltyPct >= 70
                  ? "text-rose-400"
                  : noveltyPct >= 40
                  ? "text-amber-400"
                  : "text-emerald-400"
              }`}
            >
              {noveltyPct}%
            </span>
          </p>
          <p className="text-xs text-zinc-600">
            {noveltyPct >= 70
              ? "Very surprising — the EAM hasn't seen anything like this."
              : noveltyPct >= 40
              ? "Somewhat novel — partial overlap with stored patterns."
              : "Familiar — the EAM recognizes this pattern well."}
          </p>
        </div>
      )}
    </div>
  );
}
