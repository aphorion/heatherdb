"use client";

import { useState } from "react";
import Grid, { emptyGrid } from "./Grid";
import { api, type Grid as GridType, type CompleteResponse } from "@/lib/api";

export default function CompletePanel() {
  const [grid, setGrid] = useState<GridType>(emptyGrid());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompleteResponse | null>(null);

  const hasContent = grid.some((row) => row.some((c) => c === 1));

  async function handleComplete() {
    if (!hasContent || loading) return;
    setLoading(true);
    try {
      const res = await api.complete(grid);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setGrid(emptyGrid());
    setResult(null);
  }

  const fidelityPct = result ? Math.round(result.fidelity * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Complete</h2>
        <p className="text-sm text-zinc-500">
          Draw part of a pattern. The EAM completes the rest from memory.
          <span className="text-violet-400"> Purple cells</span> = SDM&apos;s
          completion.
        </p>
      </div>

      <div className="flex justify-center">
        <Grid
          grid={grid}
          onChange={result ? undefined : setGrid}
          readOnly={!!result}
          overlay={result?.completed}
          confidence={result?.confidence}
        />
      </div>

      <div className="flex items-center gap-3 justify-center">
        {!result ? (
          <>
            <button
              onClick={handleComplete}
              disabled={!hasContent || loading}
              className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Completing..." : "Complete"}
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
        <div className="text-center space-y-2 animate-in fade-in duration-300">
          <p className="text-sm text-zinc-400">
            Recognition:{" "}
            <span
              className={`font-mono font-bold ${
                fidelityPct >= 60
                  ? "text-emerald-400"
                  : fidelityPct >= 30
                  ? "text-amber-400"
                  : "text-rose-400"
              }`}
            >
              {fidelityPct}%
            </span>
          </p>
          <p className="text-xs text-zinc-600">
            {fidelityPct >= 60
              ? "Strong recognition — the EAM confidently completed this pattern."
              : fidelityPct >= 30
              ? "Partial recognition — the completion is the EAM's best guess."
              : "Weak recognition — this pattern is unfamiliar to the EAM."}
          </p>
        </div>
      )}
    </div>
  );
}
