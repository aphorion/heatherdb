"use client";

import { useState } from "react";
import Grid, { emptyGrid } from "./Grid";
import { api, type Grid as GridType } from "@/lib/api";

interface DrawPanelProps {
  onPatternWritten?: () => void;
}

export default function DrawPanel({ onPatternWritten }: DrawPanelProps) {
  const [grid, setGrid] = useState<GridType>(emptyGrid());
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ id: number; fidelity: number } | null>(
    null
  );

  const hasContent = grid.some((row) => row.some((c) => c === 1));

  async function handleWrite() {
    if (!hasContent || loading) return;
    setLoading(true);
    try {
      const res = await api.write(grid, name || `Pattern`);
      setResult(res);
      onPatternWritten?.();
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
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Draw</h2>
        <p className="text-sm text-zinc-500">
          Draw a pattern and write it to the EAM. Click cells to toggle. Drag to
          paint.
        </p>
      </div>

      <div className="flex justify-center">
        <Grid grid={grid} onChange={setGrid} />
      </div>

      <div className="flex items-center gap-3 justify-center">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Pattern name (optional)"
          className="px-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 w-48"
        />
        <button
          onClick={handleWrite}
          disabled={!hasContent || loading}
          className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Writing..." : "Write to SDM"}
        </button>
        <button
          onClick={handleClear}
          className="px-4 py-2 text-sm text-zinc-500 border border-zinc-800 rounded-lg hover:border-zinc-600 hover:text-zinc-300 transition-colors"
        >
          Clear
        </button>
      </div>

      {result && (
        <div className="text-center animate-in fade-in duration-300">
          <p className="text-sm text-zinc-400">
            Pattern #{result.id} written.{" "}
            <span
              className={
                fidelityPct >= 70
                  ? "text-emerald-400"
                  : fidelityPct >= 40
                  ? "text-amber-400"
                  : "text-rose-400"
              }
            >
              {fidelityPct}% familiar
            </span>{" "}
            to the EAM.
          </p>
        </div>
      )}
    </div>
  );
}
