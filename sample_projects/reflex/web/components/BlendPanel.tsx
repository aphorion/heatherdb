"use client";

import { useState, useEffect } from "react";
import Grid from "./Grid";
import { api, type Pattern, type BlendResponse } from "@/lib/api";

export default function BlendPanel() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BlendResponse | null>(null);

  useEffect(() => {
    api.patterns().then((r) => setPatterns(r.patterns));
  }, []);

  function toggleSelect(id: number) {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
    setResult(null);
  }

  async function handleBlend() {
    if (selected.size < 2 || loading) return;
    setLoading(true);
    try {
      const res = await api.blend(Array.from(selected));
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  const fidelityPct = result ? Math.round(result.fidelity * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Blend</h2>
        <p className="text-sm text-zinc-500">
          Select 2+ patterns. The EAM shows what lives between them — the
          interference pattern where memories overlap.
        </p>
      </div>

      {patterns.length === 0 ? (
        <p className="text-sm text-zinc-600 italic">
          No patterns stored yet. Draw some first.
        </p>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {patterns.map((p) => (
              <button
                key={p.id}
                onClick={() => toggleSelect(p.id)}
                className={`p-2 rounded-lg border transition-colors ${
                  selected.has(p.id)
                    ? "border-violet-500 bg-violet-500/10"
                    : "border-zinc-800 hover:border-zinc-600"
                }`}
              >
                <MiniGrid grid={p.grid} />
                <p className="text-[10px] text-zinc-500 mt-1 truncate">
                  {p.name || `#${p.id}`}
                </p>
              </button>
            ))}
          </div>

          <div className="flex justify-center">
            <button
              onClick={handleBlend}
              disabled={selected.size < 2 || loading}
              className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Blending..." : `Blend ${selected.size} patterns`}
            </button>
          </div>
        </>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in duration-300">
          <div className="flex justify-center">
            <Grid
              grid={result.grid}
              readOnly
              confidence={result.confidence}
            />
          </div>
          <p className="text-center text-sm text-zinc-400">
            Blend fidelity:{" "}
            <span className="font-mono font-bold text-violet-400">
              {fidelityPct}%
            </span>
          </p>
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
        gridTemplateColumns: `repeat(8, 8px)`,
        gap: "1px",
      }}
    >
      {grid.map((row, r) =>
        row.map((cell, c) => (
          <div
            key={`${r}-${c}`}
            className={`rounded-sm ${cell === 1 ? "bg-zinc-200" : "bg-zinc-800"}`}
            style={{ width: 8, height: 8 }}
          />
        ))
      )}
    </div>
  );
}
