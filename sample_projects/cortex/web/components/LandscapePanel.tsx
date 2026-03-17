"use client";

import { useState } from "react";
import { api, type Basin } from "@/lib/api";
import FidelityBar from "./FidelityBar";

export default function LandscapePanel() {
  const [loading, setLoading] = useState(false);
  const [basins, setBasins] = useState<Basin[]>([]);
  const [probes, setProbes] = useState(20);

  async function handleMap() {
    setLoading(true);
    try {
      const res = await api.landscape(probes);
      setBasins(res.basins);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Landscape</h2>
        <p className="text-sm text-zinc-500">
          Map the attractor basins of your notebook. See what it knows most about.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleMap}
          disabled={loading}
          className="px-5 py-2.5 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Mapping..." : "Map landscape"}
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500">Probes</label>
          <input
            type="number"
            value={probes}
            onChange={(e) =>
              setProbes(Math.max(5, Math.min(50, parseInt(e.target.value) || 20)))
            }
            min={5}
            max={50}
            className="w-16 px-2 py-1.5 bg-zinc-900 border border-zinc-800 rounded text-sm text-zinc-300 text-center focus:outline-none focus:border-zinc-600"
          />
        </div>
      </div>

      {basins.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 animate-in fade-in duration-300">
          {basins.map((basin, i) => (
            <div
              key={i}
              className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-sm font-medium text-zinc-200">{basin.label}</h3>
                <span className="text-xs font-mono text-zinc-500 shrink-0 ml-2">
                  {basin.notes.length} note{basin.notes.length !== 1 ? "s" : ""}
                </span>
              </div>
              <FidelityBar value={basin.fidelity} label="Basin depth" size="sm" />
              <div className="mt-3 space-y-1.5">
                {basin.notes.slice(0, 3).map((note) => (
                  <p
                    key={note.id}
                    className="text-xs text-zinc-500 leading-relaxed truncate"
                    title={note.text}
                  >
                    {note.text}
                  </p>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
