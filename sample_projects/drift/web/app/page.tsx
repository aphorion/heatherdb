"use client";

import { useState, useCallback } from "react";
import { api, type ChainStep, type LandscapePoint, type TrajectoryPoint } from "@/lib/api";
import ChainViz from "@/components/ChainViz";
import Landscape from "@/components/Landscape";
import Controls from "@/components/Controls";

export default function Home() {
  const [stimulus, setStimulus] = useState("");
  const [chain, setChain] = useState<ChainStep[]>([]);
  const [landscape, setLandscape] = useState<{
    concepts: LandscapePoint[];
    trajectory: TrajectoryPoint[];
  }>({ concepts: [], trajectory: [] });
  const [stats, setStats] = useState({ concepts: 0, pairs_written: 0 });
  const [params, setParams] = useState({
    alpha: 0.7,
    beta: 0.2,
    gamma: 0.1,
    steps: 12,
  });
  const [loading, setLoading] = useState(false);
  const [seeding, setSeeding] = useState(false);
  const [seedText, setSeedText] = useState("");
  const [lastStimulus, setLastStimulus] = useState("");
  const [showSeed, setShowSeed] = useState(false);

  const handleThink = useCallback(async () => {
    if (!stimulus.trim() || loading) return;
    setLoading(true);
    try {
      const res = await api.think(stimulus.trim(), params);
      setChain(res.chain);
      setLandscape(res.landscape);
      setStats(res.stats);
      setLastStimulus(stimulus.trim());
    } finally {
      setLoading(false);
    }
  }, [stimulus, params, loading]);

  const handleSeed = useCallback(async () => {
    if (!seedText.trim() || seeding) return;
    setSeeding(true);
    try {
      const res = await api.seed(seedText.trim());
      setStats({ concepts: res.concepts, pairs_written: res.pairs_written });
      setSeedText("");
      setShowSeed(false);
    } finally {
      setSeeding(false);
    }
  }, [seedText, seeding]);

  const handleLoadStarter = useCallback(async () => {
    setSeeding(true);
    try {
      const res = await api.loadStarter();
      setStats({ concepts: res.concepts, pairs_written: res.pairs_written });
    } finally {
      setSeeding(false);
    }
  }, []);

  const handleReset = useCallback(async () => {
    await api.reset();
    setChain([]);
    setLandscape({ concepts: [], trajectory: [] });
    setStats({ concepts: 0, pairs_written: 0 });
    setLastStimulus("");
  }, []);

  return (
    <div className="min-h-screen">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* header */}
        <div className="mb-8">
          <h1 className="text-xl font-bold tracking-tight">Drift</h1>
          <p className="text-xs text-zinc-600 mt-0.5">
            Chained EAM reconstruction. Give it a word — watch the thought
            traverse the interference landscape.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-8">
          {/* main column */}
          <div className="space-y-6">
            {/* stimulus input */}
            <div className="flex gap-2">
              <input
                value={stimulus}
                onChange={(e) => setStimulus(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleThink()}
                placeholder="Enter a stimulus word..."
                className="flex-1 px-4 py-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-sm font-[family-name:var(--font-geist-mono)] text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
              />
              <button
                onClick={handleThink}
                disabled={!stimulus.trim() || loading}
                className="px-5 py-2.5 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "thinking..." : "think"}
              </button>
            </div>

            {/* empty state */}
            {chain.length === 0 && stats.concepts === 0 && (
              <div className="py-16 text-center space-y-4">
                <p className="text-sm text-zinc-500">
                  The EAM is empty. Seed it with knowledge first.
                </p>
                <button
                  onClick={handleLoadStarter}
                  disabled={seeding}
                  className="px-5 py-2 bg-violet-500/10 border border-violet-500/30 text-violet-300 text-sm rounded-lg hover:bg-violet-500/20 transition-colors disabled:opacity-40"
                >
                  {seeding ? "loading..." : "Load starter knowledge"}
                </button>
              </div>
            )}

            {chain.length === 0 && stats.concepts > 0 && (
              <div className="py-16 text-center">
                <p className="text-sm text-zinc-500">
                  {stats.concepts} concepts loaded. Enter a stimulus word to
                  start a thought chain.
                </p>
              </div>
            )}

            {/* chain visualization */}
            {chain.length > 0 && (
              <ChainViz chain={chain} stimulus={lastStimulus} />
            )}

            {/* landscape */}
            {landscape.concepts.length > 0 && (
              <div className="border-t border-zinc-800/50 pt-6">
                <p className="text-xs text-zinc-600 mb-3">
                  Concept space (PCA projection). White = stimulus. Purple =
                  thought trajectory.
                </p>
                <Landscape
                  concepts={landscape.concepts}
                  trajectory={landscape.trajectory}
                  stimulus={lastStimulus}
                />
              </div>
            )}
          </div>

          {/* sidebar */}
          <div className="space-y-6">
            {/* parameters */}
            <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-4">
              <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                Chain parameters
              </p>
              <Controls
                {...params}
                onChange={setParams}
              />
            </div>

            {/* stats */}
            <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-2">
              <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                Memory
              </p>
              <div className="flex justify-between text-xs">
                <span className="text-zinc-500">concepts</span>
                <span className="text-zinc-300 font-[family-name:var(--font-geist-mono)]">
                  {stats.concepts}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-zinc-500">patterns</span>
                <span className="text-zinc-300 font-[family-name:var(--font-geist-mono)]">
                  {stats.pairs_written}
                </span>
              </div>
            </div>

            {/* seed knowledge */}
            <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                  Add knowledge
                </p>
                <button
                  onClick={() => setShowSeed(!showSeed)}
                  className="text-xs text-zinc-600 hover:text-zinc-400"
                >
                  {showSeed ? "hide" : "show"}
                </button>
              </div>

              {showSeed && (
                <>
                  <textarea
                    value={seedText}
                    onChange={(e) => setSeedText(e.target.value)}
                    placeholder="Type or paste knowledge..."
                    className="w-full h-24 bg-zinc-900 border border-zinc-800 rounded text-xs text-zinc-300 p-2 placeholder:text-zinc-700 resize-none focus:outline-none focus:border-zinc-600"
                  />
                  <button
                    onClick={handleSeed}
                    disabled={!seedText.trim() || seeding}
                    className="w-full py-1.5 text-xs bg-zinc-800 text-zinc-300 rounded hover:bg-zinc-700 disabled:opacity-40 transition-colors"
                  >
                    {seeding ? "writing..." : "seed"}
                  </button>
                </>
              )}

              {stats.concepts === 0 && (
                <button
                  onClick={handleLoadStarter}
                  disabled={seeding}
                  className="w-full py-1.5 text-xs bg-violet-500/10 text-violet-300 rounded hover:bg-violet-500/20 disabled:opacity-40 transition-colors"
                >
                  {seeding ? "loading..." : "load starter"}
                </button>
              )}
            </div>

            {/* actions */}
            {stats.concepts > 0 && (
              <button
                onClick={handleReset}
                className="text-xs text-zinc-700 hover:text-rose-400 transition-colors"
              >
                reset everything
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
