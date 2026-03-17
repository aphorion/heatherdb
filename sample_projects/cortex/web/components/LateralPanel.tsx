"use client";

import { useState, useEffect } from "react";
import { api, type LateralResponse, type LateralInsight } from "@/lib/api";
import FidelityBar from "./FidelityBar";
import ResultCard from "./ResultCard";

function InsightCard({ insight }: { insight: LateralInsight }) {
  return (
    <div className="p-4 rounded-lg border border-violet-500/30 bg-violet-500/5 space-y-3">
      {/* The bridge — the actionable translation */}
      {insight.bridge && (
        <p className="text-sm text-zinc-200 leading-relaxed">
          {insight.bridge}
        </p>
      )}
      {/* The source lateral note — smaller, shows origin */}
      <div className="flex items-start gap-2 pt-2 border-t border-violet-500/10">
        <div className="w-1 h-full min-h-[16px] bg-violet-500/30 rounded-full shrink-0 mt-0.5" />
        <div>
          <p className="text-xs text-zinc-500 leading-relaxed">
            {insight.source.text}
          </p>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-[10px] text-violet-500/60 uppercase tracking-wider">
              structural analog
            </span>
            <span className="text-[10px] font-mono text-zinc-600">
              {Math.round(insight.source.similarity * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function LateralPanel() {
  const [problem, setProblem] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LateralResponse | null>(null);
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [showDirect, setShowDirect] = useState(false);

  useEffect(() => {
    api.llmStatus().then((s) => setLlmEnabled(s.enabled)).catch(() => {});
  }, []);

  async function handleLateral() {
    if (!problem.trim() || loading) return;
    setLoading(true);
    setShowDirect(false);
    try {
      const res = await api.lateral(problem.trim());
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  const hasLlm = result?.llm_answer != null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">
          Lateral
        </h2>
        <p className="text-sm text-zinc-500">
          Describe a problem. The EAM finds structural analogs from unrelated
          domains, then translates them into actionable ideas for your problem.
        </p>
      </div>

      <div className="space-y-3">
        <textarea
          value={problem}
          onChange={(e) => setProblem(e.target.value)}
          placeholder="Describe a problem or challenge..."
          className="w-full h-28 px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 resize-none transition-colors"
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.metaKey) handleLateral();
          }}
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleLateral}
            disabled={!problem.trim() || loading}
            className="px-5 py-2.5 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Thinking laterally..." : "Think sideways"}
          </button>
          {!llmEnabled && (
            <span className="text-[11px] text-zinc-600">
              Set ANTHROPIC_API_KEY for bridge translations
            </span>
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Fidelity */}
          <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg">
            <FidelityBar
              value={result.fidelity}
              label="How deeply the engine resonates"
              size="md"
            />
          </div>

          {/* The obvious path — LLM answer */}
          {hasLlm && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-zinc-500" />
                <h3 className="text-sm font-medium text-zinc-400">
                  The obvious path
                </h3>
              </div>
              <div className="p-4 bg-zinc-900/30 border border-zinc-800/50 rounded-lg">
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {result.llm_answer}
                </p>
                <p className="mt-3 text-[10px] text-zinc-600 uppercase tracking-wider">
                  LLM &mdash; most probable answer
                </p>
              </div>
            </div>
          )}

          {/* The sideways path — SDM laterals with bridges */}
          {result.laterals.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-violet-500" />
                <h3 className="text-sm font-medium text-violet-400">
                  The sideways path
                </h3>
              </div>
              <p className="text-xs text-zinc-600">
                The EAM found structural analogs from other domains and
                translated them into ideas for your problem.
              </p>
              <div className="space-y-3">
                {result.laterals.map((insight, i) => (
                  <InsightCard key={insight.source.id ?? i} insight={insight} />
                ))}
              </div>
            </div>
          )}

          {result.laterals.length === 0 && (
            <div className="p-4 border border-zinc-800/50 rounded-lg">
              <p className="text-xs text-zinc-600 italic">
                No lateral associations found. The EAM recalled the same notes
                a direct search would. Try a problem that touches multiple
                domains, or add more diverse notes to the engine.
              </p>
            </div>
          )}

          {/* Direct matches — collapsed */}
          <div>
            <button
              onClick={() => setShowDirect(!showDirect)}
              className="flex items-center gap-2 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                className={`transition-transform ${showDirect ? "rotate-90" : ""}`}
              >
                <path
                  d="M4 2l4 4-4 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Direct matches ({result.direct.length})
            </button>
            {showDirect && (
              <div className="mt-3 space-y-2 opacity-60">
                {result.direct.map((r) => (
                  <ResultCard key={r.id} result={r} />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
