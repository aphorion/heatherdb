"use client";

import { useEffect, useState } from "react";
import type { ChainStep } from "@/lib/api";

interface ChainVizProps {
  chain: ChainStep[];
  stimulus: string;
}

export default function ChainViz({ chain, stimulus }: ChainVizProps) {
  const [visibleSteps, setVisibleSteps] = useState(0);

  // animate steps appearing one by one
  useEffect(() => {
    setVisibleSteps(0);
    if (chain.length === 0) return;

    let step = 0;
    const timer = setInterval(() => {
      step++;
      setVisibleSteps(step);
      if (step >= chain.length) clearInterval(timer);
    }, 400);

    return () => clearInterval(timer);
  }, [chain]);

  if (chain.length === 0) return null;

  return (
    <div className="space-y-1">
      {/* stimulus label */}
      <div className="flex items-center gap-3 mb-4">
        <div className="px-3 py-1.5 bg-zinc-100 text-zinc-900 text-sm font-bold rounded-md font-[family-name:var(--font-geist-mono)]">
          {stimulus}
        </div>
        <div className="text-zinc-600 text-xs">stimulus</div>
      </div>

      {/* chain steps */}
      <div className="space-y-0">
        {chain.map((step, i) => {
          if (i >= visibleSteps) return null;

          const isLast = i === chain.length - 1;
          const driftWidth = Math.max(2, Math.min(40, step.drift * 100));

          return (
            <div key={step.step} className="animate-in">
              {/* connector line */}
              <div className="flex items-center ml-6 my-0">
                <div
                  className="border-l-2 border-zinc-700 h-6"
                  style={{
                    opacity: Math.max(0.2, 1 - step.drift),
                  }}
                />
                <span className="text-[10px] text-zinc-700 ml-2 font-[family-name:var(--font-geist-mono)]">
                  drift {Math.round(step.drift * 100)}%
                </span>
              </div>

              {/* concept node */}
              <div className="flex items-center gap-3">
                <div
                  className={`px-3 py-1.5 text-sm font-[family-name:var(--font-geist-mono)] rounded-md border transition-all duration-300 ${
                    step.converged
                      ? "bg-violet-500/20 border-violet-500 text-violet-300 shadow-[0_0_12px_rgba(139,92,246,0.3)]"
                      : "bg-zinc-900 border-zinc-700 text-zinc-200"
                  }`}
                >
                  {step.concept}
                </div>

                <span className="text-[10px] text-zinc-600 font-[family-name:var(--font-geist-mono)]">
                  {Math.round(step.similarity * 100)}% match
                </span>

                {/* nearby concepts */}
                {step.nearby.length > 0 && (
                  <div className="flex gap-1.5">
                    {step.nearby.map((n) => (
                      <span
                        key={n.concept}
                        className="text-[10px] text-zinc-700 font-[family-name:var(--font-geist-mono)]"
                      >
                        {n.concept}
                      </span>
                    ))}
                  </div>
                )}

                {step.converged && (
                  <span className="text-[10px] text-violet-400 font-medium">
                    settled
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* summary */}
      {visibleSteps >= chain.length && (
        <div className="mt-6 pt-4 border-t border-zinc-800/50 animate-in">
          <p className="text-xs text-zinc-500">
            {chain.length} steps.{" "}
            {chain[chain.length - 1]?.converged
              ? `Settled into "${chain[chain.length - 1].concept}" — this is the attractor the thought converged to.`
              : `Reached max steps without converging — the thought is still drifting.`}
            {" "}Distance from stimulus:{" "}
            <span className="text-zinc-400 font-[family-name:var(--font-geist-mono)]">
              {Math.round(chain[chain.length - 1].distance_from_start * 100)}%
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
