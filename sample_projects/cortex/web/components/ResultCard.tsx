"use client";

import type { NoteResult } from "@/lib/api";

interface ResultCardProps {
  result: NoteResult;
  highlight?: boolean;
}

export default function ResultCard({ result, highlight }: ResultCardProps) {
  const pct = Math.round(result.similarity * 100);

  return (
    <div
      className={`group relative p-3.5 rounded-lg border transition-all duration-200 ${
        highlight
          ? "border-violet-500/40 bg-violet-500/5 hover:border-violet-400/60"
          : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
      }`}
    >
      {highlight && (
        <div className="absolute -top-2 right-3 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider bg-violet-500/20 text-violet-400 rounded border border-violet-500/30">
          insight
        </div>
      )}
      <p className="text-sm text-zinc-200 leading-relaxed mb-2">{result.text}</p>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-1 rounded-full transition-all duration-500 ${
              highlight ? "bg-violet-500" : "bg-zinc-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-[11px] font-mono text-zinc-500">{pct}%</span>
      </div>
    </div>
  );
}
