"use client";

import type { ChatResponse, RecalledMemory } from "@/lib/api";

interface MemoryPanelProps {
  lastResponse: ChatResponse | null;
}

const typeColors: Record<string, string> = {
  vivid: "border-emerald-500/50 bg-emerald-500/5",
  clear: "border-amber-500/50 bg-amber-500/5",
  vague: "border-zinc-700 bg-zinc-800/30",
};

const typeTextColors: Record<string, string> = {
  vivid: "text-emerald-400",
  clear: "text-amber-400",
  vague: "text-zinc-500",
};

export default function MemoryPanel({ lastResponse }: MemoryPanelProps) {
  if (!lastResponse) {
    return (
      <aside className="w-72 border-l border-zinc-800 p-4">
        <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-4">
          Recalled Memories
        </h2>
        <p className="text-xs text-zinc-700">
          Send a message to see what the EAM recalls.
        </p>
      </aside>
    );
  }

  const { recalled_memories, fidelity, recent_tokens, recalled_tokens } =
    lastResponse;
  const fidelityPct = Math.round(fidelity * 100);

  return (
    <aside className="w-72 border-l border-zinc-800 overflow-y-auto">
      <div className="p-4 space-y-4">
        <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
          Recalled Memories
        </h2>

        {/* fidelity */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span data-testid="fidelity-label" className="text-zinc-500">Fidelity</span>
            <span
              className={`font-mono font-bold ${
                fidelityPct >= 60
                  ? "text-emerald-400"
                  : fidelityPct >= 35
                  ? "text-amber-400"
                  : "text-zinc-500"
              }`}
            >
              {fidelityPct}%
            </span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${fidelityPct}%`,
                backgroundColor:
                  fidelityPct >= 60
                    ? "#10b981"
                    : fidelityPct >= 35
                    ? "#f59e0b"
                    : "#3f3f46",
              }}
            />
          </div>
          <p className="text-[10px] text-zinc-600">
            {fidelityPct >= 60
              ? "Strong recognition — familiar topic"
              : fidelityPct >= 35
              ? "Partial recognition"
              : "New territory — relying on recent context"}
          </p>
        </div>

        {/* token breakdown */}
        <div className="flex gap-3 text-[10px] text-zinc-600">
          <span>Recent: {recent_tokens}t</span>
          <span>Recalled: {recalled_tokens}t</span>
        </div>

        {/* memories */}
        {recalled_memories.length === 0 ? (
          <p className="text-xs text-zinc-700">No memories recalled.</p>
        ) : (
          <div className="space-y-2">
            {recalled_memories.map((mem, i) => (
              <MemoryCard key={`${mem.id}-${i}`} memory={mem} />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

function MemoryCard({ memory }: { memory: RecalledMemory }) {
  return (
    <div
      className={`p-2.5 rounded-md border ${typeColors[memory.memory_type]}`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span
          className={`text-[10px] font-mono ${typeTextColors[memory.memory_type]}`}
        >
          {memory.memory_type}
        </span>
        <span className="text-[10px] font-mono text-zinc-600">
          {Math.round(memory.similarity * 100)}%
        </span>
      </div>
      <p className="text-xs text-zinc-400 line-clamp-3">{memory.content}</p>
      <p className="text-[10px] text-zinc-700 mt-1">{memory.role}</p>
    </div>
  );
}
