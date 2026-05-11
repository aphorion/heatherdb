import { useEffect, useState } from "react";
import type { AppCtx } from "@/App";
import type { CollectionSummary } from "@/lib/heather";
import { fmtCount } from "@/lib/format";

export default function Collections({ ctx }: { ctx: AppCtx }) {
  const [list, setList] = useState<CollectionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (!ctx.client) return;
    let cancelled = false;
    ctx.client.collections()
      .then((r) => !cancelled && setList(r.collections))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)));
    return () => { cancelled = true; };
  }, [ctx.client]);

  const visible = (list ?? []).filter((c) => c.name.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="px-8 py-8">
      <header className="mb-6 flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="kicker mb-2">Collections</div>
          <h1 className="text-2xl font-normal tracking-tight">
            {list?.length ?? "—"} <span className="text-ink-muted">memory shapes</span>
          </h1>
        </div>
        <input
          className="input w-64"
          placeholder="filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </header>

      {error && (
        <div className="border border-accent-alarm/30 bg-accent-alarm/[0.06] text-accent-alarm font-mono text-11 uppercase tracking-ops px-4 py-3 mb-6">
          {error}
        </div>
      )}

      {!list && !error && (
        <Skeleton />
      )}

      {list && visible.length === 0 && (
        <div className="border border-dashed border-ink-line px-6 py-16 text-center text-ink-muted text-[14px]">
          No collections {filter ? "match the filter" : "exist on this server yet"}.
        </div>
      )}

      <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
        {visible.map((c) => (
          <button
            key={c.name}
            onClick={() => ctx.navigate({ kind: "inspector", collection: c.name })}
            className="text-left panel hover:border-white/30 transition-colors group"
          >
            {/* Tile preview — a tiny synthetic landscape so the card doesn't feel empty.
                Real preview swap happens after a /projection roundtrip in the future. */}
            <div className="aspect-[4/3] relative overflow-hidden">
              <TilePreview seed={c.name} accent="#a855f7" />
            </div>
            <div className="px-3 py-2.5 border-t border-ink-line">
              <div className="text-[13px] text-white truncate font-medium">{c.name}</div>
              <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mt-0.5">
                {fmtCount(c.num_locations)} attractors
                <span className="text-ink-ghost"> · </span>
                {fmtCount(c.total_writes)} writes
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="panel">
          <div className="aspect-[4/3] bg-white/[0.02] animate-pulse" />
          <div className="px-3 py-2.5 border-t border-ink-line space-y-1.5">
            <div className="h-3 bg-white/[0.04] w-3/4" />
            <div className="h-2 bg-white/[0.03] w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}

/* Cheap deterministic preview — quasi-spiral derived from the name hash.
   Replaced with real /projection data on demand from the inspector. */
function TilePreview({ seed, accent }: { seed: string; accent: string }) {
  const hash = Array.from(seed).reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7) || 1;
  const N = 60;
  const points = Array.from({ length: N }, (_, i) => {
    const r = 0.15 + 0.7 * ((i * 0.617 + hash * 0.0001) % 1);
    const a = i * (Math.PI * (3 - Math.sqrt(5))) + hash * 0.0017;
    return { cx: 50 + Math.cos(a) * r * 38, cy: 50 + Math.sin(a) * r * 28 };
  });
  return (
    <svg viewBox="0 0 100 60" preserveAspectRatio="none" className="w-full h-full">
      <defs>
        <radialGradient id={`g-${hash}`} cx="50%" cy="50%" r="60%">
          <stop offset="0%"  stopColor={`${accent}25`} />
          <stop offset="100%" stopColor={`${accent}00`} />
        </radialGradient>
      </defs>
      <rect width="100" height="60" fill={`url(#g-${hash})`} />
      {points.map((p, i) => (
        <circle key={i} cx={p.cx} cy={p.cy} r={i % 7 === 0 ? 1 : 0.5} fill="rgba(255,255,255,0.5)" />
      ))}
    </svg>
  );
}
