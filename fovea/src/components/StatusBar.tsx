import type { AppCtx } from "@/App";
import { useEffect, useState } from "react";

export default function StatusBar({ ctx }: { ctx: AppCtx }) {
  const [stats, setStats] = useState<{ locs: number; writes: number; cols: number } | null>(null);

  useEffect(() => {
    if (!ctx.client) { setStats(null); return; }
    let cancelled = false;
    const tick = () => {
      ctx.client!.stats()
        .then((s) => {
          if (cancelled) return;
          setStats({ locs: s.total_locations, writes: s.total_writes, cols: s.collections.length });
        })
        .catch(() => !cancelled && setStats(null));
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { cancelled = true; clearInterval(id); };
  }, [ctx.client]);

  return (
    <footer className="h-7 shrink-0 px-4 border-t border-ink-line bg-surface-1 flex items-center justify-between text-10 font-mono uppercase tracking-ops text-ink-ghost">
      <div className="flex items-center gap-4">
        {ctx.serverOk === null && <span>connecting…</span>}
        {ctx.serverOk === false && <span className="text-accent-alarm">server unreachable</span>}
        {ctx.serverOk === true && stats && (
          <>
            <span>{stats.cols.toLocaleString()} <span className="text-ink-ghost/70">collections</span></span>
            <span className="text-ink-ghost/40">|</span>
            <span>{stats.locs.toLocaleString()} <span className="text-ink-ghost/70">attractors</span></span>
            <span className="text-ink-ghost/40">|</span>
            <span>{stats.writes.toLocaleString()} <span className="text-ink-ghost/70">writes</span></span>
          </>
        )}
      </div>
      <div>fovea — heatherdb operator instrument</div>
    </footer>
  );
}
