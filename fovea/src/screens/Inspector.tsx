import { useEffect, useMemo, useState } from "react";
import type { AppCtx } from "@/App";
import type {
  ProjectionPoint, ActivatedLocation, CollectionSummary,
} from "@/lib/heather";
import AttractorMap from "@/components/AttractorMap";
import { fmtCount, fmtMs } from "@/lib/format";

type Tab = "landscape" | "stats";

export default function Inspector({ ctx, collection }: { ctx: AppCtx; collection: string }) {
  const [tab, setTab] = useState<Tab>("landscape");
  const [points, setPoints] = useState<ProjectionPoint[] | null>(null);
  const [synthetic, setSynthetic] = useState(false);
  const [summary, setSummary] = useState<CollectionSummary | null>(null);
  const [activations, setActivations] = useState<ActivatedLocation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Load summary always; try /projection but fall back to a deterministic
  // synthetic landscape (golden-spiral) if the server doesn't expose it.
  useEffect(() => {
    if (!ctx.client) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setActivations([]);
    setSynthetic(false);

    const loadCols = ctx.client.collections().then((cols) => {
      if (cancelled) return null;
      const sum = cols.collections.find((c) => c.name === collection) ?? null;
      setSummary(sum);
      return sum;
    });

    const loadProj = ctx.client.projection(collection).catch(() => null);

    Promise.all([loadCols, loadProj])
      .then(([sum, proj]) => {
        if (cancelled) return;
        if (proj && proj.points.length > 0) {
          setPoints(proj.points);
        } else {
          // Fallback: synthesise a placeholder spiral sized to the collection.
          // Real positions need /projection (only the proxy ships it today).
          const n = Math.min(800, sum?.num_locations ?? 200);
          setPoints(syntheticLandscape(collection, n));
          setSynthetic(true);
        }
      })
      .catch((e) => !cancelled && setError(String(e?.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [ctx.client, collection]);

  // Probe a fingerprint analyze to highlight the natural attractors —
  // sends the collection's own fingerprint as a query, the activations
  // it returns are the dominant attractors of the population.
  const probeFingerprint = async () => {
    if (!ctx.client) return;
    try {
      const fp = await ctx.client.fingerprint(collection);
      const r = await ctx.client.analyze(collection, fp.fingerprint);
      setActivations(r.activated_locations);
    } catch (e) {
      setError(String((e as Error).message));
    }
  };

  const top = useMemo(() => {
    return [...activations].sort((a, b) => b.weight - a.weight).slice(0, 8);
  }, [activations]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-ink-line flex items-center justify-between gap-4">
        <div className="min-w-0 flex items-baseline gap-3">
          <button
            onClick={() => ctx.navigate({ kind: "collections" })}
            className="font-mono text-10 uppercase tracking-ops text-ink-muted hover:text-white"
          >
            ← collections
          </button>
          <span className="text-ink-ghost">/</span>
          <h1 className="text-[18px] truncate font-medium">{collection}</h1>
          {summary && (
            <span className="font-mono text-10 uppercase tracking-ops text-ink-muted">
              {fmtCount(summary.num_locations)} attractors · {fmtCount(summary.total_writes)} writes
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={probeFingerprint} className="btn">probe fingerprint</button>
          <button
            onClick={() => ctx.navigate({ kind: "read", collection })}
            className="btn"
          >
            open in read →
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="px-6 border-b border-ink-line flex items-center gap-4">
        <Tab active={tab === "landscape"} onClick={() => setTab("landscape")}>Landscape</Tab>
        <Tab active={tab === "stats"}     onClick={() => setTab("stats")}>Stats</Tab>
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 flex">
        {tab === "landscape" && (
          <>
            <div className="flex-1 min-w-0 relative">
              {error && <ErrorBanner msg={error} />}
              {loading && !points && <CenterMsg>Loading projection…</CenterMsg>}
              {points && points.length === 0 && <CenterMsg>Empty collection — no attractors yet.</CenterMsg>}
              {points && points.length > 0 && (
                <>
                  <AttractorMap points={points} activations={activations} />
                  {synthetic && (
                    <div className="absolute bottom-2 right-2 font-mono text-10 uppercase tracking-ops text-accent-warm bg-black/80 border border-accent-warm/30 px-2 py-1">
                      synthetic landscape · server lacks /projection
                    </div>
                  )}
                </>
              )}
            </div>
            <aside className="w-[300px] shrink-0 border-l border-ink-line p-4 flex flex-col gap-4 overflow-y-auto">
              <SidePanel title="Top activations">
                {activations.length === 0 ? (
                  <p className="text-ink-muted text-[12px] leading-relaxed">
                    Hit <code className="text-white">probe fingerprint</code> to fire the
                    collection's own centroid through <code className="text-white">/analyze</code>.
                    The dominant attractors will glow in the landscape and list here.
                  </p>
                ) : (
                  <ul className="space-y-1.5">
                    {top.map((a) => (
                      <li key={a.id} className="flex items-baseline justify-between gap-3">
                        <span className="font-mono text-11 text-white">#{a.id}</span>
                        <Bar value={a.weight} />
                        <span className="font-mono text-11 text-ink-dim w-12 text-right">
                          {a.weight.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </SidePanel>
              <SidePanel title="Hint">
                <p className="text-ink-muted text-[12px] leading-relaxed">
                  Drag to pan · scroll to zoom · hover a dot to read its id
                  and weight. Activated cells get a colored halo.
                </p>
              </SidePanel>
            </aside>
          </>
        )}

        {tab === "stats" && (
          <div className="flex-1 min-w-0 p-6 space-y-4 overflow-y-auto">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="attractors"    value={fmtCount(summary?.num_locations)} />
              <Stat label="writes"        value={fmtCount(summary?.total_writes)} />
              <Stat label="dimension"     value={summary?.dimension?.toString() ?? "—"} />
              <Stat label="avg writes/loc"
                    value={
                      summary && summary.num_locations
                        ? (summary.total_writes / summary.num_locations).toFixed(1)
                        : "—"
                    } />
            </div>
            <SidePanel title="Health">
              <p className="text-ink-muted text-[12px]">
                Server: <span className="text-white">{ctx.connection?.url}</span>
              </p>
              <Latencies ctx={ctx} collection={collection} />
            </SidePanel>
          </div>
        )}
      </div>
    </div>
  );
}

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`relative py-2.5 font-mono text-11 uppercase tracking-ops transition-colors
        ${active ? "text-white" : "text-ink-muted hover:text-white"}`}
    >
      {children}
      {active && <span className="absolute bottom-0 left-0 right-0 h-px bg-white" />}
    </button>
  );
}

function SidePanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel p-3">
      <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-2">{title}</div>
      {children}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-1">{label}</div>
      <div className="text-2xl font-mono">{value}</div>
    </div>
  );
}

function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div className="flex-1 h-1 bg-white/[0.06] relative">
      <div
        className="absolute top-0 left-0 h-full"
        style={{ width: `${pct * 100}%`, background: "#a855f7", boxShadow: "0 0 6px -2px #a855f7" }}
      />
    </div>
  );
}

function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="absolute top-3 left-3 right-3 border border-accent-alarm/30 bg-accent-alarm/[0.06] text-accent-alarm font-mono text-11 uppercase tracking-ops px-3 py-2">
      {msg}
    </div>
  );
}

function CenterMsg({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 grid place-items-center text-ink-muted text-[14px] font-light">
      {children}
    </div>
  );
}

/**
 * Deterministic golden-spiral placeholder when the server doesn't expose
 * /projection. Activations from /analyze still mean nothing geometrically
 * here, but the landscape still feels alive and the operator gets a sense
 * of the population scale. Real positions land when the engine grows a
 * /projection endpoint (or you run the FastAPI proxy in heatherdb-pi-demo).
 */
function syntheticLandscape(seed: string, n: number): ProjectionPoint[] {
  const hash = Array.from(seed).reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7) || 1;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const offset = (hash % 1000) * 0.001;
  return Array.from({ length: n }, (_, i) => {
    const r = Math.sqrt((i + 0.5) / n);
    const a = i * golden + offset;
    return {
      id: i,
      x: Math.cos(a) * r,
      y: Math.sin(a) * r,
      weight: 0.3 + 0.7 * (1 - r),
    };
  });
}

function Latencies({ ctx, collection }: { ctx: AppCtx; collection: string }) {
  const [last, setLast] = useState<{ analyze_ms: number; read_ms: number } | null>(null);
  return (
    <div className="mt-2">
      <button
        className="btn"
        onClick={async () => {
          if (!ctx.client) return;
          const fp = await ctx.client.fingerprint(collection);
          const t1 = performance.now();
          await ctx.client.read(collection, fp.fingerprint);
          const read_ms = performance.now() - t1;
          const t2 = performance.now();
          await ctx.client.analyze(collection, fp.fingerprint);
          const analyze_ms = performance.now() - t2;
          setLast({ read_ms, analyze_ms });
        }}
      >
        Sample latency
      </button>
      {last && (
        <div className="font-mono text-11 text-ink-muted mt-3 space-y-1">
          <div>read    {fmtMs(last.read_ms)}</div>
          <div>analyze {fmtMs(last.analyze_ms)}</div>
        </div>
      )}
    </div>
  );
}
