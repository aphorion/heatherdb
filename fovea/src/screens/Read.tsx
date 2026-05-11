import { useEffect, useState } from "react";
import type { AppCtx } from "@/App";
import type { CollectionSummary, AnalyzeResult } from "@/lib/heather";
import { fmtMs, fmtCount } from "@/lib/format";

export default function Read({
  ctx, initialCollection,
}: { ctx: AppCtx; initialCollection?: string }) {
  const [cols, setCols] = useState<CollectionSummary[]>([]);
  const [collection, setCollection] = useState<string>(initialCollection ?? "");
  const [vectorText, setVectorText] = useState<string>("");
  const [strategy, setStrategy] = useState<"iterative" | "fast">("iterative");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    if (!ctx.client) return;
    ctx.client.collections().then((r) => {
      setCols(r.collections);
      if (!collection && r.collections[0]) setCollection(r.collections[0].name);
    }).catch(() => {});
  }, [ctx.client]);

  const useFingerprint = async () => {
    if (!ctx.client || !collection) return;
    try {
      const fp = await ctx.client.fingerprint(collection);
      setVectorText(JSON.stringify(fp.fingerprint, null, 2));
    } catch (e) { setError(String((e as Error).message)); }
  };

  const useRandom = () => {
    const c = cols.find((x) => x.name === collection);
    const d = c?.dimension ?? 128;
    const v = Array.from({ length: d }, () => +(Math.random() * 2 - 1).toFixed(4));
    // L2 normalize so it looks like a real query.
    const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
    setVectorText(JSON.stringify(v.map((x) => +(x / norm).toFixed(4)), null, 2));
  };

  const run = async () => {
    if (!ctx.client || !collection) return;
    setError(null); setResult(null);
    let q: number[];
    try { q = JSON.parse(vectorText); }
    catch { setError("Vector must be a JSON array of numbers."); return; }
    if (!Array.isArray(q) || q.some((x) => typeof x !== "number")) {
      setError("Vector must be a JSON array of numbers."); return;
    }
    setBusy(true);
    const t0 = performance.now();
    try {
      const r = await ctx.client.analyze(collection, q, strategy);
      setResult(r);
    } catch (e) { setError(String((e as Error).message)); }
    finally {
      setLatency(performance.now() - t0);
      setBusy(false);
    }
  };

  return (
    <div className="px-6 py-6 max-w-6xl">
      <header className="mb-6">
        <div className="kicker mb-2">Read · Analyze</div>
        <h1 className="text-2xl font-normal tracking-tight">
          Ask the memory a question.
        </h1>
        <p className="text-ink-dim text-[14px] leading-relaxed mt-3 max-w-xl">
          Send a vector to <code className="text-white">/analyze</code> and see
          the convergence trace — which attractors fired, with what weight,
          how many iterations to settle.
        </p>
      </header>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto_auto] gap-3 items-end mb-3">
        <Field label="Collection">
          <select
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            className="input w-full"
          >
            {cols.map((c) => (
              <option key={c.name} value={c.name} className="bg-black">
                {c.name} · {fmtCount(c.num_locations)} attractors
              </option>
            ))}
          </select>
        </Field>
        <Field label="Strategy">
          <div className="flex gap-1">
            <Toggle active={strategy === "iterative"} onClick={() => setStrategy("iterative")}>iterative</Toggle>
            <Toggle active={strategy === "fast"}      onClick={() => setStrategy("fast")}>fast</Toggle>
          </div>
        </Field>
        <button onClick={useFingerprint} className="btn">use fingerprint</button>
        <button onClick={useRandom}      className="btn">random vector</button>
      </div>

      {/* Vector editor */}
      <div className="mb-3">
        <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-2">Query vector (JSON array)</div>
        <textarea
          value={vectorText}
          onChange={(e) => setVectorText(e.target.value)}
          rows={8}
          spellCheck={false}
          className="w-full bg-surface-2 border border-ink-line text-white font-mono text-[12px] p-3 focus:border-white/40 focus:outline-none resize-y"
          placeholder='[0.1, -0.3, 0.5, …]'
        />
      </div>

      <div className="flex items-center gap-3 mb-6">
        <button onClick={run} disabled={busy || !collection || !vectorText} className="btn btn-primary">
          {busy ? "analyzing…" : "Analyze →"}
        </button>
        {latency != null && <span className="font-mono text-11 text-ink-muted">{fmtMs(latency)}</span>}
        {error && <span className="font-mono text-11 text-accent-alarm">{error}</span>}
      </div>

      {/* Result */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-3">
          <section className="panel p-4">
            <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-3">
              Activation trace
            </div>
            {result.activated_locations.length === 0 ? (
              <div className="text-ink-muted text-[14px]">No locations activated.</div>
            ) : (
              <ul className="space-y-1.5">
                {result.activated_locations
                  .slice()
                  .sort((a, b) => b.weight - a.weight)
                  .slice(0, 16)
                  .map((a) => (
                    <li key={a.id} className="flex items-baseline gap-3">
                      <span className="font-mono text-11 text-white w-12 shrink-0">#{a.id}</span>
                      <div className="flex-1 h-1 bg-white/[0.06] relative">
                        <div
                          className="absolute top-0 left-0 h-full"
                          style={{
                            width: `${Math.min(1, a.weight) * 100}%`,
                            background: "#a855f7",
                            boxShadow: "0 0 6px -2px #a855f7",
                          }}
                        />
                      </div>
                      <span className="font-mono text-11 text-ink-dim w-12 text-right">{a.weight.toFixed(3)}</span>
                      <span className="font-mono text-11 text-ink-muted w-16 text-right">sim {a.similarity.toFixed(3)}</span>
                    </li>
                  ))}
              </ul>
            )}
          </section>

          <section className="panel p-4 w-[260px]">
            <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-3">Convergence</div>
            <Stat label="iterations"  value={String(result.iterations)} />
            <Stat label="converged"   value={result.converged ? "yes" : "max"}
                  color={result.converged ? "#4ade80" : "#f59e0b"} />
            <Stat label="activated"   value={String(result.total_activations)} />
          </section>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="font-mono text-10 uppercase tracking-ops text-ink-muted">{label}</span>
      {children}
    </label>
  );
}

function Toggle({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="px-3 h-8 font-mono text-10 uppercase tracking-ops border transition-colors"
      style={{
        background: active ? "rgba(168,85,247,0.12)" : "transparent",
        borderColor: active ? "rgba(168,85,247,0.4)" : "rgba(255,255,255,0.10)",
        color: active ? "#a855f7" : "rgba(255,255,255,0.55)",
      }}
    >
      {children}
    </button>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b border-ink-line last:border-b-0">
      <span className="font-mono text-10 uppercase tracking-ops text-ink-muted">{label}</span>
      <span className="font-mono text-[14px]" style={{ color: color ?? "#fff" }}>{value}</span>
    </div>
  );
}
