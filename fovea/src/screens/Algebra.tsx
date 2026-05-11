import { useEffect, useState } from "react";
import type { AppCtx } from "@/App";
import type { CollectionSummary, AlgebraResult } from "@/lib/heather";
import { fmtCount } from "@/lib/format";

type Op = "add" | "sub";

export default function Algebra({ ctx }: { ctx: AppCtx }) {
  const [cols, setCols] = useState<CollectionSummary[]>([]);
  const [a, setA] = useState<string>("");
  const [b, setB] = useState<string>("");
  const [op, setOp] = useState<Op>("add");
  const [target, setTarget] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AlgebraResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ctx.client) return;
    ctx.client.collections().then((r) => {
      setCols(r.collections);
      if (!a && r.collections[0]) setA(r.collections[0].name);
      if (!b && r.collections[1]) setB(r.collections[1].name);
    }).catch(() => {});
  }, [ctx.client]);

  const run = async () => {
    if (!ctx.client || !a || !b) return;
    setBusy(true); setError(null); setResult(null);
    try {
      const t = target.trim() || undefined;
      const r = op === "add"
        ? await ctx.client.algebraAdd(a, b, t)
        : await ctx.client.algebraSub(a, b, t);
      setResult(r);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="px-6 py-6 max-w-5xl">
      <header className="mb-6">
        <div className="kicker mb-2">Algebra workbench</div>
        <h1 className="text-2xl font-normal tracking-tight">
          Compose memories.
        </h1>
        <p className="text-ink-dim text-[14px] leading-relaxed mt-3 max-w-xl">
          The thing vector DBs structurally cannot do — math on whole memory
          populations. Pick A, pick B, pick an op. The result is a new
          collection you can read against.
        </p>
      </header>

      {/* Composition */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto] gap-3 items-end mb-4">
        <Field label="Source A">
          <CollectionPicker cols={cols} value={a} onChange={setA} />
        </Field>

        <div className="flex flex-col items-center gap-2">
          <span className="font-mono text-10 uppercase tracking-ops text-ink-muted">op</span>
          <div className="flex gap-1">
            <OpBtn active={op === "add"} onClick={() => setOp("add")}>+</OpBtn>
            <OpBtn active={op === "sub"} onClick={() => setOp("sub")}>−</OpBtn>
          </div>
        </div>

        <Field label="Source B">
          <CollectionPicker cols={cols} value={b} onChange={setB} />
        </Field>

        <button onClick={run} disabled={busy || !a || !b} className="btn btn-primary">
          {busy ? "composing…" : "Run →"}
        </button>
      </div>

      <Field label="Target name (optional)">
        <input
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          placeholder={`auto: _algebra_${op}_${a || "a"}_${b || "b"}`}
          className="input w-full max-w-md"
        />
      </Field>

      {error && (
        <div className="mt-6 border border-accent-alarm/30 bg-accent-alarm/[0.06] text-accent-alarm font-mono text-11 uppercase tracking-ops px-4 py-3">
          {error}
        </div>
      )}

      {result && (
        <section className="panel p-5 mt-6">
          <div className="font-mono text-10 uppercase tracking-ops text-ink-muted mb-2">Result</div>
          <div className="text-[15px] mb-3">
            <span className="text-white">{result.collection}</span>
            <span className="text-ink-muted"> · </span>
            <span className="text-ink-muted">{fmtCount(result.num_locations)} attractors</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => ctx.navigate({ kind: "inspector", collection: result.collection })}
              className="btn"
            >
              open in inspector →
            </button>
            <button
              onClick={() => ctx.navigate({ kind: "read", collection: result.collection })}
              className="btn"
            >
              query →
            </button>
          </div>
        </section>
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

function CollectionPicker({
  cols, value, onChange,
}: { cols: CollectionSummary[]; value: string; onChange: (v: string) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="input w-full"
    >
      {cols.length === 0 && <option value="">(no collections)</option>}
      {cols.map((c) => (
        <option key={c.name} value={c.name} className="bg-black">
          {c.name} · {fmtCount(c.num_locations)} attractors
        </option>
      ))}
    </select>
  );
}

function OpBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-10 h-8 font-mono text-[14px] border transition-colors"
      style={{
        background: active ? "rgba(168,85,247,0.15)" : "transparent",
        borderColor: active ? "rgba(168,85,247,0.4)" : "rgba(255,255,255,0.10)",
        color: active ? "#a855f7" : "rgba(255,255,255,0.55)",
      }}
    >
      {children}
    </button>
  );
}
