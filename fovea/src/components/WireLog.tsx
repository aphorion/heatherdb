import { useEffect, useRef, useState } from "react";
import { onWire, type WireEntry } from "@/lib/heather";
import { fmtMs, fmtTime } from "@/lib/format";

const MAX = 200;

export default function WireLog() {
  const [items, setItems] = useState<WireEntry[]>([]);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return onWire((e) => {
      setItems((prev) => [...prev.slice(-(MAX - 1)), e]);
    });
  }, []);

  useEffect(() => {
    // Stick to bottom whenever a new line lands.
    const el = ref.current; if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [items]);

  return (
    <div className="h-full flex flex-col">
      <div className="h-9 px-3 border-b border-ink-line flex items-center justify-between font-mono text-10 uppercase tracking-ops text-ink-muted">
        <span>Wire · client → server</span>
        <button
          onClick={() => setItems([])}
          className="text-ink-ghost hover:text-white"
        >
          clear
        </button>
      </div>
      <div ref={ref} className="flex-1 min-h-0 overflow-auto px-3 py-2 font-mono text-[11px]">
        {items.length === 0 && (
          <div className="text-ink-ghost py-2">No traffic yet.</div>
        )}
        <ul className="space-y-1">
          {items.map((e, i) => {
            const color =
              e.error ? "text-accent-alarm"
              : e.method === "POST" ? "text-accent-cool"
              : e.method === "DELETE" ? "text-accent-warm"
              : "text-ink-dim";
            const lat = e.latency_ms ?? 0;
            const latColor =
              lat < 10  ? "text-accent-ok"
              : lat < 100 ? "text-accent-cool"
              : lat < 1000 ? "text-accent-warm"
              : "text-accent-alarm";
            return (
              <li key={i} className="flex items-baseline gap-2 truncate">
                <span className="text-ink-ghost shrink-0">{fmtTime(e.ts)}</span>
                <span className={`${color} font-medium shrink-0`}>{e.method}</span>
                <span className="text-white truncate">{e.path}</span>
                <span className="ml-auto pl-2 shrink-0">
                  {e.status && <span className="text-ink-ghost mr-2">{e.status}</span>}
                  <span className={latColor}>{fmtMs(e.latency_ms)}</span>
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
