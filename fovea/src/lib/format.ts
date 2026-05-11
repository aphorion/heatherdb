/**
 * Tiny formatters used across the operator UI. Operator UIs live or die by
 * how readable their numbers are.
 */

export function fmtCount(n: number | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function fmtMs(ms: number | undefined): string {
  if (ms == null) return "—";
  if (ms < 1)       return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000)    return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtRelativeTs(ts: number | undefined): string {
  if (!ts) return "never";
  const dt = (Date.now() - ts) / 1000;
  if (dt < 60)        return `${Math.floor(dt)}s ago`;
  if (dt < 3600)      return `${Math.floor(dt / 60)}m ago`;
  if (dt < 86400)     return `${Math.floor(dt / 3600)}h ago`;
  return `${Math.floor(dt / 86400)}d ago`;
}

export function fmtTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}
