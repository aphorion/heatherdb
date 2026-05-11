/**
 * HeatherDB HTTP client.
 *
 * Uses Tauri's HTTP plugin when running inside the desktop shell — bypasses
 * the WebView's CORS and gives us per-request timeouts. Falls back to plain
 * fetch in the browser-only `vite dev` mode (handy for UI-only iteration).
 */

import { fetch as tauriFetch } from "@tauri-apps/plugin-http";

// Detect whether we're running under Tauri. Set by Tauri at runtime.
const isTauri = "__TAURI_INTERNALS__" in (globalThis as Record<string, unknown>);

export type Health = { status: string };

export type CollectionSummary = {
  name: string;
  num_locations: number;
  total_writes: number;
  dimension?: number;
};

export type Stats = {
  total_locations: number;
  total_writes: number;
  collections: CollectionSummary[];
};

export type DatabaseInfo = {
  name: string;
  created_at: number;
  dimension: number;
  map_size_mb: number;
  collections: number;
};

export type ProjectionPoint = {
  id: number;
  x: number;       // -1..1
  y: number;       // -1..1
  weight: number;  // 0..1
};

export type Projection = {
  points: ProjectionPoint[];
  count: number;
};

export type ActivatedLocation = {
  id: number;
  similarity: number;
  weight: number;
};

export type AnalyzeResult = {
  iterations: number;
  converged: boolean;
  total_activations: number;
  activated_locations: ActivatedLocation[];
  result: number[];
};

export type ReadResult = {
  result: number[];
  iterations?: number;
  converged?: boolean;
};

export type WriteResult = { count: number };

export type AlgebraResult = {
  collection: string;
  num_locations: number;
};

export type WireEntry = {
  ts: number;
  method: "GET" | "POST" | "DELETE";
  path: string;
  status?: number;
  latency_ms?: number;
  error?: string;
};

type WireListener = (e: WireEntry) => void;
const wireListeners = new Set<WireListener>();
export function onWire(fn: WireListener): () => void {
  wireListeners.add(fn);
  return () => wireListeners.delete(fn);
}
function emitWire(e: WireEntry) {
  for (const fn of wireListeners) fn(e);
}

export type HeatherClientOpts = {
  /** HTTP Basic credentials. Engine has auth on by default. */
  username?: string;
  password?: string;
  /** Active database — every endpoint other than `/health` and `/db*`
   *  routes through `/db/{activeDb}/...`. Defaults to `default`. */
  activeDb?: string;
  timeoutMs?: number;
};

export class HeatherClient {
  /** Bare engine URL — never includes `/db/...`. */
  baseUrl: string;
  username?: string;
  password?: string;
  activeDb: string;
  timeoutMs: number;

  constructor(baseUrl: string, opts: HeatherClientOpts = {}) {
    this.baseUrl   = baseUrl.replace(/\/$/, "");
    this.username  = opts.username;
    this.password  = opts.password;
    this.activeDb  = opts.activeDb ?? "default";
    this.timeoutMs = opts.timeoutMs ?? 10_000;
  }

  /** Build the full URL for a path, applying database scoping if needed.
   *  Paths starting with `/db`, `/health`, or `/server` are sent bare —
   *  everything else is rewritten to `/db/{activeDb}/...`. */
  private fullUrl(path: string): string {
    const isAdmin =
      path === "/health" ||
      path.startsWith("/db") ||
      path.startsWith("/server");
    return isAdmin
      ? `${this.baseUrl}${path}`
      : `${this.baseUrl}/db/${encodeURIComponent(this.activeDb)}${path}`;
  }

  /* ─── plumbing ────────────────────────────────────────────────────────── */

  private async req<T>(
    method: "GET" | "POST" | "DELETE",
    path: string,
    body?: unknown
  ): Promise<T> {
    const url = this.fullUrl(path);
    const t0 = performance.now();

    // Headers — Content-Type only when there's a body, Authorization
    // when both username + password are set. Build the Basic header by
    // hand so it works under both Tauri's HTTP plugin and plain fetch.
    const headers: Record<string, string> = {};
    if (body) headers["Content-Type"] = "application/json";
    if (this.username && this.password) {
      headers["Authorization"] = `Basic ${btoa(`${this.username}:${this.password}`)}`;
    }

    const init: RequestInit = {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(this.timeoutMs),
    };

    try {
      // Tauri's HTTP plugin requires absolute URLs. When the user has
      // configured a relative baseUrl (compose-mode default `/api`), fall
      // back to the WebView's plain fetch — it's happy with either.
      const f = isTauri && /^https?:\/\//i.test(url) ? tauriFetch : fetch;
      const res = await f(url, init);
      const latency_ms = Math.round(performance.now() - t0);
      emitWire({ ts: Date.now(), method, path, status: res.status, latency_ms });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }
      // /health returns a tiny JSON; everything else does too.
      return (await res.json()) as T;
    } catch (e) {
      emitWire({
        ts: Date.now(),
        method,
        path,
        error: e instanceof Error ? e.message : String(e),
        latency_ms: Math.round(performance.now() - t0),
      });
      throw e;
    }
  }

  /* ─── endpoints ───────────────────────────────────────────────────────── */

  health = () => this.req<Health>("GET", "/health");

  /* ─── databases (multi-tenancy) ───────────────────────────────────────── */

  databases = () => this.req<{ databases: DatabaseInfo[] }>("GET", "/db");
  databaseInfo = (name: string) => this.req<DatabaseInfo>("GET", `/db/${encodeURIComponent(name)}`);
  createDatabase = (name: string, dimension: number, map_size_mb?: number) =>
    this.req<DatabaseInfo>("POST", "/db", { name, dimension, map_size_mb });
  dropDatabase = (name: string) =>
    this.req<{ dropped: boolean }>("DELETE", `/db/${encodeURIComponent(name)}`);

  /* ─── collections (scoped to activeDb) ────────────────────────────────── */

  collections = () => this.req<{ collections: CollectionSummary[] }>("GET", "/collections");

  /** Aggregate {totals + per-collection list}. Bare engine doesn't expose
   *  /stats; we synthesise it from /collections. Proxies that DO have
   *  /stats can override this trivially. */
  stats = async (): Promise<Stats> => {
    const { collections } = await this.collections();
    return {
      total_locations: collections.reduce((a, c) => a + (c.num_locations ?? 0), 0),
      total_writes:    collections.reduce((a, c) => a + (c.total_writes   ?? 0), 0),
      collections,
    };
  };

  fingerprint = (collection: string) =>
    this.req<{ fingerprint: number[] }>("GET", `/collections/${collection}/fingerprint`);

  /** Optional — only the FastAPI proxy in heatherdb-pi-demo currently serves
   *  /projection. Bare engine returns 404; the Inspector handles that. */
  projection = (collection: string) =>
    this.req<Projection>("GET", `/collections/${collection}/projection`);

  read = (collection: string, query: number[], strategy: "iterative" | "fast" = "iterative") =>
    this.req<ReadResult>("POST", `/collections/${collection}/read`, { query, strategy });

  analyze = (collection: string, query: number[], strategy: "iterative" | "fast" = "iterative") =>
    this.req<AnalyzeResult>("POST", `/collections/${collection}/analyze`, { query, strategy });

  write = (collection: string, vectors: number[][]) =>
    this.req<WriteResult>("POST", `/collections/${collection}/write`, { vectors });

  algebraAdd = (source_a: string, source_b: string, target?: string) =>
    this.req<AlgebraResult>("POST", "/algebra/add", { source_a, source_b, target });

  algebraSub = (source_a: string, source_b: string, target?: string) =>
    this.req<AlgebraResult>("POST", "/algebra/sub", { source_a, source_b, target });

  algebraScale = (source: string, alpha: number, target?: string) =>
    this.req<AlgebraResult>("POST", "/algebra/scale", { source, alpha, target });
}
