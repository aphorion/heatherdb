/**
 * Connection storage.
 *
 * Backed by Tauri's plugin-store when running in the desktop shell — that
 * gives us a real on-disk JSON file that survives reinstalls AND is shared
 * across processes (think future CLI helpers, multi-window).
 *
 * Falls back to localStorage when running outside Tauri (`npm run dev`)
 * so UI iteration in a regular browser still works. The two backends share
 * the same async surface so callers don't branch.
 */

import { Store } from "@tauri-apps/plugin-store";

export type Connection = {
  id: string;
  label: string;
  /** Bare engine URL — no /db/ suffix. The DB is selected separately. */
  url: string;
  /** HTTP Basic auth — engine has auth on by default. Leave both empty
   *  to talk to an --auth-disabled engine. */
  username?: string;
  password?: string;
  /** "rw" | "ro" — currently informational, no enforcement yet. */
  role: "rw" | "ro";
  /** Active database for this connection. Defaults to "default". The
   *  picker on the Connections screen sets it; every API call routes
   *  through /db/{active_db}/... */
  active_db?: string;
  added_at: number;
  last_used_at?: number;
};

const STORE_FILE   = "connections.json";   // landed under Tauri's app-data dir
const KEY_LIST     = "connections";
const KEY_ACTIVE   = "active_connection";
const LS_LIST_KEY  = "fovea.connections.v1";
const LS_ACTIVE_KEY = "fovea.active_connection.v1";

const isTauri = "__TAURI_INTERNALS__" in (globalThis as Record<string, unknown>);

/* ─── backend abstraction ──────────────────────────────────────────────── */

interface Backend {
  loadList(): Promise<Connection[]>;
  saveList(list: Connection[]): Promise<void>;
  loadActiveId(): Promise<string | null>;
  saveActiveId(id: string | null): Promise<void>;
}

class TauriBackend implements Backend {
  private storePromise: Promise<Store> | null = null;
  private store(): Promise<Store> {
    if (!this.storePromise) this.storePromise = Store.load(STORE_FILE);
    return this.storePromise;
  }
  async loadList(): Promise<Connection[]> {
    const s = await this.store();
    return ((await s.get<Connection[]>(KEY_LIST)) ?? []);
  }
  async saveList(list: Connection[]): Promise<void> {
    const s = await this.store();
    await s.set(KEY_LIST, list);
    await s.save();
  }
  async loadActiveId(): Promise<string | null> {
    const s = await this.store();
    return (await s.get<string>(KEY_ACTIVE)) ?? null;
  }
  async saveActiveId(id: string | null): Promise<void> {
    const s = await this.store();
    if (id) await s.set(KEY_ACTIVE, id);
    else    await s.delete(KEY_ACTIVE);
    await s.save();
  }
}

class WebBackend implements Backend {
  async loadList(): Promise<Connection[]> {
    try { return JSON.parse(localStorage.getItem(LS_LIST_KEY) ?? "[]"); }
    catch { return []; }
  }
  async saveList(list: Connection[]): Promise<void> {
    localStorage.setItem(LS_LIST_KEY, JSON.stringify(list));
  }
  async loadActiveId(): Promise<string | null> {
    return localStorage.getItem(LS_ACTIVE_KEY);
  }
  async saveActiveId(id: string | null): Promise<void> {
    if (id) localStorage.setItem(LS_ACTIVE_KEY, id);
    else    localStorage.removeItem(LS_ACTIVE_KEY);
  }
}

const backend: Backend = isTauri ? new TauriBackend() : new WebBackend();

/* ─── public API ───────────────────────────────────────────────────────── */

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

export async function listConnections(): Promise<Connection[]> {
  return backend.loadList();
}

export async function saveConnection(
  c: Omit<Connection, "id" | "added_at"> & { id?: string }
): Promise<Connection> {
  const list = await backend.loadList();
  const existing = c.id ? list.find((x) => x.id === c.id) : undefined;
  const next: Connection = existing
    ? { ...existing, ...c, id: existing.id }
    : { id: uid(), added_at: Date.now(), ...c };
  const others = list.filter((x) => x.id !== next.id);
  await backend.saveList([next, ...others]);
  return next;
}

export async function deleteConnection(id: string): Promise<void> {
  const list = (await backend.loadList()).filter((x) => x.id !== id);
  await backend.saveList(list);
  if ((await backend.loadActiveId()) === id) await backend.saveActiveId(null);
}

export async function setActiveConnectionId(id: string | null): Promise<void> {
  await backend.saveActiveId(id);
}

export async function getActiveConnectionId(): Promise<string | null> {
  return backend.loadActiveId();
}

export async function getActiveConnection(): Promise<Connection | null> {
  const id = await backend.loadActiveId();
  if (!id) return null;
  const list = await backend.loadList();
  return list.find((x) => x.id === id) ?? null;
}

export async function touchLastUsed(id: string): Promise<void> {
  const list = await backend.loadList();
  const c = list.find((x) => x.id === id);
  if (!c) return;
  c.last_used_at = Date.now();
  await backend.saveList(list);
}
