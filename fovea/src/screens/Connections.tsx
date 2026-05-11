import { useEffect, useState } from "react";
import type { AppCtx } from "@/App";
import {
  type Connection,
  listConnections,
  saveConnection,
  deleteConnection,
} from "@/lib/connections";
import { fmtRelativeTs } from "@/lib/format";

export default function Connections({ ctx }: { ctx: AppCtx }) {
  const [list, setList] = useState<Connection[]>([]);
  const [editing, setEditing] = useState<Partial<Connection> | null>(null);

  const reload = async () => setList(await listConnections());
  useEffect(() => { void reload(); }, []);

  return (
    <div className="px-8 py-8 max-w-4xl mx-auto">
      <header className="mb-8">
        <div className="kicker mb-3">Connections</div>
        <h1 className="text-2xl font-normal tracking-tight">
          Where is HeatherDB running?
        </h1>
        <p className="text-ink-dim text-[14px] leading-relaxed mt-3 max-w-xl">
          Add the URL of any HeatherDB instance — local engine, $5 VPS, Pi
          on the LAN. Fovea connects via plain HTTP. Nothing leaves your machine.
        </p>
      </header>

      {/* List */}
      <div className="border border-ink-line">
        {list.length === 0 ? (
          <div className="px-6 py-10 text-center text-ink-muted text-[14px]">
            No connections yet. Add one below.
          </div>
        ) : (
          <ul className="divide-y divide-ink-line">
            {list.map((c) => {
              const isActive = ctx.connection?.id === c.id;
              return (
                <li key={c.id} className="flex items-center gap-4 px-4 py-3">
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{
                      background: isActive ? "#4ade80" : "rgba(255,255,255,0.18)",
                      boxShadow: isActive ? "0 0 6px #4ade80" : undefined,
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-[14px] text-white truncate">{c.label}</div>
                    <div className="font-mono text-11 text-ink-muted truncate">
                      {c.url}
                      <span className="text-ink-ghost"> · db </span>
                      <span className="text-white">{c.active_db ?? "default"}</span>
                      <span className="text-ink-ghost"> · </span>
                      <span className="uppercase">{c.role}</span>
                      {c.username && (
                        <>
                          <span className="text-ink-ghost"> · user </span>
                          <span className="text-white">{c.username}</span>
                        </>
                      )}
                      <span className="text-ink-ghost"> · </span>
                      last used {fmtRelativeTs(c.last_used_at)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      className={isActive ? "btn opacity-50 cursor-default" : "btn btn-primary"}
                      onClick={() => { if (!isActive) void ctx.switchConnection(c.id); }}
                      disabled={isActive}
                    >
                      {isActive ? "active" : "connect"}
                    </button>
                    <button
                      className="btn"
                      onClick={() => setEditing(c)}
                    >
                      edit
                    </button>
                    <button
                      className="btn hover:!text-accent-alarm hover:!border-accent-alarm/40"
                      onClick={async () => {
                        if (!confirm(`Remove "${c.label}"?`)) return;
                        await deleteConnection(c.id);
                        await reload();
                        if (isActive) await ctx.switchConnection(null);
                      }}
                    >
                      remove
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Add / edit form */}
      <div className="mt-8 panel p-5">
        <div className="kicker mb-4">{editing && "id" in (editing ?? {}) ? "Edit connection" : "Add connection"}</div>
        <ConnectionForm
          initial={editing ?? undefined}
          onSubmit={async (data) => {
            const saved = await saveConnection(data);
            setEditing(null);
            const wasEmpty = list.length === 0;
            await reload();
            if (wasEmpty) await ctx.switchConnection(saved.id);
          }}
          onCancel={editing ? () => setEditing(null) : undefined}
        />
      </div>
    </div>
  );
}

// Tauri's WebView talks to engines directly via the HTTP plugin (CORS-free).
// In browser mode (typically the dockerised Caddy front-end in compose) we
// default to the relative `/api` path so Caddy's reverse-proxy can hide
// CORS for us.
const isTauri = "__TAURI_INTERNALS__" in (globalThis as Record<string, unknown>);
const DEFAULT_URL = isTauri ? "http://127.0.0.1:6380" : "/api";

function ConnectionForm({
  initial, onSubmit, onCancel,
}: {
  initial?: Partial<Connection>;
  onSubmit: (c: Omit<Connection, "id" | "added_at"> & { id?: string }) => void;
  onCancel?: () => void;
}) {
  const [label,    setLabel]    = useState(initial?.label    ?? "Local");
  const [url,      setUrl]      = useState(initial?.url      ?? DEFAULT_URL);
  const [username, setUsername] = useState(initial?.username ?? "admin");
  const [password, setPassword] = useState(initial?.password ?? "");
  const [activeDb, setActiveDb] = useState(initial?.active_db ?? "default");
  const [role,     setRole]     = useState<"rw" | "ro">(initial?.role ?? "rw");

  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!label.trim() || !url.trim()) return;
        onSubmit({
          id: initial?.id,
          label: label.trim(),
          url: url.trim(),
          username: username.trim() || undefined,
          password: password || undefined,
          active_db: activeDb.trim() || "default",
          role,
        });
      }}
    >
      <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-3 items-end">
        <Field label="Label">
          <input value={label} onChange={(e) => setLabel(e.target.value)} className="input w-full" placeholder="Local" />
        </Field>
        <Field label="Engine URL (no /db suffix)">
          <input value={url} onChange={(e) => setUrl(e.target.value)} className="input w-full" placeholder="http://127.0.0.1:6380" />
        </Field>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_1fr_auto] gap-3 items-end">
        <Field label="Username">
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="input w-full"
            placeholder="admin"
            autoComplete="username"
          />
        </Field>
        <Field label="Password">
          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            className="input w-full"
            placeholder="leave blank for --auth-disabled engines"
            autoComplete="current-password"
          />
        </Field>
        <Field label="Active database">
          <input
            value={activeDb}
            onChange={(e) => setActiveDb(e.target.value)}
            className="input w-full"
            placeholder="default"
          />
        </Field>
        <Field label="Role">
          <div className="flex gap-1">
            <RolePill active={role === "rw"} onClick={() => setRole("rw")}>r/w</RolePill>
            <RolePill active={role === "ro"} onClick={() => setRole("ro")}>r/o</RolePill>
          </div>
        </Field>
      </div>

      <p className="font-mono text-10 uppercase tracking-ops text-ink-ghost">
        First-boot creates an <code className="text-white">admin</code> user — copy the password from the engine's
        boot log. Active database picks which <code className="text-white">/db/{"{db}"}/...</code> everything routes through.
      </p>

      <div className="flex items-center gap-2 pt-1">
        <button type="submit" className="btn btn-primary">{initial?.id ? "save" : "add"}</button>
        {onCancel && <button type="button" className="btn" onClick={onCancel}>cancel</button>}
      </div>
    </form>
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

function RolePill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
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
