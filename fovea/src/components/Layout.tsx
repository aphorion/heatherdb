import type { AppCtx, Route } from "@/App";
import Sidebar from "./Sidebar";
import StatusBar from "./StatusBar";
import WireLog from "./WireLog";
import { useState } from "react";

export default function Layout({
  ctx, route, children,
}: {
  ctx: AppCtx; route: Route; children: React.ReactNode;
}) {
  const [wireOpen, setWireOpen] = useState(false);

  return (
    <div className="h-full w-full flex flex-col bg-black text-white">
      {/* TOP CHROME — brand + active connection */}
      <header className="h-11 shrink-0 px-4 border-b border-ink-line flex items-center justify-between gap-4 bg-surface-1">
        <div className="flex items-center gap-3">
          <Mark />
          <div className="flex items-baseline gap-2">
            <span className="font-medium tracking-tight text-[14px]">Fovea</span>
            <span className="font-mono text-10 uppercase tracking-ops text-ink-ghost">
              v0.1
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {ctx.connection && (
            <span className="pill">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background:
                    ctx.serverOk === null ? "#ffd166"
                    : ctx.serverOk ? "#4ade80" : "#f87171",
                  boxShadow:
                    ctx.serverOk
                      ? "0 0 6px #4ade80"
                      : ctx.serverOk === false
                        ? "0 0 6px #f87171"
                        : undefined,
                }}
              />
              <span className="text-white/75">{ctx.connection.label}</span>
              <span className="text-ink-ghost">·</span>
              <span className="text-ink-ghost normal-case">
                {ctx.connection.url.replace(/^https?:\/\//, "")}
              </span>
            </span>
          )}
          <button
            onClick={() => setWireOpen((p) => !p)}
            className="btn"
            title="Toggle wire log"
            aria-pressed={wireOpen}
          >
            Wire
          </button>
        </div>
      </header>

      {/* MAIN AREA */}
      <div className="flex-1 min-h-0 flex">
        <Sidebar ctx={ctx} route={route} />

        <main className="flex-1 min-w-0 min-h-0 overflow-hidden flex">
          <div className="flex-1 min-w-0 min-h-0 overflow-y-auto overflow-x-hidden">
            {children}
          </div>

          {wireOpen && (
            <aside className="w-[360px] shrink-0 border-l border-ink-line bg-surface-1">
              <WireLog />
            </aside>
          )}
        </main>
      </div>

      {/* BOTTOM STATUS */}
      <StatusBar ctx={ctx} />
    </div>
  );
}

function Mark() {
  return (
    <svg width="16" height="16" viewBox="0 0 64 64" className="text-white">
      <circle cx="32" cy="32" r="20" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="32" cy="32" r="6"  fill="currentColor" />
    </svg>
  );
}
