import type { AppCtx, Route } from "@/App";

const NAV: Array<{ label: string; route: Route; needsConnection: boolean }> = [
  { label: "Connections", route: { kind: "connections" }, needsConnection: false },
  { label: "Collections", route: { kind: "collections" }, needsConnection: true  },
  { label: "Read",        route: { kind: "read" },        needsConnection: true  },
  { label: "Algebra",     route: { kind: "algebra" },     needsConnection: true  },
];

export default function Sidebar({ ctx, route }: { ctx: AppCtx; route: Route }) {
  return (
    <nav className="w-[180px] shrink-0 border-r border-ink-line bg-surface-1 flex flex-col">
      <ul className="flex-1 py-3">
        {NAV.map((n) => {
          const disabled = n.needsConnection && !ctx.connection;
          const active =
            route.kind === n.route.kind ||
            (route.kind === "inspector" && n.route.kind === "collections");
          return (
            <li key={n.label}>
              <button
                disabled={disabled}
                onClick={() => ctx.navigate(n.route)}
                className={[
                  "w-full text-left px-4 py-2 text-[13px] flex items-center gap-3",
                  "border-l-2 transition-colors",
                  active
                    ? "border-accent text-white bg-white/[0.04]"
                    : disabled
                      ? "border-transparent text-ink-ghost cursor-not-allowed"
                      : "border-transparent text-ink-dim hover:text-white hover:bg-white/[0.02]",
                ].join(" ")}
              >
                <span
                  aria-hidden
                  className="w-1 h-1 rounded-full"
                  style={{ background: active ? "var(--accent, #a855f7)" : "currentColor", opacity: active ? 1 : 0.4 }}
                />
                <span className="font-mono text-11 uppercase tracking-ops">{n.label}</span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="p-3 border-t border-ink-line">
        <a
          href="https://github.com/aphorion/heather-db"
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-10 uppercase tracking-ops text-ink-ghost hover:text-ink-muted"
        >
          aphorion/heather-db ↗
        </a>
      </div>
    </nav>
  );
}
