import { useEffect, useMemo, useState } from "react";
import { HeatherClient } from "@/lib/heather";
import {
  getActiveConnection,
  type Connection,
  setActiveConnectionId,
  touchLastUsed,
} from "@/lib/connections";
import Layout from "@/components/Layout";
import Connections from "@/screens/Connections";
import Collections from "@/screens/Collections";
import Inspector from "@/screens/Inspector";
import Read from "@/screens/Read";
import Algebra from "@/screens/Algebra";

export type Route =
  | { kind: "connections" }
  | { kind: "collections" }
  | { kind: "inspector"; collection: string }
  | { kind: "read";      collection?: string }
  | { kind: "algebra" };

export type AppCtx = {
  client: HeatherClient | null;
  connection: Connection | null;
  serverOk: boolean | null;
  navigate: (r: Route) => void;
  switchConnection: (id: string | null) => Promise<void>;
};

export default function App() {
  const [conn, setConn] = useState<Connection | null>(null);
  const [route, setRoute] = useState<Route>({ kind: "connections" });
  const [serverOk, setServerOk] = useState<boolean | null>(null);
  const [bootstrapped, setBootstrapped] = useState(false);

  // Hydrate the active connection from the store on first mount.
  useEffect(() => {
    let cancelled = false;
    getActiveConnection()
      .then((c) => {
        if (cancelled) return;
        setConn(c);
        setRoute(c ? { kind: "collections" } : { kind: "connections" });
      })
      .finally(() => !cancelled && setBootstrapped(true));
    return () => { cancelled = true; };
  }, []);

  const client = useMemo(
    () => (conn ? new HeatherClient(conn.url) : null),
    [conn?.id, conn?.url]
  );

  // Lightweight health probe whenever the active connection changes.
  useEffect(() => {
    let cancelled = false;
    if (!client) { setServerOk(null); return; }
    setServerOk(null);
    client.health()
      .then(() => !cancelled && setServerOk(true))
      .catch(() => !cancelled && setServerOk(false));
    return () => { cancelled = true; };
  }, [client]);

  // Re-check on a slow interval so the status dot stays honest.
  useEffect(() => {
    if (!client) return;
    const id = setInterval(() => {
      client.health().then(() => setServerOk(true)).catch(() => setServerOk(false));
    }, 8000);
    return () => clearInterval(id);
  }, [client]);

  const switchConnection = async (id: string | null) => {
    await setActiveConnectionId(id);
    if (id) await touchLastUsed(id);
    const next = await getActiveConnection();
    setConn(next);
    setRoute(next ? { kind: "collections" } : { kind: "connections" });
  };

  const ctx: AppCtx = {
    client,
    connection: conn,
    serverOk,
    navigate: setRoute,
    switchConnection,
  };

  // Avoid a flash of "no connections" while the store hydrates.
  if (!bootstrapped) return <BootSplash />;

  return (
    <Layout ctx={ctx} route={route}>
      <ScreenSwitch ctx={ctx} route={route} />
    </Layout>
  );
}

function ScreenSwitch({ ctx, route }: { ctx: AppCtx; route: Route }) {
  if (!ctx.client || route.kind === "connections") {
    return <Connections ctx={ctx} />;
  }
  switch (route.kind) {
    case "collections": return <Collections ctx={ctx} />;
    case "inspector":   return <Inspector  ctx={ctx} collection={route.collection} />;
    case "read":        return <Read       ctx={ctx} initialCollection={route.collection} />;
    case "algebra":     return <Algebra    ctx={ctx} />;
  }
}

function BootSplash() {
  return (
    <div className="h-full w-full grid place-items-center bg-black text-white">
      <div className="font-mono text-10 uppercase tracking-ops text-ink-muted">
        loading…
      </div>
    </div>
  );
}
