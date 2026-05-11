# Fovea

> Bring sharp inspection to HeatherDB.

Fovea is the operator's instrument for [HeatherDB](https://github.com/aphorion/heather-db) —
a desktop app that lets you see the attractor landscape, trace activations,
and compose memories with algebra. Tauri 2 shell, React + Tailwind UI, talks
to any HeatherDB instance over plain HTTP.

Named after the fovea centralis — the central pit of the retina where vision
is sharpest. The part that brings the world into focus.

```
                ┌───────────────────────────────────┐
                │        ◐  ◯◯◯ ◯  ◯  ◯◯◯ ◐         │
                │      ◯  ◯  *  *◯◯ ◯◯  ◯  ◯        │
                │     ◯  *  ◍  ◍ ◯ ◯  *◯  ◯  ◯      │
                │      ◯  ◯  ◯  ◯  ◯  ◯◯◯  ◯        │
                │        ◯◯◯ ◯  ◯  ◯◯ ◯ ◯           │
                │                                   │
                │       fovea — what you see        │
                └───────────────────────────────────┘
```

## What's in v0.1

| Screen | What it does |
|--------|--------------|
| **Connections** | Add HeatherDB instances by URL. Switch between local engine, $5 VPS, Pi on the LAN. Multi-connection storage, one active at a time. |
| **Collections** | Gallery view of every collection on the active server. Each tile is a tiny synthetic landscape preview; size + write counts in the footer. Filter by name. |
| **Inspector** | Per-collection deep dive. **Landscape** tab: pannable / zoomable canvas of the attractor population. Hover a point for its id and weight. Click *probe fingerprint* to fire the collection's centroid through `/analyze` — dominant attractors light up with accent halos and list in the side panel. **Stats** tab: counts + sample latencies on demand. |
| **Read** | Query workbench. Pick a collection, paste a vector or generate one, hit Analyze. See the convergence trace — every activated location with its weight + similarity, iteration count, convergence flag. |
| **Algebra** | Compose collections with `+` / `−`. Pick A, pick B, op, optional target name. The result is a new collection you can immediately open in the inspector. |

A toggleable **Wire log** in the right rail shows every HTTP call to the
server — method, path, status, latency — colour-coded by speed.

## Why a separate app?

HeatherDB has the landing site (marketing + showcase labs) and the engine
itself. Fovea is the third leg: the **operator instrument**. It's a desktop
app because:

- **CORS-free HTTP** — Tauri's HTTP plugin bypasses the WebView's CORS,
  so Fovea can talk to a plain-HTTP HeatherDB anywhere on the network.
- **Persistent connections** — local storage of multiple servers, one-click
  switching, no auth dances.
- **Future-friendly for big visuals** — the attractor landscape is a canvas
  today; WebGL later for >100k attractors. A native shell makes that easy.

## Run it

### Prerequisites

- Node 20+ and `npm` (or `pnpm`/`yarn`/`bun` — adjust commands)
- Rust 1.81+ (`rustup install stable`)
- Platform deps for Tauri:
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Linux**: `build-essential libwebkit2gtk-4.1-dev libssl-dev pkg-config`
  - **Windows**: WebView2 (preinstalled on Win11), MSVC build tools

A running HeatherDB instance to point at — see
[aphorion/heather-db](https://github.com/aphorion/heather-db) for the
one-line VPS deploy or `cargo run -p heather_server` locally.

### Develop

```bash
git clone https://github.com/aphorion/fovea
cd fovea

npm install

# (one-time) generate the platform icons from the SVG source
npx tauri icon src-tauri/icons/icon.svg

# Web-only iteration — no native shell, fastest UI loop
npm run dev
# → http://localhost:1420   (works in your browser; CORS may bite you,
#                             use Tauri mode for full talk-to-HeatherDB)

# Full Tauri dev shell — recommended
npm run tauri:dev
```

### Web mode (Docker compose)

Inside the engine repo's `docker compose up`, Fovea's web build runs
behind a Caddy front-end at `http://localhost:8080`. Caddy reverse-proxies
`/api/*` to the engine, so the SPA never touches CORS.

#### Enable HTTP basic auth

The compose service loads `fovea/.env` if present. Two vars enable
auth on **everything**, including the `/api/*` reverse-proxy:

```bash
cp fovea/.env.example fovea/.env

# Generate the bcrypt hash with caddy itself — no install needed.
docker run --rm caddy:2-alpine \
  caddy hash-password --plaintext "your-strong-password"

# Paste the hash into fovea/.env:
#   AUTH_USER=op
#   AUTH_PASS_HASH=$2a$14$abcd…
```

`docker compose up -d` brings the stack up with auth enforced. The
browser shows a Basic-Auth prompt before the app loads; engine
endpoints are equally gated.

If both vars are unset the entrypoint runs in **no-auth mode** and
prints a loud warning — fine for local-only / single-user laptops, not
fine on a shared network. Setting only one is treated as a config
error and the container refuses to start.

### Build a release binary

```bash
npm run tauri:build
```

Outputs to `src-tauri/target/release/bundle/`:

| Platform | Artefact |
|----------|----------|
| macOS    | `.app` + `.dmg` |
| Linux    | `.AppImage`, `.deb`, `.rpm` |
| Windows  | `.msi` + `.exe` (NSIS) |

Code-sign + notarize per Tauri's
[distribution guide](https://v2.tauri.app/distribute/) for production.

## First-run flow

1. Launch the app → **Connections** screen, no servers yet.
2. Add one (label = anything; URL = `http://127.0.0.1:6380` for local). Click **add**.
3. Click **connect** on the row. Top-bar status dot turns green.
4. Sidebar lights up. Click **Collections** to see what's on the server.
5. Click any collection → Inspector → click **probe fingerprint** to see the
   dominant attractors light up.

## Architecture sketch

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri 2 shell (Rust)                                       │
│    ├─ tauri-plugin-http     ← bypass WebView CORS           │
│    └─ tauri-plugin-store    ← future: cross-process state   │
│  ─────────────────────────────────────────────────────────  │
│  React 19 + TypeScript + Tailwind (Vite)                    │
│    src/                                                     │
│      App.tsx              router + active-connection state  │
│      lib/                                                   │
│        heather.ts         HTTP client + wire-log event bus  │
│        connections.ts     plugin-store backed (web fallback)│
│        format.ts          fmtMs, fmtCount, fmtRelativeTs    │
│      components/                                            │
│        Layout.tsx         top chrome + sidebar + wire panel │
│        Sidebar.tsx        nav, gated on active connection   │
│        StatusBar.tsx      live counts (poll /stats)         │
│        WireLog.tsx        scrolling HTTP tail               │
│        AttractorMap.tsx   canvas landscape (pan/zoom/hover) │
│      screens/                                               │
│        Connections.tsx                                      │
│        Collections.tsx                                      │
│        Inspector.tsx      Landscape + Stats tabs            │
│        Read.tsx           query workbench + activation tree │
│        Algebra.tsx        +/- composer                      │
└─────────────────────────────────────────────────────────────┘
```

## Roadmap (v0.2+)

- **Live write tail** — SSE-driven view of writes reshaping the population.
- **Co-activation heatmap** — surface emergent clusters without a query.
- **Snapshot diff** — semantic diff between two snapshots (new / merged / dropped attractors).
- **Encoder bridge** — bundle a small ONNX sentence-transformer so you can
  type text and query directly without paste-the-vector.
- **Algebra** — `−α` (scaled subtract), `∩` (intersect), saved recipes.
- **Multi-connection split view** — A/B same query against two servers.
- **WebGL landscape** — for collections >50k attractors.

## Where this lives

Fovea is shipped **inside** the engine repo at
[`aphorion/heather-db/fovea`](../). One `docker compose up` from the
engine repo root brings up the engine + Fovea web build together
(see the engine's top-level README for the compose layout).

## License

TBD.

---

**Related**
- [aphorion/heatherdb-landing](https://github.com/aphorion/heatherdb-landing) — the product website + scene-driven demo labs
- [aphorion/heatherdb-pi-demo](https://github.com/aphorion/heatherdb-pi-demo) — conference rig
- [aphorion/heatherdb-samples](https://github.com/aphorion/heathedb-sample_projects) — 26 example projects
