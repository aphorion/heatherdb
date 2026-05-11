# Fovea — operator GUI

Fovea is the desktop instrument for HeatherDB. It's a Tauri 2 + React
app that connects to any HeatherDB instance over HTTP, browses
databases and collections, runs reads with live activation traces,
composes new collections via algebra, and surfaces the wire log.

Source lives in [`fovea/`](../fovea/) inside this repo.

```
┌────────────────────────────────────────────────────────────────────┐
│  ◉ Connections     ▌ DemoLocal · 127.0.0.1:6380 · db default      │
├──────────────┬──────────────────────────────────────┬─────────────┤
│ Connections  │  ATTRACTOR LANDSCAPE                 │   Top acts  │
│ Collections  │  ┌──────────────────────────────┐   │   #237 ▮▮▮  │
│ Read         │  │      ●●●     ●  ●●           │   │   #401 ▮▮   │
│ Algebra      │  │ ●●  ●  ◐  ●  ●●●●            │   │   #82  ▮    │
│              │  │  ●● ●●●●●  ●  ●● ●           │   │   …         │
│              │  │     ●●  ●●  ●● ●             │   │             │
│              │  └──────────────────────────────┘   │             │
│              │  Wire · client → server   GET 8ms   │             │
└──────────────┴──────────────────────────────────────┴─────────────┘
```

## Two ways to run it

### Desktop (Tauri)

The "real" Fovea — single installable binary, talks to engines anywhere.

```bash
cd fovea
npm install
npx tauri icon src-tauri/icons/icon.svg     # one-time
npm run tauri:dev                            # dev shell
# or
npm run tauri:build                          # signed/installable
```

Bundles land at `fovea/src-tauri/target/release/bundle/`:

| OS      | Output                              |
|---------|-------------------------------------|
| macOS   | `.app` + `.dmg`                     |
| Linux   | `.AppImage`, `.deb`, `.rpm`         |
| Windows | `.msi` + `.exe` (NSIS)              |

### Web (Caddy in front)

`docker compose up` from the repo root brings up the engine + Fovea
together. Fovea's container is a multi-stage build: Node builds the
SPA, Caddy serves it AND reverse-proxies `/api/*` to the engine.

| URL                     | What |
|-------------------------|------|
| `http://localhost:8080` | Fovea web UI (Caddy) |
| `http://localhost:6380` | Engine (direct, if you want raw `curl`) |

#### Optional Caddy auth

Set in `fovea/.env`:

```bash
AUTH_USER=op
AUTH_PASS_HASH='$2a$14$…'   # docker run --rm caddy:2-alpine caddy hash-password --plaintext '…'
```

This adds a second layer of auth at the Caddy edge — useful if you
want a public Fovea URL without exposing the engine to anonymous
traffic. The engine's user/password is **always** required regardless
(it's enforced inside the engine itself).

## First connection

1. Open Fovea (desktop or web). Land on the **Connections** screen.
2. Click **add**. Fill in:

   | Field                | Value                                  |
   |----------------------|----------------------------------------|
   | Label                | anything memorable (`Local`, `Prod`)  |
   | Engine URL           | `http://127.0.0.1:6380` (desktop) or `/api` (web) |
   | Username             | `admin` (or whatever you set via `HEATHER_ADMIN_USER`) |
   | Password             | (the password from first-boot or env) |
   | Active database      | `default` (or any DB you've created)  |
   | Role                 | `r/w`                                  |

3. Click **add**, then **connect**. Status dot in the top bar turns
   green when `/health` returns 200 with valid creds.

The connection (URL + username + password + active database) is
persisted via Tauri's plugin-store on desktop, or `localStorage` in
web mode.

## Screens

### Connections

- Add / edit / remove servers.
- Switch active connection.
- Per-row: URL, active database, scope role, username, last used.

### Collections

- Live list of every collection in the active database.
- Mini synthetic preview per tile.
- Filter by name.
- Click → Inspector.

### Inspector

- **Landscape** tab: pannable / zoomable canvas of the attractor
  population. Hover a dot for `id` + `weight`. Activated cells get
  accent halos.
- **Stats** tab: counts + sample latencies on demand.
- **Probe fingerprint** button: fires the collection's centroid
  through `/analyze`; the dominant attractors light up.
- Falls back to a synthetic golden-spiral landscape with a clear
  `synthetic landscape · server lacks /projection` badge when the
  server doesn't expose `/projection` (only the FastAPI proxy in
  `heatherdb-pi-demo` does, today).

### Read

- Pick a collection. Paste a vector (or use *fingerprint* / *random*
  buttons).
- Strategy toggle: `iterative` / `fast`.
- Submit → activation trace + iteration count + convergence.

### Algebra

- Pick A, op (`+` / `−`), B, optional target name.
- Hand-off buttons: open the result in Inspector or Read.

## Multi-database flow

The active database is per-connection. To explore a second DB:

- Edit the connection, change "Active database", reconnect; or
- Add a second connection pointing at the same URL with a different
  active DB.

The header chip shows the active DB in accent purple from any screen,
so you always know what you're operating against.

To **create** a database from Fovea: not yet — for now use
`POST /db` (see [api.md](./api.md#post-db)) or the engine CLI. A `/db`
management screen is on the v0.2 roadmap.

## Wire log

A toggleable right-rail panel shows every HTTP request — method, path,
status, latency. Latencies colour-coded:

| Range       | Colour |
|-------------|--------|
| < 10 ms     | green  |
| < 100 ms    | cyan   |
| < 1 s       | amber  |
| ≥ 1 s       | red    |

Useful when explaining the engine's speed to skeptics.

## Files + state

| Mode    | Where state lives |
|---------|-------------------|
| Desktop | Tauri's app-data dir (`~/Library/Application Support/co.aphorion.fovea/connections.json` on macOS; `~/.local/share/co.aphorion.fovea/` on Linux; `%APPDATA%\co.aphorion.fovea\` on Windows). |
| Web     | Browser `localStorage` under `fovea.connections.v1` + `fovea.active_connection.v1`. |

Passwords are stored in the same backend — there's no extra encryption
layer in Fovea. On desktop the file is owner-readable on Unix; in the
browser it's localStorage (visible in devtools). Don't store
production credentials in browser-mode Fovea — use the desktop app or
keep the browser tab on a trusted machine.

## What's not yet there

| Feature                                | Status |
|----------------------------------------|--------|
| Database management UI (create / drop) | Roadmap v0.2 |
| Server-stats screen                    | Roadmap v0.2 |
| Live write tail (SSE)                  | Roadmap v0.3 |
| Co-activation heatmap                  | Roadmap v0.3 |
| WebGL landscape (>50k attractors)      | Roadmap v0.3 |
| Encoder bridge (text/image → vector)   | Roadmap v0.3 |
| Multi-connection split view            | Roadmap v0.3 |
