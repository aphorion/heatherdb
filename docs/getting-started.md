# Getting started

By the end of this page HeatherDB is running on your machine, you have written
a vector into it, and you have read a corrupted version of that vector back.
Five minutes, most of it downloading.

HeatherDB comes in two shapes and this page covers both:

- **Server** — one binary listening on port 6380, spoken to over HTTP. This is
  what the tutorials use, and what the rest of this page installs.
- **Embedded** — the engine as a Rust library inside your own process, no
  server and no network. [Skip to embedded](#running-it-embedded) if that is
  what you came for.

## 1. Install it

Download the binary for your platform. It is a single self-contained
executable — no runtime, no system dependency beyond libc, nothing to
configure before it runs.

Every build, with its size and SHA-256, is listed on the
[downloads page](https://heather.aphorion.co/download); the archives
themselves are served from
[GitHub releases](https://github.com/aphorion/heatherdb/releases).

<!--tabs:os-->
<!--tab:macOS-->

```bash
curl -fsSL https://heather.aphorion.co/install.sh | sh
```

The script detects Apple Silicon or Intel, resolves the latest release,
verifies its SHA-256 against the checksum manifest published with it, and puts
`heather` in `/usr/local/bin` — or `~/.local/bin` when it cannot escalate, and
it says so. `HEATHER_VERSION` and `HEATHER_INSTALL_DIR` override both choices.

To do it by hand, take the archive from the release. Asset names carry the
version, so set it once — the [downloads page](https://heather.aphorion.co/download)
lists the current one:

```bash
VERSION=v0.3.0
curl -fsSLO https://github.com/aphorion/heatherdb/releases/download/$VERSION/heather-$VERSION-macos-aarch64.tar.gz
tar -xzf heather-$VERSION-macos-aarch64.tar.gz
sudo cp heather-$VERSION-macos-aarch64/heather /usr/local/bin/
```

Swap `aarch64` for `x86_64` on an Intel Mac. The builds are not codesigned, so
Gatekeeper holds the first run until you clear the quarantine attribute:

```bash
xattr -d com.apple.quarantine /usr/local/bin/heather
heather --version
```

<!--tab:Linux-->

```bash
curl -fsSL https://heather.aphorion.co/install.sh | sh
```

The script picks the right architecture and libc, checks the SHA-256, and
installs to `/usr/local/bin` (or `~/.local/bin` without sudo). By hand:

```bash
VERSION=v0.3.0
curl -fsSLO https://github.com/aphorion/heatherdb/releases/download/$VERSION/heather-$VERSION-linux-x86_64-gnu.tar.gz
tar -xzf heather-$VERSION-linux-x86_64-gnu.tar.gz
sudo cp heather-$VERSION-linux-x86_64-gnu/heather /usr/local/bin/
```

Swap `x86_64` for `aarch64` on ARM, and `gnu` for `musl` on Alpine or anywhere
glibc is old — the musl builds are static.

On Debian or Ubuntu the `.deb` is the better path: it brings a systemd
service, a dedicated system user, and admin credentials generated into
`/etc/heatherdb/env` on first install.

```bash
VERSION=v0.3.0
curl -fsSLO https://github.com/aphorion/heatherdb/releases/download/$VERSION/heatherdb_${VERSION#v}_amd64.deb
sudo apt install ./heatherdb_${VERSION#v}_amd64.deb
```

```bash
heather --version
```

<!--tab:Windows-->

```powershell
irm https://heather.aphorion.co/install.ps1 | iex
```

The script downloads the x64 build, verifies its hash, extracts it to
`%LOCALAPPDATA%\Heather`, and appends that to your user `PATH` — reopen the
terminal before running `heather`, since a `PATH` change only reaches
processes started after it.

By hand, take the zip from the
[downloads page](https://heather.aphorion.co/download), extract it, and put
`heather.exe` somewhere on your `PATH`:

```powershell
$VERSION = "v0.3.0"
Invoke-WebRequest "https://github.com/aphorion/heatherdb/releases/download/$VERSION/heather-$VERSION-windows-x86_64.zip" -OutFile heather.zip
Expand-Archive heather.zip -DestinationPath $env:LOCALAPPDATA\Heather
$env:PATH += ";$env:LOCALAPPDATA\Heather"
```

```powershell
heather --version
```

<!--tab:From source-->

Any platform with a Rust toolchain, 1.88 or newer. This is what you want if
you are changing the engine, or on a platform we do not ship a build for.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/aphorion/heatherdb
cd heather-db
cargo build --release -p heather_server
```

The first build takes a few minutes and produces one binary,
`target/release/heather`. Put it on your `PATH`:

```bash
sudo cp target/release/heather /usr/local/bin/
```

On Windows the binary lands at `target\release\heather.exe`, and Rustup will
prompt you to install the Visual Studio Build Tools (the MSVC toolchain)
first.

<!--/tabs-->

## 2. Start it

The engine needs three things: somewhere to keep data, and an admin username
and password to mint on first boot.

<!--tabs:os-->
<!--tab:macOS-->

```bash
HEATHER_DATA_DIR=/tmp/heather-quickstart \
HEATHER_DIMENSION=8 \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='quickstart-password' \
  heather
```

<!--tab:Linux-->

```bash
HEATHER_DATA_DIR=/tmp/heather-quickstart \
HEATHER_DIMENSION=8 \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='quickstart-password' \
  heather
```

If you installed the `.deb`, the service is already running under systemd
instead — `sudo systemctl status heatherdb`, with the generated password in
`/etc/heatherdb/env`.

<!--tab:Windows-->

```powershell
$env:HEATHER_DATA_DIR = "$env:TEMP\heather-quickstart"
$env:HEATHER_DIMENSION = "8"
$env:HEATHER_ADMIN_USER = "admin"
$env:HEATHER_ADMIN_PASSWORD = "quickstart-password"
heather
```

<!--tab:From source-->

Run the binary you just built, from the repository root:

```bash
HEATHER_DATA_DIR=/tmp/heather-quickstart \
HEATHER_DIMENSION=8 \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='quickstart-password' \
  ./target/release/heather
```

<!--/tabs-->

The log tells you three things:

```
INFO heather: Opened server path=/tmp/heather-quickstart databases=1 names=["default"]
INFO heather::auth: Auth: bootstrapped admin user from HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD
INFO heather: HeatherDB server listening addr=0.0.0.0:6380
```

A `default` database was created, an `admin` user was minted from the two
environment variables, and the engine is listening on 6380.

`HEATHER_DIMENSION=8` sets the vector width of that `default` database so the
vectors below are short enough to type. It is fixed for the life of a database
— see [Tune a database](how-to/tune-a-database.md) for how to pick a real one.
**The admin password must be at least 8 characters.** A shorter one stops the
engine at boot rather than being accepted quietly:

```
error: bootstrap admin user: password must be at least 8 characters
```

Leave `HEATHER_ADMIN_PASSWORD` unset in production and the engine generates a
random 24-character password, prints it once on stderr, and writes it to
`$HEATHER_DATA_DIR/initial-admin-password`.

Leave that terminal running and open a second one for the rest.

**There is no `--daemon` flag, and that is deliberate.** The engine runs in the
foreground and logs to stdout, because every supervisor you would actually use
prefers it that way: systemd wants `Type=simple`, and a container needs the
engine to *be* the process rather than fork away from it. A process that
daemonises itself hides its exit code, loses its logs, and forces the
supervisor to chase a PID file.

To run it in the background, use the thing that is already watching it:

| Where | How |
|---|---|
| A server | the `.deb`, which installs a systemd unit — `systemctl start heatherdb` |
| A container | `docker run -d`, as below |
| Your laptop, briefly | `heather … > heather.log 2>&1 &` |

The last one is fine for a tutorial and wrong for anything you care about,
because nothing restarts it. See [Run it in
production](how-to/production.md).

### Or run it in a container

If you would rather not install anything, the image is the same engine and the
rest of this page works unchanged against it.

```bash
docker run -d --name heatherdb \
  -p 6380:6380 \
  -v heatherdb_data:/var/lib/heatherdb \
  -e HEATHER_DIMENSION=8 \
  -e HEATHER_ADMIN_USER=admin \
  -e HEATHER_ADMIN_PASSWORD='quickstart-password' \
  aphorion/heatherdb:latest
```

Images are multi-arch (`linux/amd64` and `linux/arm64`) and signed with cosign
keyless. Without the named volume the data directory dies with the container.
Follow the log with `docker logs -f heatherdb`, and see
[Deploy HeatherDB](how-to/deploy.md#docker) for the verification command and
the production flags.

## 3. Check it is alive

```bash
curl http://localhost:6380/health
```
```json
{"status":"ok"}
```

`/health` is the only route that needs no credentials. Everything else does:

```bash
curl -s -u admin:quickstart-password http://localhost:6380/collections
```
```json
{"collections":[]}
```

No collections yet — nothing has been written.

## 4. Write something, then read it back wrong

Write one vector into a collection called `quickstart`. The collection is
created on first write; you do not declare it.

```bash
curl -s -u admin:quickstart-password \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[1, 1, 1, 1, 0, 0, 0, 0]]}' \
  http://localhost:6380/collections/quickstart/write
```
```json
{"count":1}
```

Now ask for it with a query that is *wrong* — half the components zeroed, one
of them flipped to the opposite sign:

```bash
curl -s -u admin:quickstart-password \
  -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, -1, 0, 0, 0, 0]}' \
  http://localhost:6380/collections/quickstart/read
```
```json
{"result":[0.4999,0.4999,0.4999,0.4999,0.0,0.0,0.0,0.0]}
```

The exact numbers will differ. What matters is the shape: you asked with a
mangled query and the engine returned the pattern that query belongs to, not
the query itself and not "no match". That is a read in this database —
a reconstruction rather than a lookup.

If you want to know how sure it was, compare the two:

```bash
python3 - <<'PY'
import math
a = [1, 1, 1, 1, 0, 0, 0, 0]
b = [0.4999, 0.4999, 0.4999, 0.4999, 0.0, 0.0, 0.0, 0.0]   # paste your result
dot = sum(x * y for x, y in zip(a, b))
print(dot / (math.hypot(*a) * math.hypot(*b)))
PY
```

That number is [fidelity](terms/fidelity.md), and it is what tells a
remembered pattern from a stranger. [Your first
memory](tutorials/first-memory.md) does this properly, with 200 vectors and a
query corrupted past recognition.

## Running it embedded

The server is a thin HTTP layer over a library. If your program is written in
Rust you can skip the network entirely and open the memory in your own process
— same engine, same on-disk format, no port and no credentials.

<!--tabs:shape-->
<!--tab:Embedded-->

`Cargo.toml`:

```toml
[dependencies]
heather_db = { git = "https://github.com/aphorion/heatherdb" }
```

`src/main.rs`:

```rust
use std::path::Path;
use heather_db::{EAMConfig, Hive, ReadStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One LMDB environment, many collections. 8 dimensions, 256 MB map size.
    let hive = Hive::open(Path::new("/tmp/heather-embedded"), EAMConfig::new(8)?, 256)?;
    let col = hive.get_or_create_collection("quickstart")?;

    col.write(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])?;

    let query = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
    let result = col.read(&query, ReadStrategy::HopfieldIter)?;

    println!("{result:?}");
    Ok(())
}
```

`Hive` and `Collection` are `Send + Sync` and every method takes `&self`, so
one hive can be shared across threads — reads take a shared lock, writes an
exclusive one. The data directory is interchangeable with the server's: point
`heather --data-dir` at it later and the same collections are there.

<!--tab:Server-->

What the rest of this page does. One binary, HTTP on port 6380, any language
that can make a request:

```bash
curl -s -u admin:quickstart-password \
  -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, -1, 0, 0, 0, 0]}' \
  http://localhost:6380/collections/quickstart/read
```

Pick this when more than one process needs the memory, when the callers are
not written in Rust, or when you want the operator surface — users and scopes,
backups, snapshots, the audit log — that the binary brings with it.

<!--/tabs-->

## Clean up

```bash
rm -rf /tmp/heather-quickstart          # or: docker rm -f heatherdb
```

Nothing else was installed outside the binary you copied.

## Where to go next

- [Your first memory](tutorials/first-memory.md) — the tutorial proper (stop
  the quickstart server first; the lesson starts its own on the same port):
  200 vectors, a query corrupted past recognition, and the fidelity signal
  used to tell a known pattern from an unknown one.
- [Deploy HeatherDB](how-to/deploy.md) — Compose, `.deb`, systemd, Kubernetes,
  and what to do before you expose it.
- [Configuration](reference/configuration.md) — every flag, environment
  variable, and file the engine reads.
- [Manage users and authentication](how-to/manage-users.md) — scoped users and
  password rotation, before anyone else touches it.
