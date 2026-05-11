#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#
#                   ▄█    █▄       ▄████████  ▄██████▄  ▄██████▄
#                  ███    ███     ███    ███ ███    ███ ███    ███
#                  ███    ███     ███    █▀  ███    ███ ███    █▀
#                 ▄███▄▄▄▄███▄▄  ▄███▄▄▄     ███    ███ ███
#                ▀▀███▀▀▀▀███▀  ▀▀███▀▀▀     ███    ███ ███
#                  ███    ███     ███    █▄  ███    ███ ███    █▄
#                  ███    ███     ███    ███ ███    ███ ███    ███
#                  ███    █▀      ██████████  ▀██████▀   ▀██████▀
#
#                       deploy-vps.sh — five bucks, all your memories
#
#  One-shot HeatherDB deploy to a $5 VPS over SSH. Idempotent, opinionated,
#  unreasonably nice to look at. Does the boring stuff so you don't have to:
#
#    · Builds a release binary on the VPS (or uploads one you've prebuilt)
#    · Creates the heatherdb user, /opt/heatherdb tree, /etc/heatherdb/env
#    · Installs the systemd unit + enables auto-start on reboot
#    · Smoke-tests /health on port 6380 before declaring victory
#
#  Usage:
#
#    ./deploy-vps.sh user@vps.example.com           # build on the VPS
#    ./deploy-vps.sh user@vps.example.com --upload  # cross-build + scp
#
#  Env overrides (all optional):
#    REMOTE_USER       SSH user (default: parsed from arg)
#    REMOTE_HOST       SSH host (default: parsed from arg)
#    HEATHER_VERSION   git ref to check out on the VPS (default: main)
#    HEATHER_DIM       vector dimension (default: 128)
#    INSTALL_DIR       remote install dir (default: /opt/heatherdb)
#    SERVICE_USER      systemd user to run as (default: heatherdb)
#    SKIP_SMOKE_TEST   set to 1 to skip the /health check
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── colours ──────────────────────────────────────────────────────────────────
B="\033[1m"; D="\033[2m"; X="\033[0m"
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"
BLUE="\033[34m"; MAG="\033[35m"; CYAN="\033[36m"; PURPLE="\033[38;5;177m"

# ─── ascii art ────────────────────────────────────────────────────────────────
banner() {
  printf "${PURPLE}"
  cat <<'EOF'

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │      ░█░█░█▀▀░█▀█░▀█▀░█░█░█▀▀░█▀▄░█▀▄░█▀▄                       │
  │      ░█▀█░█▀▀░█▀█░░█░░█▀█░█▀▀░█▀▄░█░█░█▀▄                       │
  │      ░▀░▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀░░▀▀░                       │
  │                                                                 │
  │      memory that learns ⋅ at the price of a fancy sandwich      │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

EOF
  printf "${X}"
}

step()    { printf "\n${B}${BLUE}━━ %s${X}\n" "$*"; }
ok()      { printf "  ${GREEN}✓${X} %s\n" "$*"; }
warn()    { printf "  ${YELLOW}!${X} %s\n" "$*"; }
err()     { printf "  ${RED}✗${X} %s\n" "$*" >&2; }
say()     { printf "  ${D}%s${X}\n" "$*"; }
note()    { printf "  ${CYAN}∙${X} %s\n" "$*"; }
goofy()   { printf "  ${MAG}♪${X} ${D}%s${X}\n" "$*"; }

shrug() {
  cat <<'EOF'
                  ¯\_(ツ)_/¯
EOF
}

# Cute thinking animation while a long thing runs.
spinner() {
  local pid=$1 msg="$2"
  local frames=( '·    ' ' ·   ' '  ·  ' '   · ' '    ·' '   · ' '  ·  ' ' ·   ' )
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r  ${CYAN}[%s]${X} %s" "${frames[i % ${#frames[@]}]}" "$msg"
    i=$((i+1))
    sleep 0.12
  done
  printf "\r  ${GREEN}[ ✓ ]${X} %s\n" "$msg"
}

# ─── arg parsing ──────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
  banner
  cat <<EOF
${B}usage:${X}  $0 ${YELLOW}user@host${X} [--upload]

  No host given. The VPS, while affordable, is not telepathic. ${D}(yet.)${X}

EOF
  exit 1
fi

REMOTE="$1"
shift || true
UPLOAD_BINARY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --upload) UPLOAD_BINARY=1 ;;
    --help|-h)
      banner
      sed -n '/^# Usage:/,/^# ───/p' "$0" | sed 's/^# \?//' | head -n 18
      exit 0
      ;;
    *) err "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

REMOTE_USER="${REMOTE_USER:-${REMOTE%@*}}"
REMOTE_HOST="${REMOTE_HOST:-${REMOTE#*@}}"
HEATHER_VERSION="${HEATHER_VERSION:-main}"
HEATHER_DIM="${HEATHER_DIM:-128}"
INSTALL_DIR="${INSTALL_DIR:-/opt/heatherdb}"
SERVICE_USER="${SERVICE_USER:-heatherdb}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SYSTEMD_UNIT="$SCRIPT_DIR/systemd/heatherdb.service"

# ─── handshake ────────────────────────────────────────────────────────────────
banner

step "Handshake"
say "Remote   ${B}$REMOTE_USER${X}${D}@${X}${B}$REMOTE_HOST${X}"
say "Install  ${B}$INSTALL_DIR${X}"
say "Run as   ${B}$SERVICE_USER${X}"
say "Engine   ${B}heather_server${X}  (dim=$HEATHER_DIM, port=6380)"
[ "$UPLOAD_BINARY" -eq 1 ] && say "Binary   ${YELLOW}prebuild + upload${X}" || say "Binary   ${YELLOW}build on remote${X}  (ref=$HEATHER_VERSION)"
echo

note "Press Ctrl-C in the next 3 seconds to back out."
sleep 3 || true

# ─── ssh sanity ───────────────────────────────────────────────────────────────
step "SSH check — knocking politely"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE" 'true' 2>/dev/null; then
  err "Couldn't reach $REMOTE non-interactively. Got keys set up?"
  goofy "(if you've never ssh'd here before, do that first — your VPS needs to know you mean it)"
  exit 1
fi
ok "SSH works. The VPS waved back."

# ─── remote env probe ─────────────────────────────────────────────────────────
step "Asking the VPS what it's working with"
remote_info="$(ssh "$REMOTE" '
  set -e
  . /etc/os-release 2>/dev/null || true
  echo "OS:${PRETTY_NAME:-unknown}"
  echo "ARCH:$(uname -m)"
  echo "RAM:$(awk "/MemTotal/ {printf \"%.1f GB\", \$2/1024/1024}" /proc/meminfo 2>/dev/null || echo unknown)"
  echo "DISK:$(df -h / | awk "NR==2 {print \$4 \" free of \" \$2}")"
  echo "RUST:$(command -v cargo >/dev/null && cargo --version || echo missing)"
  echo "SUDO:$(sudo -n true 2>/dev/null && echo passwordless || echo password-required)"
')"

while IFS=: read -r k v; do
  printf "  ${D}%-6s${X} %s\n" "$k" "$v"
done <<< "$remote_info"

REMOTE_ARCH="$(echo "$remote_info" | awk -F: '/^ARCH:/{print $2}')"
HAS_RUST="$(echo "$remote_info" | awk -F: '/^RUST:/{print $2}')"

# ─── bootstrap: user, dirs, env file ──────────────────────────────────────────
step "Carving out a home for HeatherDB"
ssh "$REMOTE" "sudo bash -s" <<EOF
set -e
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" "$SERVICE_USER"
fi
mkdir -p "$INSTALL_DIR/data" /etc/heatherdb
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Defaults — only created if not already there. Edit /etc/heatherdb/env to override.
if [ ! -f /etc/heatherdb/env ]; then
  cat > /etc/heatherdb/env <<ENV
HEATHER_DATA_DIR=$INSTALL_DIR/data
HEATHER_DIMENSION=$HEATHER_DIM
HEATHER_PORT=6380
HEATHER_HOST=0.0.0.0
HEATHER_MAP_SIZE_MB=4096
HEATHER_REQUEST_TIMEOUT=600
RUST_LOG=info
ENV
  chmod 0644 /etc/heatherdb/env
fi
EOF
ok "User ${B}$SERVICE_USER${X} ready, dirs created, env file in place."

# ─── binary: build remotely OR upload ─────────────────────────────────────────
if [ "$UPLOAD_BINARY" -eq 1 ]; then
  step "Cross-building + scp'ing the binary"
  TARGET="x86_64-unknown-linux-gnu"
  [[ "$REMOTE_ARCH" == "aarch64"* || "$REMOTE_ARCH" == "arm64"* ]] && TARGET="aarch64-unknown-linux-gnu"

  say "remote arch: $REMOTE_ARCH → target: $TARGET"
  if ! command -v cross >/dev/null; then
    err "'cross' not found locally. Install with: cargo install cross"
    exit 1
  fi
  ( cd "$REPO_ROOT" && cross build --release --target "$TARGET" -p heather_server ) &
  spinner $! "compiling release binary (this is the slow step)"

  scp "$REPO_ROOT/target/$TARGET/release/heather_server" "$REMOTE:/tmp/heather_server" >/dev/null
  ssh "$REMOTE" "sudo install -m 0755 -o $SERVICE_USER -g $SERVICE_USER /tmp/heather_server $INSTALL_DIR/heather_server && rm /tmp/heather_server"
  ok "Binary uploaded."
else
  step "Building on the remote"
  if [ "$HAS_RUST" = "missing" ]; then
    warn "Rust isn't installed on the VPS. Installing rustup non-interactively."
    ssh "$REMOTE" 'curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null 2>&1' &
    spinner $! "installing rustup (one-time, ~3 min)"
  fi

  ssh "$REMOTE" "
    set -e
    source \$HOME/.cargo/env 2>/dev/null || true
    if [ ! -d /tmp/heatherdb-src ]; then
      git clone --depth 1 --branch '$HEATHER_VERSION' https://github.com/aphorion/heather-db /tmp/heatherdb-src
    else
      ( cd /tmp/heatherdb-src && git fetch --depth 1 origin '$HEATHER_VERSION' && git reset --hard FETCH_HEAD )
    fi
    cd /tmp/heatherdb-src && cargo build --release -p heather_server >/dev/null
    sudo install -m 0755 -o '$SERVICE_USER' -g '$SERVICE_USER' \
      target/release/heather_server '$INSTALL_DIR/heather_server'
  " &
  spinner $! "building heather_server on the VPS (this is the slow step)"
  ok "Built and installed."
fi

# ─── systemd unit ─────────────────────────────────────────────────────────────
step "Wiring up the systemd unit"
if [ ! -f "$SYSTEMD_UNIT" ]; then
  err "Couldn't find $SYSTEMD_UNIT"
  exit 1
fi
scp "$SYSTEMD_UNIT" "$REMOTE:/tmp/heatherdb.service" >/dev/null
ssh "$REMOTE" "
  sudo install -m 0644 -o root -g root /tmp/heatherdb.service /etc/systemd/system/heatherdb.service
  rm /tmp/heatherdb.service
  sudo systemctl daemon-reload
  sudo systemctl enable heatherdb >/dev/null 2>&1
  sudo systemctl restart heatherdb
"
ok "systemctl restart heatherdb — unit is enabled, boot-persistent, and running."

# ─── smoke test ───────────────────────────────────────────────────────────────
if [ "$SKIP_SMOKE_TEST" = "1" ]; then
  warn "Skipping smoke test (SKIP_SMOKE_TEST=1)"
else
  step "Smoke test — does the database remember anything?"
  for try in 1 2 3 4 5; do
    if ssh "$REMOTE" "curl -fsS http://127.0.0.1:6380/health" >/tmp/heatherdb_health 2>/dev/null; then
      ok "  /health responded:"
      sed 's/^/      /' /tmp/heatherdb_health
      rm -f /tmp/heatherdb_health
      break
    fi
    say "attempt $try/5 — waiting for the server to wake up…"
    sleep 1
    if [ "$try" = 5 ]; then
      err "/health never responded. Logs:"
      ssh "$REMOTE" "sudo journalctl -u heatherdb -n 30 --no-pager" | sed 's/^/      /'
      goofy "the database is having a moment. give it a minute then re-run, or check the logs above."
      exit 1
    fi
  done
fi

# ─── confetti ─────────────────────────────────────────────────────────────────
echo
printf "${GREEN}"
cat <<'EOF'

       ╔══════════════════════════════════════════════════════════════╗
       ║                                                              ║
       ║            DEPLOYED.  Memory is online.                      ║
       ║                                                              ║
       ║                .・゜゜・✧・゜゜・.                            ║
       ║                                                              ║
       ╚══════════════════════════════════════════════════════════════╝

EOF
printf "${X}"

echo
note "Engine reachable on the VPS at:"
echo "      ${B}http://${REMOTE_HOST}:6380${X}"
echo
note "Quick sanity:"
echo "      ${D}curl http://${REMOTE_HOST}:6380/health${X}"
echo "      ${D}curl http://${REMOTE_HOST}:6380/stats${X}"
echo
note "Logs:"
echo "      ${D}ssh ${REMOTE} 'journalctl -u heatherdb -f'${X}"
echo
note "Tweak config at:  ${B}/etc/heatherdb/env${X}  then  ${D}systemctl restart heatherdb${X}"
echo
note "Want HTTPS? Drop a Caddy or nginx reverse proxy in front. Two lines."
echo
goofy "go bother it with some vectors. it has been waiting."
echo
