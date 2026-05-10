# syntax=docker/dockerfile:1.7
#
# HeatherDB — production image
# ────────────────────────────
# Two stages:
#   1. `builder`  — full Rust toolchain, builds the release binary.
#   2. `runtime`  — distroless or debian-slim base, ~25 MB final image,
#                    runs as a non-root user, exposes port 6380.
#
# Build:
#   docker build -t heatherdb:latest .
#
# Run (with persistent data):
#   docker run -d --name heatherdb \
#     -p 6380:6380 \
#     -v heatherdb_data:/var/lib/heatherdb \
#     -e HEATHER_DIMENSION=128 \
#     heatherdb:latest
#
# Run (with config flags):
#   docker run --rm -p 6380:6380 -v $(pwd)/data:/var/lib/heatherdb \
#     heatherdb:latest --dimension 384 --port 6380

# ─── builder ──────────────────────────────────────────────────────────────────
FROM rust:1.81-bookworm AS builder

WORKDIR /src

# Cache deps separately — copy manifests first.
COPY Cargo.toml Cargo.lock ./
COPY heather_db/Cargo.toml      ./heather_db/
COPY heather_server/Cargo.toml  ./heather_server/
COPY heather_algebra/Cargo.toml ./heather_algebra/
COPY heather_cortex/Cargo.toml  ./heather_cortex/
COPY heather_fornix/Cargo.toml  ./heather_fornix/

# Stub mains so dep resolution succeeds before the real source lands.
RUN mkdir -p heather_db/src heather_server/src heather_algebra/src heather_cortex/src heather_fornix/src \
 && echo "fn main() {}"     > heather_server/src/main.rs \
 && for c in heather_db heather_algebra heather_cortex heather_fornix; do \
      echo "pub fn _stub() {}" > $c/src/lib.rs; \
    done \
 && cargo build --release -p heather_server || true

# Now the real source.
COPY heather_db       ./heather_db
COPY heather_server   ./heather_server
COPY heather_algebra  ./heather_algebra
COPY heather_cortex   ./heather_cortex
COPY heather_fornix   ./heather_fornix

# Force rebuild of the stub-replaced crates.
RUN touch heather_db/src/lib.rs heather_server/src/main.rs \
 && cargo build --release -p heather_server \
 && strip target/release/heather_server

# ─── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# LMDB needs no system libs at runtime — heed bundles it. We only pull in
# ca-certificates so HEATHER_FORNIX or future outbound calls work, and tini for
# correct signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user owns the data dir.
RUN groupadd --system --gid 65532 heatherdb \
 && useradd  --system --gid heatherdb --uid 65532 \
       --home-dir /var/lib/heatherdb --shell /usr/sbin/nologin heatherdb \
 && mkdir -p /var/lib/heatherdb \
 && chown -R heatherdb:heatherdb /var/lib/heatherdb

COPY --from=builder /src/target/release/heather_server /usr/local/bin/heather_server

ENV HEATHER_DATA_DIR=/var/lib/heatherdb \
    HEATHER_DIMENSION=128 \
    HEATHER_HOST=0.0.0.0 \
    HEATHER_PORT=6380 \
    HEATHER_MAP_SIZE_MB=4096 \
    HEATHER_REQUEST_TIMEOUT=600 \
    RUST_LOG=info

VOLUME ["/var/lib/heatherdb"]
EXPOSE 6380
USER heatherdb

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:6380/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/heather_server"]
