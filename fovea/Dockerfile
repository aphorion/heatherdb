# syntax=docker/dockerfile:1.7
#
# Fovea — web build + Caddy + (optional) HTTP basic auth
# ──────────────────────────────────────────────────────
# Two stages:
#   1. node:20-alpine builds the React + Vite SPA into /dist.
#   2. caddy:2-alpine serves it AND reverse-proxies /api/* to the
#      heatherdb container so the SPA never sees CORS.
#
# Auth is opt-in via AUTH_USER + AUTH_PASS_HASH env vars (loaded from
# fovea/.env when running via docker compose). When both are set,
# everything (including /api/*) sits behind HTTP Basic Auth. When
# neither is set the entrypoint runs the no-auth Caddyfile and emits
# a loud warning. Setting only one is treated as a config error.
#
# Build:
#   docker build -t fovea:latest fovea/
#
# Run standalone (with auth):
#   docker run -p 8080:80 \
#     -e HEATHER_HOST=heatherdb:6380 \
#     -e AUTH_USER=op \
#     -e AUTH_PASS_HASH='$2a$14$…'   # bcrypt — see README
#     fovea:latest

# ─── builder ──────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /src

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY tsconfig.json tsconfig.node.json vite.config.ts tailwind.config.ts postcss.config.js index.html ./
COPY src ./src

# Build the SPA — tsc -b runs first to fail fast on type errors.
RUN npx tsc -b && npx vite build

# ─── runtime ──────────────────────────────────────────────────────────────────
FROM caddy:2-alpine AS runtime

ENV HEATHER_HOST=heatherdb:6380

COPY --from=builder /src/dist /srv

# Two Caddyfile variants — entrypoint picks the right one based on env.

COPY <<'EOF' /etc/caddy/Caddyfile.auth
{
    auto_https off
    admin off
}
:80 {
    log { output stdout; format console }
    encode zstd gzip

    basicauth /* {
        {$AUTH_USER} {$AUTH_PASS_HASH}
    }

    handle /api/* {
        uri strip_prefix /api
        reverse_proxy {$HEATHER_HOST}
    }

    handle {
        root * /srv
        try_files {path} /index.html
        file_server
    }
}
EOF

COPY <<'EOF' /etc/caddy/Caddyfile.noauth
{
    auto_https off
    admin off
}
:80 {
    log { output stdout; format console }
    encode zstd gzip

    handle /api/* {
        uri strip_prefix /api
        reverse_proxy {$HEATHER_HOST}
    }

    handle {
        root * /srv
        try_files {path} /index.html
        file_server
    }
}
EOF

# Tiny entrypoint: pick a Caddyfile based on whether auth env vars are set,
# fail loudly on partial config, exec into Caddy.
COPY <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/sh
set -eu

if [ -n "${AUTH_USER:-}" ] && [ -n "${AUTH_PASS_HASH:-}" ]; then
  echo "fovea ⋅ basic auth enabled (user: ${AUTH_USER})"
  cp /etc/caddy/Caddyfile.auth /etc/caddy/Caddyfile
elif [ -z "${AUTH_USER:-}" ] && [ -z "${AUTH_PASS_HASH:-}" ]; then
  cat <<'WARN'
┌──────────────────────────────────────────────────────────────────────────┐
│  fovea ⋅ WARNING — running without auth.                                │
│                                                                          │
│  Anyone who can reach this container's port can use the inspector AND   │
│  hit /api/* on the engine. Set AUTH_USER + AUTH_PASS_HASH to enable     │
│  HTTP Basic Auth — see fovea/README.md for the hash command.            │
└──────────────────────────────────────────────────────────────────────────┘
WARN
  cp /etc/caddy/Caddyfile.noauth /etc/caddy/Caddyfile
else
  echo "fovea ⋅ ERROR — AUTH_USER and AUTH_PASS_HASH must be set together." >&2
  echo "fovea ⋅       Got one but not the other; refusing to start." >&2
  exit 1
fi

exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 80
CMD ["/usr/local/bin/entrypoint.sh"]
