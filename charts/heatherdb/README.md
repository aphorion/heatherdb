# heatherdb (Helm chart)

Deploys HeatherDB — a single-writer engine (embedded LMDB) — as a
one-replica `StatefulSet` with persistent storage, a headless Service
(for the StatefulSet), a client-facing `ClusterIP` Service, and optional
Ingress. Mirrors the same `docker run`/`docker-compose` contract as the
image itself: env-var configuration, `/health` for probes, auth on by
default.

Like Postgres, this does **not** horizontally scale by adding replicas
against the same volume — the chart always deploys exactly one pod.
There is no `replicaCount` value.

## Install

```sh
# Private image (GHCR) — create a pull secret first:
kubectl create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username=<gh-username> \
  --docker-password=<gh-pat-with-read:packages>

helm install heatherdb charts/heatherdb \
  --set imagePullSecrets[0].name=ghcr-pull-secret
```

Once the image is also published to Docker Hub (`aphorion/heatherdb`),
point `image.repository` there instead if you don't want to manage a
pull secret (once that repo is public) or to use Docker Hub credentials
in its place.

## Admin credentials

Three ways, in order of preference:

1. **Bring your own Secret** (recommended — keeps the password out of
   Helm's release history):

   ```sh
   kubectl create secret generic heatherdb-auth \
     --from-literal=HEATHER_ADMIN_USER=admin \
     --from-literal=HEATHER_ADMIN_PASSWORD='...'
   helm install heatherdb charts/heatherdb --set auth.existingSecret=heatherdb-auth
   ```

2. **Let the chart create one from values** (`auth.username`/
   `auth.password`) — fine for local/dev clusters, but the password
   lands in `helm get values` and release history in plaintext.

3. **Leave both unset** — the engine mints a random admin password on
   first boot and logs it once:

   ```sh
   kubectl logs heatherdb-0 | grep -A8 first-boot
   kubectl exec heatherdb-0 -- heather user passwd admin --password '<new>'
   ```

`auth.disabled=true` sets `HEATHER_AUTH_DISABLED=1` — dev-only, never
on a cluster reachable outside a trusted network.

## Values

See `values.yaml` for the full set (image, service, persistence,
resources, env tunables, auth, ingress, probes, scheduling). Key ones:

| Key | Default | Notes |
|---|---|---|
| `image.repository` | `ghcr.io/aphorion/heather-db` | switch to `aphorion/heatherdb` once on Docker Hub |
| `persistence.size` | `10Gi` | LMDB data dir (`/var/lib/heatherdb`) |
| `env.mapSizeMb` | `4096` | LMDB mmap ceiling — keep `resources.limits.memory` above this |
| `env.dimension` | `128` | vector dimension; immutable per-DB once data exists |

## Backups

`heather backup create`/`snapshot create` are subcommands of the same
binary — run them via `kubectl exec`:

```sh
kubectl exec heatherdb-0 -- heather --data-dir /var/lib/heatherdb \
  snapshot create --db default --output /var/lib/heatherdb/snapshots
```

`snapshot create` is safe against a running engine (live LMDB copy).
Copy the result out with `kubectl cp` before it's needed, since it
lives on the same PVC.

## Uninstall

```sh
helm uninstall heatherdb
```

The PVC created by the StatefulSet's `volumeClaimTemplate` is **not**
deleted by `helm uninstall` (standard Helm/StatefulSet behavior) —
remove it explicitly once you're done with the data:

```sh
kubectl delete pvc data-heatherdb-0
```
