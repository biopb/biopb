# biopb `web/` — the browser front end

A self-contained pnpm workspace holding **all** of biopb's browser UI as one
Vite + React single-page app. The **control plane** (`biopb-control`, port 8813)
is the single web origin: it serves this bundle at its root and proxies the data
plane and each MCP session. There is no build-time namespacing — the app is
always built with base `/`.

For the front-end internals — the `@biopb/tensor-flight-client` data-plane SDK,
axis mapping / `computeScaleHint`, the token/store model, and the slice-render
data flow — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```
web/
  pnpm-workspace.yaml          # this dir is its own workspace (own pnpm-lock.yaml)
  package.json                 # workspace root scripts (dev/build/test/lint/sync-version)
  tsconfig.base.json           # shared TS compiler options
  scripts/sync-version.js      # stamps the JS packages from the tensor server's _version.py
  packages/
    app/                       # @biopb/web — the SPA (Vite + React + React Router + Zustand)
    tensor-flight-client/      # @biopb/tensor-flight-client — browser Arrow-Flight/HTTP SDK
```

## The surfaces (one SPA, base `/`)

The control serves `index.html` for any non-API, non-proxy GET, so every surface
is a client route of the same bundle:

| Route (served by control)      | Component        | Talks to                              |
|--------------------------------|------------------|---------------------------------------|
| `/`                            | `DashboardPage`  | control's own `/api/*`                 |
| `/viewer`, `/admin`, `/unlock` | viewer + admin   | `/data_plane/*` (proxied to sidecar)  |
| `/mcp/admin`                   | `McpAdminPage`   | control's `/api/mcp_config`            |
| `/session/:id/observe`         | `ObservePage`    | `/session/:id/api/*` (proxied to child)|

`main.tsx` wires the routes; the dashboard and observe pages are `React.lazy`
chunks so the observe shell doesn't pull the heavy Pixi/Arrow viewer. Assets are
requested from the absolute root (`/assets/*`), so they resolve no matter which
prefix (`/`, `/viewer`, `/session/<id>/observe`) the shell was served under.

## Develop

```sh
pnpm install                     # in web/
pnpm dev                         # Vite dev server (HMR) on :5173
```

`pnpm dev` proxies `/api`, `/data_plane` (incl. the `/data_plane/ws/render`
websocket), and `/session` to a live control on `http://localhost:8813` (start
one with `biopb control start`). In dev the viewer defaults its data plane to the
proxied `/data_plane`, so plain `pnpm dev` renders end-to-end against that control
— no env var needed. Override with `VITE_TENSOR_API=<url> pnpm dev` only to point
the viewer at a standalone sidecar instead.

## Build

```sh
pnpm build                       # tensor-flight-client, then @biopb/web -> packages/app/dist
```

`VITE_TENSOR_API=/data_plane` is the only build-time env the viewer needs (it
points the viewer at the control-proxied data plane). The control serves the
resulting `packages/app/dist`; point it there with
`biopb control start --static-dir <repo>/web/packages/app/dist` (the default is
the installed bundle at `~/.local/share/biopb/webapp`). CI tars `packages/app/dist`
into `webapp.tar.gz`, which the installer unpacks to that default location and the
Docker image copies to `/app/webapp`.

## Test / lint

```sh
pnpm test                        # vitest (tensor-flight-client)
pnpm lint                        # eslint (@biopb/web)
```
