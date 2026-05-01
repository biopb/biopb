# `@biopb/web` Architecture

This document describes the architecture of the `packages/web` Next.js application, including runtime flow, authentication gating, and environment variables.

## Overview

`@biopb/web` is a Next.js frontend for the BioPB TensorFlight experience.

Key responsibilities:
- serve the browser UI
- enforce website access controls
- expose a protected `/api/token` route for the browser to obtain auth info
- initialize the client-side TensorFlight client
- proxy the frontend to the backend Arrow Flight/HTTP sidecar via client code

The web app does not host the tensor data service itself. It uses the FastAPI sidecar hosted at `BIOPB_TENSOR_API` (default `http://localhost:8816`).

## Main components

### 1. Next.js app

- `package.json`
  - `dev`: `next dev`
  - `build`: `next build`
  - `start`: `next start`
- `next.config.mjs`
  - enables `reactStrictMode`
  - transpiles `@biopb/tensor-flight-client`

### 2. Middleware (`middleware.ts`)

The middleware implements the website token gate and route access rules.

Routes that bypass auth:
- `/livez`
- `/readyz`
- `/healthz`
- `/_next/*`
- `/favicon.ico`
- `/unlock`

For all other routes, the middleware requires either:
- a valid session cookie named `biopb_session`, or
- a `?token=...` URL parameter matching `BIOPB_WEB_TOKEN`

When token validation succeeds via URL param, the middleware sets the session cookie and redirects to the clean URL.

### 3. Auth / session token endpoint (`app/api/token/route.ts`)

This route returns JSON for the browser with:
- `token` — the auth token for direct FastAPI sidecar calls
- `apiBase` — the FastAPI sidecar base URL
- `devMode` — whether web dev-mode bypass is active

The browser fetches this from `ClientBootstrap`.

### 4. Client bootstrap (`app/ClientBootstrap.tsx`)

On first mount, the client:
1. fetches `/api/token`
2. initializes the shared `AppStore` with `apiBase`, `token`, and `devMode`
3. loads available sources from the backend

This makes the UI fully dynamic and driven by the authenticated session state.

### 5. Global app state (`app/store.ts`)

Uses `zustand` to maintain:
- `TensorFlightClient` instance
- connection state
- `apiBase`
- `devMode`
- available sources
- active source and tensor selection
- slice controls
- session management

The store also supports clearing the session cookie and redirecting to `/unlock`.

### 6. Tensor client library

`@biopb/web` depends on `@biopb/tensor-flight-client`.

That package provides the browser-side HTTP client for the FastAPI sidecar, including:
- `listSources()`
- `getSource()` / metadata
- `slice()` / chunk fetch
- diagnostics and health checks

## Runtime flow

1. Browser requests the web app.
2. Next.js middleware checks auth.
3. If allowed, page renders and client bootstrap runs.
4. Browser fetches `/api/token`.
5. `/api/token` responds with `apiBase`, `token`, and `devMode`.
6. `ClientBootstrap` initializes the `TensorFlightClient`.
7. The app loads available tensor sources and renders the UI.
8. User interactions cause the frontend to call the FastAPI sidecar via `TensorFlightClient`.

## Environment variables

### `NEXT_PUBLIC_DEV_MODE`
- Type: boolean string (`"true"` / not set)
- Purpose: disables the web token gate in the browser app when the app is built in dev bypass mode
- When enabled:
  - `/api/token` returns `token: null`
  - middleware skips auth enforcement
- This should only be enabled when the server is bound to localhost and `BIOPB_WEB_DEV_BYPASS` is also enabled in the backend stack.

### `BIOPB_WEB_TOKEN`
- Type: string
- Purpose: website access token used by `middleware.ts`
- If not set, the middleware allows access and logs a warning.
- Used to validate the `biopb_session` cookie or the `?token=` query parameter.

### `BIOPB_TENSOR_API`
- Type: URL string
- Purpose: backend FastAPI sidecar base URL
- Default: `http://localhost:8816`
- Exposed to the browser via `/api/token`

## Security model

- The web app protects browser access with a session cookie and token gate.
- `/api/token` is only reachable after passing middleware auth.
- The browser uses the returned token to call the FastAPI sidecar directly.
- Dev mode bypass is intentionally restricted to localhost only.

## Notes

- `@biopb/web` is decoupled from the Python backend, so the frontend can stay running while the backend restarts.
- The app uses the HTTP sidecar endpoint at runtime, not server-side proxying of tensor data.

## Common commands

From repo root:
```bash
pnpm --filter @biopb/web dev
pnpm --filter @biopb/web build
pnpm --filter @biopb/web start
```

## File map

- `packages/web/package.json`
- `packages/web/next.config.mjs`
- `packages/web/middleware.ts`
- `packages/web/app/ClientBootstrap.tsx`
- `packages/web/app/api/token/route.ts`
- `packages/web/app/store.ts`

