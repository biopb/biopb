# biopb-tensor-server agent guide

## Role and layering

This project is the data plane: an Arrow Flight server plus an optional FastAPI
HTTP sidecar. Read `ARCHITECTURE.md` before changing adapters, discovery,
caching, lifecycle, wire geometry, or security.

Respect the package layers: `core/` defines contracts and stores; `serving/`
implements Flight/HTTP runtime; `sources/` owns discovery/reconciliation;
`adapters/` and `cache/` implement formats and chunk storage. Route every source
through the registry/claim pipeline rather than creating a parallel path.

## Data invariants

- The wire axis guarantee is canonical trailing `Z, Y, X, S`; other axes retain
  relative order before them. Apply normalization at the adapter/registry seam.
  Ambiguous axes degrade to identity, and writable or remote-owned
  noncanonical orders are refused rather than guessed.
- Keep adapter claims ordered most-specific first (for example OME-TIFF before
  TIFF sequence and OME-Zarr before plain Zarr).
- Default to reopening cheap file handles per read. Persistent handles require
  a demonstrated high open cost, `close()`, and TTL reaping.
- Preserve progressive discovery: `SERVING` may precede completion of the
  background catalog scan; freshness is reported separately.
- The file cache lock/WAL is released before draining Flight, while mmaps remain
  open until a clean drain. Preserve bounded shutdown and its ordering.

## Network and security constraints

Network bind settings are CLI concerns, not server config. Public Flight binds
require a token by default; public HTTP-sidecar binds without a token must fail
closed. TLS is opt-in at the server CLI, and the sidecar normally stays on
loopback behind control. Never log raw paths or credentials where the redaction
contract applies. Pool remote clients by endpoint plus credentials/trust, not
endpoint alone.

## Development and validation

From the repository root:

```sh
uv sync --all-packages --all-extras
uv run --no-sync pytest biopb-tensor-server/tests
```

Run the narrow test file first. The `integration` marker may require external
services or network access. Useful smoke commands include
`biopb-tensor-server validate`, `serve`, and `launch`; direct users of
`TensorFlightServer` must call `mark_ready()` themselves.

Root ruff settings apply. This component follows the product `release-v*`
version line. Keep Zarr on the documented 2.x API and do not relax dependency
pins without checking the compatibility rationale in `pyproject.toml`.
