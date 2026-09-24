# Splitting the napari plugin out of biopb-mcp

Status: step 1 (below) is implemented; steps 2-4 are not.

**Components:** `biopb-mcp` (the code leaves it), the core SDK (`biopb`, gains a
`control` subpackage), a new repository `biopb/biopb-napari-widget` for the
plugin, `install/`.

## Goal

The napari contributions in `biopb-mcp` — the Tensor Browser, the two
image-processing widgets with their gRPC client, the two OME-Zarr writers —
depend on the SDK and on napari, and on nothing in `biopb_mcp.mcp`. They
belong in a repository of their own, on PyPI and the napari hub, where an
ordinary napari user can install them. Today they ship only inside the
biopb-mcp wheel set on GitHub releases (biopb-mcp is not on PyPI), so a napari
user without the installer cannot get the Tensor Browser at all.

The MCP kernel keeps using the plugin: it docks the Tensor Browser at bootstrap
and adds tensors through the plugin's layer pipeline. **biopb-mcp depends on
the plugin package, never the reverse.**

## What moves

| today in `biopb_mcp` | lines | goes to |
|---|---|---|
| `tensor_browser/` | 2,500 | plugin |
| `image_processing/` (widgets, chunking, gRPC client) | 1,800 | plugin |
| `_writers.py` (OME-Zarr writers) | 200 | plugin |
| `_tensor_utils.py` (`add_tensor_layer`) | 650 | plugin |
| `_viewer_compute.py` (`wrap_levels`, used by `add_tensor_layer`) | 150 | plugin |
| `_utils.py` (`_filter_boxes`), `_typing.py` | 60 | plugin |
| `_catalog.py` (the widget's view of a catalog row) | 90 | plugin |
| `_control_client.py`, the HTTP client part | 110 | SDK, `biopb.control` |
| `_connection.py` | 800 | dissolves (below) |
| `napari.yaml` | | plugin, renamed |

`add_tensor_layer` is the one module both sides need and cannot live in the SDK:
it builds napari layers, and the SDK stays napari-free. It is also the reason
the browser and MCP `add_tensor` cannot disagree about what a label set is, so
it must not be duplicated. It goes with the plugin, and the kernel's
`add_tensor` (`mcp/_helpers.py`) imports it from there. That fixes the
dependency direction.

`start_control_detached` in `_control_client.py` launches the control process
for the stdio shim. It stays in biopb-mcp: `biopb.control` is a client of a
running control, not a way to start one.

`plugins/` (kernel-namespace plugins: `rolling_ball`, `chunked_label`, …) is
not part of this: it imports nothing from biopb-mcp either, but it belongs to
the algorithm plane, not to napari.

## The connection dissolves into the control

`_connection.TensorConnection` re-derives, client-side, facts the control or
the plane already holds. The SDK carries a thin, stdlib-only client of the
control's HTTP API, and the plane answers for itself over Flight.

| in `_connection` today | becomes |
|---|---|
| `auto_connect`, `connect_when_booted`, `server_start_timeout` | `biopb.control.data_plane()` over `/health` and `/api/data_plane/ensure`: the plane's URL and token, or `None` when no control answers |
| `health`, `scan_in_progress`, `scan_source_count`, the health poll and its two backoff keys | `TensorFlightClient.health_check()`, the plane's own answer (`full_scan_in_progress`); it works with or without a control |
| `refresh`, `query_sources`, `resolve_source`, `add_source`, `remove_source`, `warm_source` | the tensor server's, over Flight; `biopb.tensor.TensorFlightClient` has them |
| `start_source_watch` | a re-list timer over the same Flight call |
| `is_localhost` | `biopb._data_plane.is_local_url`, which exists |
| `is_connected`, `mark_disconnected` | the caller's own state (`client is None`) |

What remains is one small object in the SDK, `biopb.tensor.Connection`: the
endpoint from `biopb.control.ensure_data_plane()`, `connect()`, `client`, `url`
and `last_message`, which carries the connect error messages. It caches
nothing. `health` answers anyone, so `connect()` makes one catalog call as
well, which is the one that checks the token. It lives in the SDK because everything that
connects uses it, and most of them have no napari: the kernel, the scratch
kernel, the saved-workflow bootstrap cell (`mcp/_notebook.py`) and
`workflow_env()`, which returns it to a saved workflow, so `client` stays.

The kernel builds one and hands it to the Tensor Browser, so the agent and the
widget share one handle: when the widget reconnects, the kernel's next job
picks up the new `client`. A standalone widget builds its own. What only the
widget needs sits in the plugin, on top of it: the source list, the re-list
timer and its change callbacks, the connect status it displays, the manual URL
field. The kernel's catalog watcher (bootstrap step 7) goes with them; the
agent was its other reader, through the cached `sources`, which leave the
kernel with `add_tensor`'s change below.

The kernel's `add_tensor` today reads the connection's cached catalog to pick
which tensor a bare source id means, and wraps a fetched descriptor in a
`CatalogSource` when the cache misses. That is the only kernel-side use of
`_catalog`, and it exists because the connection owns a cache. With the cache
gone, `add_tensor` asks the server (`get_descriptor`, or one `query_sources`
call) and the kernel never sees a catalog row type.

**No-control fallback.** `data_plane()` returning `None` is what the widget
shows a URL and token field for. That dial goes straight to the Flight client
with no credential file involved, which is the #628 rule as it stands: the
control's credential travels only with an endpoint the control named.
`$BIOPB_TENSOR_URL` keeps working the same way.

**The control adds nothing.**

## `biopb.control`: a public subpackage of the SDK

The SDK's `_`-prefixed modules (`_endpoints`, `_credentials`, `_data_plane`,
`_sessions`, `_algorithms`, …) are plumbing shared *inside* the monorepo. A
plugin in another repository importing `biopb._data_plane` would depend on a
private API across a repo boundary, so the client of the control's HTTP API
becomes a public subpackage:

```
biopb.control
    data_plane()          -> {"url", "token"} | None      /health, then the credential file
    ensure_data_plane()   -> the same, starting the plane  POST /api/data_plane/ensure
    base_url()            -> where the control listens     biopb._endpoints
    is_local_url(url)     -> whether url is this machine   the drag-drop gate, the TLS rule
```

Rules for it:

- **Stdlib only**, importable with no extra, like the `_` modules it wraps.
  `biopb.tensor` needs pyarrow; this must not.
- **A client, not the control.** `biopb-control` (starlette, uvicorn, httpx)
  stays uninstallable for clients, as `biopb-tensor-server` is. Nothing from
  it moves here, and nothing here starts a process.
- **Small and versioned by the SDK.** The plugin pins a floor on `biopb`, and
  this surface is what that floor promises. Server-side registries the control
  reads (`_sessions`, `_web_auth`, `_agents`) stay private at the top level.
- The client half of `_control_client.py`, `_data_plane.py` and `_endpoints.py`
  become its implementation under `biopb/control/`; `biopb._endpoints` remains
  as an alias while the monorepo's other packages migrate.

## The discovery contract

`biopb.control` is one implementation of a language-neutral contract, so that a
Java client (next to the Java `TensorFlightClient`) can find the same plane with
the same token: [discovery-contract.md](discovery-contract.md).

## Config

The widgets read a dozen keys from biopb-mcp's config — the image-processing
server URL, detection defaults, grid sizes, gRPC sizes, memory thresholds,
timeouts — and the object-detection widget writes its fields back
(`_save_config`). These move with the widgets as the plugin's own settings,
stored where the plugin keeps them, not in `mcp-config.json`: a second reader
and writer of that file across a repo boundary is what this split removes.
`widget`, `detection` and `grid` leave biopb-mcp's schema. `timeout`, `grpc`
and `memory` stay, because the kernel's `ops` reads them too; the plugin has
its own copies of the keys its widgets use.

The two health-poll intervals and the server-start timeout dissolve with
`_connection`. The `pyramid` section is read by nothing (the server advertises
every tensor's pyramid) and is deleted, not moved.

## Naming

The repository is `biopb/biopb-napari-widget`, published to PyPI as
`biopb-napari-widget` with the import package `biopb_napari_widget`. The
manifest takes the same name, and its command ids with it
(`biopb-napari-widget.tensor_browser`, …). Nothing outside `napari.yaml`
refers to the ids.

`napari-biopb` on PyPI is an older, separate project and is left as it is.

## The kernel's use of the plugin

The kernel imports three things, and they are the plugin's API to biopb-mcp:

- `TensorBrowserWidget(viewer, connection=…, compute_scheduler=…)`, docked at
  bootstrap. `connection` is the SDK's `Connection`.
- `add_tensor_layer`, for MCP `add_tensor`.
- `wrap_levels`, for the viewer's single-process scheduler pin.

## Costs

- **Tests and CI.** `test_tensor_browser_widget`, `test_tensor_browser_drop`,
  `test_tensor_browser_metadata`, `test_widget` and `test_writers` move, and
  the new repository needs the headless-GL Qt job that `mcp-ci.yaml` already
  carries.
- **Protocol drift.** The SDK refuses v1 Flight servers after protocol v2; in
  the monorepo a plugin cannot lag such a change, in its own repo it can. The
  plugin pins a floor on `biopb` and CI runs it against the SDK's head.
- **napari version.** biopb-mcp pins napari exactly, for the viewer proxy's
  tripwire against napari internals. That pin stays in biopb-mcp; the plugin
  floats, which is a benefit for its users.
- **Installer.** `install/install.sh` installs biopb-mcp from a wheel set; with
  the plugin on PyPI the dependency resolves, but the frozen build's wheel list
  gains an entry.

## Shape of the work

Each step leaves the monorepo working.

1. The discovery contract as its own page under `docs/`; `biopb.control` in
   the SDK, implementing it; the widget and kernel switch to it; the
   no-control URL field; health from the Flight client; `Connection` in the
   SDK; the source list and watcher move into the widget; `_connection` is
   deleted.
2. An SDK release to PyPI carrying `biopb.control` and `Connection`, since the plugin's floor
   can only name a published version.
3. The plugin repository: the modules above, `napari.yaml`, the widgets'
   settings, their tests and CI, published to PyPI as `biopb-napari-widget`.
4. biopb-mcp depends on `biopb-napari-widget`, imports the widget,
   `add_tensor_layer` and `wrap_levels` from it, and deletes its copies and the
   moved config sections; installer wheel list updated.
