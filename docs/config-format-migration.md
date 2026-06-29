# Config Format Migration — TOML → JSON (coexistence phase)

**Status:** Phases 1–2 done — read-side JSON/TOML coexistence (`biopb._config_location` + TOML deprecation warning) and warn-level value validation (`_CONSTRAINTS`, `config.py`). Phase 3 deferred — installer still writes TOML (`install/biopb-engine.ps1`); JSON Schema emitter not built.
**Component:** `biopb-tensor-server` (config), `biopb` umbrella CLI, `biopb-mcp`
**Tracking:** biopb/biopb#34 (the agreed plan lives in that issue's first comment)

---

## Why migrate

TOML was chosen for hand-editing ergonomics (comments, `[[sources]]`). But the
server config is **machine-generated** — the installer writes it, and a future
generator will too — so nobody hand-edits it. Once that is true, TOML's
advantages evaporate and its one wart dominates:

- **No stdlib TOML *writer*.** The installer hand-templates TOML text with manual
  escaping (`biopb-engine.ps1` `-replace '\\','\\' -replace '"','\"'`) plus a BOM
  workaround. JSON generation is stdlib on both ends (`json.dumps` /
  `ConvertTo-Json`), and that fragile block disappears.
- **Ecosystem alignment.** `biopb-mcp` already writes
  `~/.config/biopb-mcp/config.json`. Server-on-JSON unifies the format.
- **Comments are moot** for a generated file, so JSON's lack of them costs nothing.
- **Pairs with JSON Schema**, giving validation + generation one shared contract.

## Key property that makes this contained

Format matters **only at the read step**. `parse_config(data: dict)` already
operates on a plain dict — TOML vs JSON differ only in how that dict is produced.
So the data model (`ServerConfig` and friends) is untouched, and one read-side
change covers every code path that loads a config.

```
load_config(path)
  └── _read_config_file(path)        ← the ONLY format-aware step
        ├── .json  → json.load
        ├── .toml  → tomllib.load  (+ deprecation warning)
        └── other  → sniff: JSON first, then TOML
  └── parse_config(dict)             ← unchanged, format-agnostic
```

## What is implemented now (coexistence)

1. **Dual-format reader.** `biopb_tensor_server.config.load_config` dispatches on
   file extension, with a content sniff for extension-less paths. Reading TOML
   logs a one-line deprecation warning naming `biopb.json` as canonical. Invalid
   files raise a `ValueError` naming the path.

2. **JSON-preferred default-path resolution — one shared implementation.**
   `find_config(dir)` returns the first of `biopb.json` → `biopb.toml` that
   exists, else the canonical `biopb.json` path. When **both** exist the legacy
   TOML is silently shadowed, so it logs a warning naming the ignored file.
   It lives once in **`biopb._config_location`** (a stdlib-only module in the
   core `biopb` package — no heavy adapter/discovery imports, so it is cheap to
   import on every CLI invocation) and the three consumers all call it:
   - `biopb_tensor_server.config` re-exports it (`find_config` + name constants)
   - `biopb.cli` sets `DEFAULT_CONFIG = find_config()`
   - `biopb_mcp._connection` sets `DEFAULT_SERVER_CONFIG = find_config()`

   This works because all three already depend on the core `biopb` package
   (`biopb-tensor-server` lists `biopb`; `biopb-mcp` lists `biopb[tensor]`; the
   umbrella CLI *is* core), so there is no longer a per-consumer twin to drift.

3. **CLI help text** updated from "TOML config file" to "config file (JSON or
   TOML)" across `serve`/`launch`/`validate`/`list` and `biopb server …`.

Back-compat is total: an existing `~/.config/biopb/biopb.toml` still loads and is
still picked up by every default-path resolver (it only loses to a `biopb.json`
sitting beside it).

4. **Value validation (#34 proper).** Out-of-range / bad-enum knobs used to be
   accepted silently and blow up later on the request path (`downscale_factor=0`
   → `ZeroDivisionError` in `GetFlightInfo`; `pixel_budget_cubic_root<=0` →
   infinite loop in the precache worker; `reduction_method="bogus"` → a read-time
   error; `downscale_factor=1` → a silently single-level pyramid). A single
   declarative `_CONSTRAINTS` table (`_Range` / `_Enum` per field) is enforced in
   each config dataclass's `__post_init__`, so **every** construction path — both
   file formats and direct dataclass construction — is covered. Messages name the
   section, key, value, and accepted range/enum.

   Severity is **warn** during the deprecation window (`_STRICT_VALIDATION =
   False`): a config that loaded before must not become a hard startup failure on
   upgrade. The disable sentinel `full_rescan_interval <= 0` is intentionally
   *not* constrained. Flip `_STRICT_VALIDATION → True` (warn → raise) when the
   legacy read path is removed.

## Deferred

- **Installer writes JSON.** Replace the TOML text templating in `install.sh` /
  `biopb-engine.ps1` with `json.dumps` / `ConvertTo-Json` and write `biopb.json`.
  Once shipped, new installs are JSON-native; old TOML installs keep working.
- **JSON Schema emitter.** Generated from the same `_CONSTRAINTS` table, feeding
  the config generator + editor autocomplete + optional pre-flight validation.
- **End state.** Drop the `.toml` read path and flip validation warn → raise.

## Sequencing (from #34)

1. ~~Read-side coexistence.~~ ✅
2. ~~Value-validation `_CONSTRAINTS` (warn).~~ ✅
3. Installer emits JSON; schema emitter; make JSON the only documented format.
4. Drop `.toml` read path; flip warn → hard-fail.

## Equivalent configs

TOML:

```toml
[server]
host = "127.0.0.1"
port = 9000

[cache]
backend = "memory"

[[sources]]
type = "zarr"
url = "/data/a.zarr"
source_id = "a"
dim_labels = ["z", "y", "x"]
```

JSON (canonical):

```json
{
  "server": { "host": "127.0.0.1", "port": 9000 },
  "cache": { "backend": "memory" },
  "sources": [
    {
      "type": "zarr",
      "url": "/data/a.zarr",
      "source_id": "a",
      "dim_labels": ["z", "y", "x"]
    }
  ]
}
```

Both produce a byte-identical `ServerConfig`. Covered by
`biopb-tensor-server/tests/config_format_test.py`.
