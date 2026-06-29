# Config Format Migration — TOML → JSON (coexistence phase)

**Status:** Read-side coexistence implemented; installer/generator + validation deferred
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

2. **JSON-preferred default-path resolution.** `find_config(dir)` returns the
   first of `biopb.json` → `biopb.toml` that exists, else the canonical
   `biopb.json` path. The three entry points that hardcoded `biopb.toml` now
   prefer JSON:
   - `biopb_tensor_server.config.find_config` (source of truth)
   - `biopb.cli._default_config` (umbrella `biopb server …`)
   - `biopb_mcp._connection._default_server_config` (auto-start spawner) — a
     4-line twin, because `biopb-mcp` has **no runtime dependency** on
     `biopb_tensor_server` and cannot import the canonical helper.

3. **CLI help text** updated from "TOML config file" to "config file (JSON or
   TOML)" across `serve`/`launch`/`validate`/`list` and `biopb server …`.

Back-compat is total: an existing `~/.config/biopb/biopb.toml` still loads and is
still picked up by every default-path resolver (it only loses to a `biopb.json`
sitting beside it).

## Deferred (not part of "coexistence for now")

- **Installer writes JSON.** Replace the TOML text templating in `install.sh` /
  `biopb-engine.ps1` with `json.dumps` / `ConvertTo-Json` and write `biopb.json`.
  Once shipped, new installs are JSON-native; old TOML installs keep working.
- **Value validation (#34 proper).** The `CONSTRAINTS` table + `__post_init__`
  enforcement (severity = warn during the window). Format-independent, lands on
  the same data model.
- **JSON Schema emitter.** Generated from `CONSTRAINTS`, feeding the config
  generator + editor autocomplete + optional pre-flight validation.
- **End state.** Drop the `.toml` read path and flip validation warn → raise.

## Sequencing (from #34)

1. ~~Read-side coexistence (this doc).~~ ✅
2. Installer emits JSON; value-validation `CONSTRAINTS` (warn).
3. Schema emitter; make JSON the only documented format.
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
