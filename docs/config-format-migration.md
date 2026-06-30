# Config Format Migration — TOML → JSON (coexistence phase)

**Status:** Phases 1–3 done — read-side JSON/TOML coexistence (`biopb._config_location` + TOML deprecation warning), warn-level value validation (`_CONSTRAINTS`, `config.py`), the installer now writing `biopb.json` (all four front-ends: `install/install.sh`, `install/biopb-engine.ps1`, `install/install.ps1`, `install/gui/biopb-setup.iss`), and the JSON Schema emitter (`config_schema.py` + `biopb-tensor-server config-schema`). Remaining: `.toml` read path not yet dropped; `$schema` not yet embedded in generated configs.
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

## Installer writes JSON (done)

The TOML text-templating in the installers is replaced by stdlib JSON generation
(`json.dump` on POSIX, `ConvertTo-Json` on Windows), so the fragile hand-rolled
escaping (`biopb-engine.ps1`'s `-replace '\\','\\' -replace '"','\"'` + the BOM
workaround) is gone. The four front-ends — `install.sh` (POSIX), `biopb-engine.ps1`
(Windows engine), `install.ps1` and `gui/biopb-setup.iss` (Windows front-ends) —
now write `~/.config/biopb/biopb.json` and detect an existing config in **either**
format for the keep prompt (`biopb.json` wins, matching `find_config`).

Because JSON has a stdlib *reader*, a re-run that points at a new data folder no
longer rewrites the whole file: the existing config is loaded, its
server/cache/… settings are **preserved**, and only the `sources` list is
replaced with the chosen folder (a `[[sources]]`-clobbering rewrite before). A
legacy `biopb.toml` is migrated to JSON on that path (POSIX reads it via
`tomllib`/`tomli` to carry settings; Windows has no TOML parser so it starts from
installer defaults) and the old file is backed up so the server's both-files
shadow warning never fires. The default template also drops the now-deprecated
`[metadata_db] enabled = true` (DB on by default, biopb/biopb#225). New installs
are JSON-native; old TOML installs keep working until they next change folders.

## JSON Schema emitter (done)

`biopb_tensor_server.config_schema.build_config_schema()` projects the config
dataclasses + the `_CONSTRAINTS` table as a Draft 2020-12 JSON Schema for the
**on-disk** config, so the key set and value bounds in the schema match exactly
what the server reads and enforces at startup — one definition, no drift. The
key set + types come from the dataclasses (introspected, routed to their on-disk
section), bounds/enums from the `_Range`/`_Enum` objects (each grew a
`to_json_schema()`). Two small declarative pieces remain: `_ONDISK_OVERRIDES`
(the few fields whose wire form differs — `[compute]` is read into `ServerConfig`;
`cache.*_mb`/`*_gb` convert to byte fields) and `_DEPRECATED_ALIASES` (legacy
keys the parser still accepts but that aren't dataclass fields: `watcher_type`/
`poll_interval`, source `path`, the pyramid knobs under `[precache]`, plus the
deprecated `metadata_db.enabled`). Sections keep `additionalProperties: true`, so
the schema enforces the dangerous values #34 exists to catch and documents every
known key (deprecated ones flagged `deprecated: true`) without being a closed
dictionary.

Case-insensitive enums (`log_level`, `reduction_method`) emit **no** hard
`enum` — the server folds case, so a canonical-set enum would reject values it
accepts — and instead carry the accepted set in the property `description`.

**Subsumes the unknown-key warning (#234).** That feature warned on unrecognized
config keys (the silent drop-to-default trap, e.g. `[cache] memory_max_entries`
instead of `max_entries`) from three hardcoded `_KNOWN_*` tables — a second
source of truth parallel to `_CONSTRAINTS`, the exact drift this work removes.
`config._warn_unknown_config_keys` now derives its known-section / known-key sets
by walking `build_config_schema()`'s properties (`config_schema.known_config_keys`),
and those three tables are deleted. Runtime behavior is unchanged (warn-only,
same messages, legacy aliases stay quiet); the schema is simply the single source
for the key set too. The published schema stays `additionalProperties: true` —
editor autocomplete shouldn't error on unknown keys during the migration window —
while the server emits the warnings.

Emit it with `biopb-tensor-server config-schema [-o file.json]`. Reference the
saved file from a config via `"$schema"` for editor autocomplete, or feed it to
any JSON Schema validator for pre-flight checks. Drift-guard tests
(`tests/config_schema_test.py`) assert every `_CONSTRAINTS` entry **and** every
scalar dataclass field is reflected in the schema, that the runtime warning uses
the schema-derived sets, and that the schema accepts the installer default while
rejecting each known-bad value (`downscale_factor` 0/1, `pixel_budget_cubic_root`
0, out-of-range `port`, bad `backend`, `backlog_high_water > 1`, …).

## Deferred

- **Embed `$schema` in generated configs.** Have the installer write a
  `"$schema"` pointer (needs the schema hosted at a stable URL, e.g. the `$id`
  `https://biopb.org/schemas/tensor-server-config.json`) so editors validate
  out of the box.
- **End state.** Drop the `.toml` read path and flip validation warn → raise.

## Sequencing (from #34)

1. ~~Read-side coexistence.~~ ✅
2. ~~Value-validation `_CONSTRAINTS` (warn).~~ ✅
3. ~~Installer emits JSON~~ ✅; ~~schema emitter~~ ✅; make JSON the only documented format.
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
