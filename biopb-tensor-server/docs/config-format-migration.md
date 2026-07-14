# Config format: TOML → JSON (coexistence)

**Status:** shipped — read-side JSON/TOML coexistence, warn-level value
validation, installer writing `biopb.json`, and the JSON Schema emitter are all
in. Still design: the `.toml` read path is not yet dropped and `$schema` is not
yet embedded in generated configs (see Not done / future). Touches
`biopb-tensor-server` (config), the `biopb` umbrella CLI, and `biopb-mcp`.

## Why

TOML was chosen for hand-editing ergonomics (comments, `[[sources]]`). But the
server config is **machine-generated** — the installer writes it, and a future
generator will too — so nobody hand-edits it, and TOML's advantages evaporate
while its one wart dominates:

- **No stdlib TOML *writer*.** Installers hand-template TOML text with manual
  escaping plus a BOM workaround. JSON generation is stdlib on both ends
  (`json.dumps` / `ConvertTo-Json`), and that fragile block disappears.
- **Ecosystem alignment.** `biopb-mcp` already writes
  `~/.config/biopb/mcp-config.json`; server-on-JSON unifies the format.
- **Comments are moot** for a generated file, so JSON's lack of them costs nothing.
- **Pairs with JSON Schema**, giving validation + generation one shared contract.

**What makes this contained:** format matters *only at the read step*.
`parse_config(data: dict)` already operates on a plain dict — TOML vs JSON differ
only in how that dict is produced. The data model (`ServerConfig` and friends) is
untouched, and one read-side change covers every code path that loads a config.

```
load_config(path)
  └── _read_config_file(path)        ← the ONLY format-aware step
        ├── .json  → json.load
        ├── .toml  → tomllib.load  (+ deprecation warning)
        └── other  → sniff: JSON first, then TOML
  └── parse_config(dict)             ← unchanged, format-agnostic
```

## Architecture (shipped)

**Dual-format reader.** `core.config.load_config` dispatches on file extension
(content sniff for extension-less paths). Reading TOML logs a one-line
deprecation warning naming `biopb.json` as canonical; an invalid file raises
`ValueError` naming the path.

**JSON-preferred default-path resolution — one shared impl.** `find_config(dir)`
returns the first of `biopb.json` → `biopb.toml` that exists, else the canonical
`biopb.json`. When both exist the legacy TOML is silently shadowed, so it warns
naming the ignored file. It lives once in **`biopb._config_location`** (stdlib-only
module in the core `biopb` package — no heavy adapter/discovery imports, cheap to
import per CLI invocation) and all three consumers call it: `core.config`
re-exports it (`find_config` + name constants), `biopb.cli` sets
`DEFAULT_CONFIG = find_config()`, and `biopb_mcp._connection` sets
`DEFAULT_SERVER_CONFIG = find_config()`. All three already depend on core `biopb`,
so there is no per-consumer twin to drift. CLI help text reads "config file (JSON
or TOML)". Back-compat is total: an existing `biopb.toml` still loads and is still
picked up (it only loses to a `biopb.json` beside it).

**Value validation.** Out-of-range / bad-enum knobs used to be accepted silently
and blow up later on the request path (`downscale_factor=0` → `ZeroDivisionError`
in `GetFlightInfo`; `pixel_budget_cubic_root<=0` → infinite loop in the precache
worker; `reduction_method="bogus"` → read-time error; `downscale_factor=1` → a
silent single-level pyramid). A single declarative `_CONSTRAINTS` table
(`_Range` / `_Enum` per field) is enforced in each config dataclass's
`__post_init__`, so **every** construction path — both file formats and direct
dataclass construction — is covered; messages name section, key, value, and
accepted range/enum. Severity is **warn** during the deprecation window
(`_STRICT_VALIDATION = False`): a config that loaded before must not become a hard
startup failure on upgrade. The disable sentinel `full_rescan_interval <= 0` is
intentionally not constrained.

**Installer writes JSON.** All four front-ends — `install/install.sh` (POSIX),
`install/biopb-engine.ps1` (Windows engine), `install/install.ps1`, and
`install/gui/biopb-setup.iss` — write `~/.config/biopb/biopb.json` via stdlib JSON
generation and detect an existing config in **either** format for the keep prompt
(`biopb.json` wins, matching `find_config`). Because JSON has a stdlib *reader*, a
re-run pointing at a new data folder no longer rewrites the whole file: the
existing config is loaded, its server/cache/… settings preserved, and only the
`sources` list replaced. A legacy `biopb.toml` is migrated to JSON on that path
(POSIX reads it via `tomllib`/`tomli`; Windows has no TOML parser so it starts
from installer defaults) and the old file is backed up so the both-files shadow
warning never fires. The default template drops the removed `[metadata_db]
enabled` flag — the metadata DB is now mandatory (biopb/biopb#225); a lingering
flag is ignored with a warning.

**JSON Schema emitter.** `core.config_schema.build_config_schema()` projects the
config dataclasses + `_CONSTRAINTS` as a Draft 2020-12 JSON Schema for the
**on-disk** config, so the key set and value bounds match exactly what the server
reads and enforces — one definition, no drift. Key set + types come from the
dataclasses (introspected, routed to their on-disk section); bounds/enums from the
`_Range`/`_Enum` objects (each has `to_json_schema()`). Two declarative pieces:
`_ONDISK_OVERRIDES` (fields whose wire form differs — `cache.*_mb`/`*_gb` convert
to byte fields) and `_DEPRECATED_ALIASES` (legacy keys the parser accepts but that
aren't dataclass fields — `watcher_type`/`poll_interval`, source `path`, the
`[precache]` pyramid knobs, and the removed `metadata_db.enabled`, flagged
`deprecated: true`). Sections keep `additionalProperties: true`, so the schema
catches the dangerous values while documenting every known key without being a
closed dictionary. Emit with **`biopb-tensor-server config-schema [-o file.json]`**.

**Schema paired with the server's real validator at the admin endpoint.** The
schema deliberately can't express the case-insensitive enums (`log_level`,
`reduction_method` emit no hard `enum` — the server folds case, so a canonical-set
enum would reject values it accepts; the accepted set rides in the property
`description` instead). So `PUT /api/config` validates a submitted config with
**both** the JSON Schema **and** `config.validate_config_dict` — the same
`_CONSTRAINTS` gate the server runs at load, exposed as structured `{path,
message}` problems on the schema's on-disk paths (via
`config_schema.ondisk_location`) so the two dedupe by path. `validate_config_dict`
shares its core (`_config_problems`) with load-time `_validate_config` and is
independent of the warn/raise `_STRICT_VALIDATION` policy — one rule set behind
both surfaces. A config the form accepts is therefore always one the server loads.
See `tensor-server-admin-endpoint.md`.

## Gotchas

- **One source of truth for the key set.** `config._warn_unknown_config_keys`
  (the unknown-key warning, #234 — catches e.g. `[cache] memory_max_entries`
  instead of `max_entries`) derives its known-section / known-key sets by walking
  `build_config_schema()`'s properties (`config_schema.known_config_keys`); the old
  hardcoded `_KNOWN_*` tables were deleted. The schema is the single source for the
  key set too. Runtime behavior is unchanged (warn-only), and the published schema
  stays `additionalProperties: true` so editor autocomplete doesn't error on
  unknown keys during the window.
- **Case-insensitive enums carry no `enum`** — see the admin-endpoint pairing
  above; a schema-only check at save-time would accept a `log_level` /
  `reduction_method` the server then refuses at load.
- **Both formats produce a byte-identical `ServerConfig`** — covered by
  `tests/config_format_test.py`; schema drift-guards in `tests/config_schema_test.py`
  assert every `_CONSTRAINTS` entry and every scalar dataclass field is reflected,
  that the runtime warning uses the schema-derived sets, and that the schema
  accepts the installer default while rejecting each known-bad value.

## Not done / future

- **Drop the `.toml` read path and flip validation warn → raise.** The end state:
  remove `tomllib` dispatch from `_read_config_file`, make `find_config` return
  only `biopb.json`, and flip `_STRICT_VALIDATION → True` so an out-of-range knob
  is a hard startup failure. Gated on new installs being JSON-native and old TOML
  installs having migrated (they migrate the next time they change data folders).
- **Embed `$schema` in generated configs.** Have the writer emit a `"$schema"`
  pointer so editors validate out of the box. **Resolution path:** the planned
  `save_config` writer (`tensor-server-admin-endpoint.md`) drops a sibling
  `biopb.schema.json` and embeds a **relative** `"$schema": "./biopb.schema.json"`,
  giving offline editor auto-validation with **no hosting dependency**; switch the
  pointer to a hosted `$id` URL (e.g.
  `https://biopb.org/schemas/tensor-server-config.json`) once it is published.

## Equivalent configs

```toml
[server]
host = "127.0.0.1"
port = 9000
[cache]
backend = "memory"
[[sources]]
type = "zarr"
url = "/data/a.zarr"
dim_labels = ["z", "y", "x"]
```

```json
{
  "server": { "host": "127.0.0.1", "port": 9000 },
  "cache": { "backend": "memory" },
  "sources": [
    { "type": "zarr", "url": "/data/a.zarr", "dim_labels": ["z", "y", "x"] }
  ]
}
```

Both produce a byte-identical `ServerConfig`.
