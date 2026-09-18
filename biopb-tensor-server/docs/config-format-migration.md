# Config format: TOML → JSON

**Status:** done and swept up — JSON is the only format, value validation runs
on one shared checker (clamp at load, reject at the strict surfaces), the
installers write `biopb.json`, and the JSON Schema emitter is in. The
coexistence window (dual-format read + warn-level validation) ran from the
initial migration to the read-path removal; the one-way door out of it
(`migrate-config`, the `.toml` refusals, the installers' automatic conversion,
`find_config`'s legacy branch) stood for another two months and is now gone too.
Nothing in the tree knows the word TOML. Touches `biopb-tensor-server` (config)
and the `biopb` umbrella CLI.

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
  └── _read_config_file(path)        ← json.load, extension-blind
  └── parse_config(dict)             ← unchanged, format-agnostic
```

## Architecture (shipped)

**JSON-only reader.** `core.config.load_config` reads JSON, and does not look at
the extension to decide — an unconventionally-named config still loads.

**Default-path resolution — one shared impl.** `find_config(dir)` returns
`biopb.json`, present or not, so the caller seeding a fresh config and the one
reading an existing config name the same file. It has no legacy branch: a
leftover `biopb.toml` is an ordinary unrelated file. `find_config` lives once in
**`biopb._locations`**
(stdlib-only module in the core `biopb` package — no heavy adapter/discovery
imports, cheap to import per CLI invocation); `core.config` re-exports it
(`find_config` + name constants) and `biopb.cli` sets
`DEFAULT_CONFIG = find_config()`.

**Value validation — one scheme, shared with biopb-mcp and the control.**
Out-of-range / bad-enum knobs used to be accepted silently and blow up later on
the request path (`downscale_factor=0` → `ZeroDivisionError` in `GetFlightInfo`;
`pixel_budget_cubic_root<=0` → infinite loop in the precache worker;
`reduction_method="bogus"` → read-time error; `downscale_factor=1` → a silent
single-level pyramid). The rules are a single declarative `_CONSTRAINTS` table
(`_Range` / `_Enum` per field); the *checking* is `biopb._config_validate`, the
one walker every biopb config surface calls. See that module's docstring for the
policy; in short:

- **At the read step** (`parse_config` → `_clamp_invalid`), a violation is
  **warned and replaced with the dataclass default**. The bad value never reaches
  the request path — the actual requirement — and the server still starts. It is
  a control-plane child restarted on crash with capped backoff, so refusing to
  load would turn one bad number into a permanent restart loop reported as "the
  data plane keeps dying". biopb-mcp clamps for the mirror-image reason (a raise
  there is a dead MCP client and no viewer).
- **At the strict surfaces**, the same check *reports* instead:
  `validate_config_dict` → `PUT /api/config` (422, every problem, per field, so
  the form highlights them) and `biopb-tensor-server validate` (exit 1). A human
  is there to act on it, so it is not clamped away.

Validation lives at the read step, not in `__post_init__`: every value that can
be wrong arrives from a file or an HTTP body, both of which funnel through it,
and one check point is what lets biopb-mcp — whose runtime form is a merged dict,
never a constructed dataclass — share the same code. Direct dataclass
construction in server code is therefore unvalidated by design (a programming
error, not a runtime input).

One value that *looks* out of range is a sentinel and stays legal:
`full_rescan_interval <= 0` disables the periodic full scan. (`server.port = 0`
used to be another, for an OS-assigned ephemeral port; the bind left the config
entirely in biopb/biopb#604 and `--port 0` carries that meaning now.)

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
runs the same `_config_problems` the load path runs — it just builds the config
with `_build_config` (unclamped) so there is still something to report, where
`parse_config` would already have substituted the defaults. One rule set behind
both surfaces, so a config the form accepts is always one the server loads.
See `tensor-server-admin-endpoint.md`.

## Gotchas

- **One source of truth for the key set.** `config._warn_unknown_config_keys`
  (the unknown-key warning, #234 — catches e.g. `[cache] file_max_segment_bytes`
  instead of `file_max_segment_mb`) derives its known-section / known-key sets by walking
  `build_config_schema()`'s properties (`config_schema.known_config_keys`); the old
  hardcoded `_KNOWN_*` tables were deleted. The schema is the single source for the
  key set too. An unknown key stays **warn-and-ignore** — like a bad *value*,
  which is warned and defaulted, a config written by a newer tool must never be a
  startup failure — and the published schema stays
  `additionalProperties: true` so editor autocomplete doesn't error on it.
- **Case-insensitive enums carry no `enum`** — see the admin-endpoint pairing
  above; a schema-only check at save-time would accept a `log_level` /
  `reduction_method` the server then refuses at load.
- **A leftover TOML is inert** — `tests/config_format_test.py` pins that a
  `biopb.toml` beside (or instead of) the JSON never shifts what `find_config`
  answers and never warns; schema drift-guards in `tests/config_schema_test.py`
  assert every `_CONSTRAINTS` entry and every scalar dataclass field is reflected,
  that the runtime warning uses the schema-derived sets, and that the schema
  accepts the installer default while rejecting each known-bad value.

## Not done / future

- **Embed `$schema` in generated configs.** Have the writer emit a `"$schema"`
  pointer so editors validate out of the box. **Resolution path:** the planned
  `save_config` writer (`tensor-server-admin-endpoint.md`) drops a sibling
  `biopb.schema.json` and embeds a **relative** `"$schema": "./biopb.schema.json"`,
  giving offline editor auto-validation with **no hosting dependency**; switch the
  pointer to a hosted `$id` URL (e.g.
  `https://biopb.org/schemas/tensor-server-config.json`) once it is published.

## Equivalent configs

The shape mapping, for reading a pre-#34 file by eye. Nothing converts one any
more, so this is the whole of the upgrade path: a config this old is rewritten
by hand, or discarded and re-seeded by the installer.

```toml
[server]
host = "127.0.0.1"
port = 9000
[cache]
file_max_total_gb = 8
[[sources]]
type = "zarr"
url = "/data/a.zarr"
```

```json
{
  "cache": { "file_max_total_gb": 8 },
  "sources": [
    { "type": "zarr", "url": "/data/a.zarr" }
  ]
}
```

The two produced a byte-identical `ServerConfig` while both were readable.
`server.host` / `port` are themselves gone since biopb/biopb#604 — the bind is a
CLI flag — so that section of an old file has no JSON counterpart at all.
