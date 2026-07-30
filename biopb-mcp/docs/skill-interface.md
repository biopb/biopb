# Skill Interface — Curated Agent Workflows Sourced from biopb.org

**Status:** Partly implemented. P0 contract delivered in `biopb-site` (schema,
builder/validator, generated `catalog.json` — see Appendices A/B), which now also
enforces the required body sections and carries the catalog's scope and authoring
rules in `skills/ROADMAP.md`; **P2 (MCP retrieval) shipped** — the `find_skills`
tool, the `skill://{skill_id}` resource, the `services.skills_*` config, and the
local skills directory (§3f) are live in `biopb-mcp` (`mcp/_skills.py`,
`mcp/_server.py`), as is `requires:` resolution in the kernel (§3g,
`mcp/_requires.py`). P1 and P3–P4 not started.
**Component:** `biopb-mcp` (discovery + retrieval), `biopb-site` (authoring + publishing)
**Related:** the MCP `guide://*` resources and `find_skills`-style discovery, the
`mcp.services` config block, the fail-open remote fetch in
`biopb_mcp/mcp/_update.py`, the dynamic op discovery in
`biopb_mcp/mcp/_process_ops.py`, the server's `_BASE_INSTRUCTIONS`
("ask the user whether a new skill should be generated…").

---

## Goal

Give the agent a library of **curated, reusable workflows ("skills")** — e.g.
"segment cells with Cellpose", "build a multiscale pyramid and load it", "measure
labels and export a table". Each skill is a markdown file with YAML frontmatter,
authored and reviewed through a **git workflow** in `biopb-site`, published on
`https://biopb.org/`, and consumed at runtime by the `biopb-mcp` server through:

1. a **discovery tool** (`find_skills`) that queries a catalog, and
2. a **dynamic resource list** (`skill://<id>`) that returns the full workflow body.

This realizes the loop the server already gestures at in its instructions —
*"after a task, ask whether a new skill should be generated and added to the
agent's toolbox"* — where the toolbox is the curated catalog and "adding" is a PR.

Two repos, **one contract**: the published `catalog.json`. The site owns
*authoring + publishing*; the MCP server owns *discovery + retrieval* and degrades
gracefully when offline.

```
   biopb-site repo (curation = git)          biopb-mcp server (runtime)
   ┌───────────────────────────┐             ┌────────────────────────────┐
   │ skills/<id>.md (frontmtr)  │   CI build  │  find_skills(query) TOOL   │
   │ scripts/build_catalog.py   │──generates─▶│    → queries catalog       │
   │ skills/catalog.json (gen)  │   + rsync   │                            │
   │ docs/skills.md (browser)   │             │  skill://<id> RESOURCES    │
   └───────────────────────────┘             │    → lazy-fetch .md body    │
            │ served at                       │  (dynamic list from catalog)│
            ▼                                  └──────────┬─────────────────┘
   https://biopb.org/skills/catalog.json  ◀── httpx GET ──┘  fail-open:
   https://biopb.org/skills/<id>.md       ◀── httpx GET ──┘  cache → bundled
```

---

## Design principle — variation is a publisher problem, not a consumer problem

Skill files are authored by humans and agents over time; their format *will* drift
(missing fields, `tags` as a string vs. a list, freeform bodies, evolving
conventions). The load-bearing decision of this design is that **all variation is
normalized at one choke point — the site's build script — and never reaches the
MCP server.** Postel's law applied to skills: the build is *liberal in what it
accepts* (tolerant reader, coercion, inference, migrations) and *conservative in
what it publishes* (a strict, canonical, versioned `catalog.json` + normalized
bodies). See [§5](#5-handling-format-variation) for the full strategy.

---

## 1. The contract: `catalog.json`

Published at **`https://biopb.org/skills/catalog.json`**. Metadata only — bodies
are fetched lazily and separately, keeping the catalog small and discovery cheap.

```jsonc
{
  "catalog_version": 1,               // schema of THIS file; server guards, fails open on unknown
  "generated": "2026-06-30T12:00:00Z",
  "skills": [
    {
      "id": "cell-segmentation-cellpose",  // == filename stem; kebab; unique; stable
      "title": "Segment cells with Cellpose",
      "description": "Run Cellpose over the active image layer and load the labels.", // 1 line; drives discovery
      "tags": ["segmentation", "cellpose", "ops"],
      "version": "1.2.0",                  // author-owned semver of the skill's content
      "spec_version": 1,                   // body/frontmatter dialect; enables migrations
      "requires": ["viewer", "ops:segmentation"],  // capability tokens, resolved in the kernel (§3g)
      "updated": "2026-06-20",             // derived from git log, NOT the author
      "url": "https://biopb.org/skills/cell-segmentation-cellpose.md",
      "sha256": "e3b0c4…"                  // body integrity + client cache key
    }
  ]
}
```

Two independent version knobs (see [§5.3](#53-versioning-the-contract)):
`catalog_version` (the file schema) and per-skill `spec_version` (the authoring
dialect). Bodies are **not inlined** — the server reads `skill://<id>` on demand
and fetches `url`, verifying `sha256`.

### The skill file

`skills/<id>.md` is a Claude-style skill: frontmatter + a markdown body written to
drop into the agent's context.

```markdown
---
id: cell-segmentation-cellpose
title: Segment cells with Cellpose
description: Run Cellpose over the active image layer and load the labels.
tags: [segmentation, cellpose, ops]
version: 1.2.0
requires: [viewer, "ops:segmentation"]
---

# Segment cells with Cellpose

**When to use.** The user has a 2D/3D fluorescence image loaded and wants
instance labels for cells/nuclei.

## Steps
1. Confirm the active image layer and channel with the user.
2. Call the `segmentation` op via `ops` (see `guide://ops`)…
3. Load the returned labels with `viewer.add_labels(...)` for validation.

## Guardrails
- Prefer lazy dask; `.compute()` only the final result.
- Put intermediate results on `viewer` at each step.
```

The `url`, `sha256`, `updated`, and `spec_version` fields in the catalog are
**generated** — authors do not write them (see field policy in
[§5.1](#51-frontmatter-tolerant-read-canonical-emit)).

---

## 2. biopb-site changes (authoring + publishing)

New layout (skills live at repo root so the existing landing-page rsync serves
them from `/var/www/biopb.org/skills/` — no new hosting):

```
biopb-site/
  skills/
    cell-segmentation-cellpose.md    # curated source (frontmatter + body)
    …
    catalog.json                     # GENERATED (gitignored)
  scripts/
    build_skills_catalog.py          # normalizer + validator + generator (Appendix A/B)
  docs/skills.md                     # browser page
```

Serving falls out of the current deploy:
`skills/*.md` → `https://biopb.org/skills/<id>.md`,
`skills/catalog.json` → `https://biopb.org/skills/catalog.json`.

**CI wiring** (`.github/workflows/`):

- `docs-check.yml` (PR): run `python scripts/build_skills_catalog.py --check`.
  Malformed frontmatter, duplicate ids, a missing required section → **fail the
  PR**. This *is* the curation gate — the author gets the error, never the
  runtime agent. Tags are *not* gated: see §5.1.
- `deploy.yml` (push to main): run `python scripts/build_skills_catalog.py`
  **before** the landing rsync, so the generated `catalog.json` is in-tree at
  rsync time. The landing rsync already uploads repo root; add `PyYAML` to
  `requirements-docs.txt` (the only added dependency — validation is stdlib-only).

**Browser page** — `docs/skills.md`: a small vanilla-JS widget (Material already
enables `attr_list` / `md_in_html`) that fetches `/skills/catalog.json` and renders
a tag-filterable, searchable grid linking each `.md` and its GitHub source. Add one
`nav:` entry in `mkdocs.yml`; served at `https://biopb.org/docs/skills/`. Reuses
Material's search/nav rather than reinventing it.

---

## 3. biopb-mcp changes (discovery + retrieval)

New module **`biopb_mcp/mcp/_skills.py`**, wired into `_server.py`. Modeled on the
fail-open philosophy of `_update.py` and the discovery pattern of `_process_ops.py`.

### 3a. Discovery — a tool

A **tool** (not a resource) so it can take a query and return a tailored subset,
mirroring how `query_sources` is preferred over `list_sources`:

```python
@mcp.tool()
def find_skills(query: str = "") -> list[dict]:
    """Discover curated biopb workflows ("skills"). Call at the start of a task.

    `query` filters by title/description/tags (empty = all). Returns catalog
    metadata including the skill://<id> resource URI to read for the full
    workflow. Prefer an existing skill over improvising."""
```

### 3g. `requires:` — resolved in the kernel, by the agent

`requires:` shipped as inert metadata: emitted by `find_skills`, checked by nobody.
A skill naming a kernel plugin the install doesn't have therefore read as
*available* and dead-ended partway through its own steps. Worse, each skill body
compensated with its own prose check — a `dir()` dance in one, a `find_spec` in
another — N hand-written variants of one question, drifting apart and eating body
budget that should go to the workflow.

**`mcp/_requires.py` answers it once, as `check_skill_requirements()` in the agent's
namespace.** The agent calls it after `start_kernel`, before starting the skill;
`guide://kernel` documents it and the `find_skills` docstring points at it.

```python
check_skill_requirements(["viewer", "plugin:segmentation_qc", "pkg:basicpy>=1.2"])
# {'ok': False, 'met': ['viewer', 'pkg:basicpy>=1.2'],
#  'unmet': ["plugin:segmentation_qc — kernel plugin 'segmentation_qc' did not load…"],
#  'unknown': []}
```

**Why the kernel and not the MCP server process.** A server-side check was built
first and thrown away, for two reasons that are worth keeping written down:

1. **It can only reach a subset.** `find_skills` runs in the server process and is
   normally called *before* `start_kernel`, so the namespace, the tensor
   connection, the dask cluster and `ops` are not knowable — and the kernel need
   not even be the same interpreter (the `python3` kernelspec is not necessarily
   the tool env, which is exactly why plugins load as startup files rather than
   through entry-point metadata). Reporting the subset and staying silent on the
   rest means the agent must still check the remainder itself, so the check has
   not actually been done anywhere.
2. **It is wrong on the part it does reach.** The obvious server-side resolution
   for `plugin:<name>` is to scan `~/.config/biopb/kernel/` for `<name>.py`. But
   the loader is **fail-open per file**: a plugin that raises on `exec`, or loses a
   name to the reserved-name guard, is on disk and *not* in the namespace. "File
   present" is not "plugin loaded", and only the loader knows which happened.

So the loader **reports what survived** (`_requires.record_loaded_plugins`, called
from `_load_namespace_plugins`) and the check answers from that record, held in
module state — not in `user_ns`, where a plugin could clobber the record of itself.

| token | decided from |
|---|---|
| `viewer` | the handle (a headless kernel binds a falsy stand-in; matched by type, since a real viewer's truthiness is not ours to depend on) |
| `tensor` | the live `_conn.client`, not the per-job `client` copy |
| `dask` | the `da` handle — see below |
| `ops:<kind>` | membership in the built `ops` dict; the reason names what the servers *do* offer |
| `plugin:<name>` | the loader's record, keyed by file stem (`plugin:rolling_ball` ↔ `rolling_ball.py`) |
| `pkg:<name>[>=v]` | this interpreter's metadata, falling back to `find_spec` (a token may name the import, `skimage`, not the distribution, `scikit-image`) |

Three decisions inside that table:

- **`dask` does not mean "a distributed cluster is attached."** `da` is always
  usable; the scheduler behind it is a performance property that `server_status`
  reports. A skill saying `dask` means "this works on lazy arrays", which never
  stops being true — gating on the cluster would fail skills that run fine.
- **Version comparison is on the release tuple**, not full PEP 440 ordering, so
  `0.12.0rc8.dev32` satisfies `>=0.12.0`. A strict compare ranks every rc *below*
  its own final release and would tell anyone on a dev build to upgrade to what
  they are running. Only `>=` and `==` are understood; anything else is `unknown`,
  including an operator smuggled into the name (`pkg:biopb-mcp~=0.1`).
- **Three verdicts, not two.** `unknown` covers a token this version can't check —
  a vocabulary that grew in `biopb-site`, an unreadable version, a check that
  raised (the error text is included). Silently treating those as met would hide
  exactly the requirement the publisher went out of their way to declare; the agent
  reads a report, not a gate, and can use judgement. `ok` is `not unmet`, so an
  `unknown` never blocks.

**It informs, it never gates.** A gap is usually one command away and every fix —
installing a package, seeding a plugin, restarting the kernel — needs the user's
consent, so the reason strings name the fix and the docstring tells the agent to
ask. Nothing filters a skill out of `find_skills`, and the reason strings avoid
handing over a runnable `pip install`.

The consequence for authoring: `pkg:biopb-mcp>=X` makes a skill safe to publish
*ahead of* the release that carries the plugin it needs — an older install is told
so up front instead of failing halfway through.

### 3b. Full skill files — a dynamic resource list

At kernel/server start (and on TTL refresh), fetch the catalog and **register one
concrete resource per skill**, `skill://<id>`, then emit
`notifications/resources/list_changed`. Clients that enumerate resources then see
the curated set — the "dynamic resource list." The read handler **lazily fetches**
the body from `url`, verifies `sha256`, and caches it.

> **v1 fallback.** If dynamic registration + `list_changed` is more than we want up
> front, ship a single **resource template** `skill://{id}` instead. It won't appear
> in `resources/list` (templates list separately), but `find_skills` already hands
> the agent the exact URIs, so retrieval still works. Recommended path: template
> first (P2), upgrade to dynamic concrete resources once the catalog stabilizes (P3).

### 3c. Fetch / cache / fallback (fail-open, like `_update.py`)

- `httpx` GET the catalog with a short timeout. On **any** error
  (offline/DNS/TLS/HTTP/parse) degrade to: on-disk cache → **bundled snapshot**
  shipped in the package. Never raise into bootstrap. (The snapshot carries the
  meta-skill, so an install that has never reached the network still knows how to
  author one; refreshing it is a data drop, not a code change.)
- Cache catalog + bodies under the biopb cache dir with a TTL; `sha256` is the
  body cache key.
- Guard on `catalog_version`; an unknown future version keeps the last-good /
  bundled catalog rather than crashing.
- Treat entries defensively: unknown fields ignored, missing optionals defaulted,
  and **a single malformed entry is skipped, not fatal** — one bad skill must never
  sink `find_skills` or the resource list.

### 3d. Config (flat keys on the `services` block in `_config.py`)

As shipped, the keys are flat on `ServicesConfig` — no nested `skills` sub-block
and no `mcp.` wrapper (the config was flattened onto the tensor-server's dataclass
machinery):

```python
"services": {
    "skills_enabled": True,  # on by default
    "skills_catalog_url": "https://biopb.org/skills/catalog.json",
    "skills_cache_ttl": 3600,
    "skills_local_dir": "",  # empty -> ~/.config/biopb/skills
}
```

`skills_enabled` is **on by default** — a default install discovers skills — and
it is the switch for the *whole* subsystem: false means no fetch, no local scan,
an empty `find_skills`, and no skills directive in the handshake.

`skills_local_dir` is the **personal tier** (§3f). It is deliberately governed by
the same switch: a user who turns skills off is turning the feature off, not just
the network half.

### 3f. Local (user-authored) skills

`~/.config/biopb/skills/*.md` (`biopb._locations.mcp_skill_dir()`) are merged
into the catalog beside the curated entries. The distinction that matters: the
§3c chain is three copies of *one* catalog, so first-one-wins is right there;
local files are a **second source** and therefore union, with local winning a
shared id (a user editing their own copy of a published skill expects theirs).

- **Read on every call, not TTL-cached** — a personal skill is usually one the
  user is still editing, and an authoring loop that needs a restart to see a
  draft is unusable. Deleting the file retracts the skill just as immediately.
- **Body comes from disk**, with no `sha256` and no body cache: the file *is* the
  source of truth, so there is nothing to verify it against.
- **Consumer stays lenient** (the §"variation is a publisher problem" principle):
  a ~25-line frontmatter reader, no PyYAML in this stdlib-only module, `id`
  inferred from the filename and `description` from the first H1/prose line, so a
  bare markdown file with no frontmatter still loads. Strictness — required
  sections — stays at the publish boundary in `biopb-site`,
  which is the only place with the tooling. A second copy of those rules in the
  install would be a constant to drift, not a safeguard.
- **Fail-open per file**: one unreadable or malformed file is skipped, never
  fatal; a leading `_` marks a file private, as in the kernel-plugin loader.
- **Provenance travels**: every entry carries `origin` (`local`/`catalog`), and
  `find_skills` returns it so the agent can tell a personal draft from a
  reviewed one rather than presenting both as curated.

The three sharing tiers are then: **personal** = this directory, **lab** =
`skills_catalog_url` pointed at a self-hosted catalog, **public** = biopb.org.
Nothing else needs a sync mechanism.

### 3e. Instructions

`_SKILLS_INSTRUCTIONS` in `_server.py` carries the skills guidance:

> "At the start of a task, call `find_skills` to check for a curated workflow before
> improvising; read the matching `skill://<id>` resource for the steps. After
> accomplishing a task, ask the user whether a new skill should be generated…"

It is **appended to `_BASE_INSTRUCTIONS` only when `skills_enabled` is true**
(`set_skills_enabled`, wired from config in the launcher), so an install that
switches skills off is never pointed at a catalog that would come back empty.

---

## 4. Curation = git workflow

1. Author (often the agent, per the existing close-out prompt) drafts
   `skills/<id>.md` with frontmatter.
2. PR to `biopb-site` → `docs-check.yml` runs
   `build_skills_catalog.py --check` (schema / uniqueness / required sections) +
   `mkdocs build --strict`.
3. Human review → merge to `main` → `deploy.yml` regenerates `catalog.json` and
   publishes. Live within one deploy.
4. Versioning: author-owned `version` in frontmatter; `updated` derived from
   `git log`. The repo *is* the source of truth — no DB, no admin UI.

---

## 5. Handling format variation

The governing rule ([design principle](#design-principle--variation-is-a-publisher-problem-not-a-consumer-problem)):
absorb or reject every variation in `build_skills_catalog.py`; the runtime sees one
shape.

### 5.1 Frontmatter: tolerant read, canonical emit

A canonical model (Appendix A) coerces and infers on read, and the emitter is
strict and uniform. Field policy:

| Field | Policy on variation |
|---|---|
| `id` | **Inferable** — default to filename stem; if the author supplies one it *must* equal the stem (reject mismatch → avoids drift). |
| `description` | **Required — hard-reject if missing.** The one field discovery actually needs. |
| `title` | Fallback chain: frontmatter → first `#` H1 → humanized `id` (warn). |
| `tags` | Coerce `str → [str]` and lowercase. **Not** validated against a fixed vocabulary: a closed set needs an edit for every new topic and fails the PR introducing it, to enforce a judgment the reviewer curating the catalog is making anyway. |
| `version` | Require semver, else default `0.0.0`. |
| `updated` | **Ignore any author value**; always derive from `git log -1` — authors forget to bump it. |
| `requires` | Optional; coerce to list. |
| `spec_version` | Default `1`; selects the migration path ([§5.3](#53-versioning-the-contract)). |
| unknown keys | Collected into a `metadata` passthrough bag + warn (forward-compat), not rejected. |

Validation failures fail CI in `--check` mode, so the **author** gets the error in
the PR — never the agent at runtime.

### 5.2 Body: opaque, lint lightly

The body is freeform markdown *by design* — it is LLM context, which tolerates
prose. Do not over-constrain it. But the build:

- Normalizes mechanically: CRLF→LF, strips frontmatter, ensures a leading H1.
- **Lints as warnings, not errors**: recommend *When to use / Steps / Guardrails*;
  warn when the body references an op/tag that does not exist.
- Records `spec_version` so body *conventions* can evolve and the browser/tools can
  branch on the dialect.

### 5.3 Versioning the contract

Two independent knobs:

- **`catalog_version`** — schema of `catalog.json`. The server guards on it and
  fails open (keeps last-good/bundled) on an unknown value.
- **per-skill `spec_version`** — lets multiple authoring dialects coexist. The build
  runs **migration functions** to up-convert older dialects to the current one, so
  the *emitted* catalog is uniform even when source files lag. Rule: additive-only
  within a major; any new required field ships with a back-fill default.

### 5.4 Defensive runtime (belt and suspenders)

Even after build-time normalization, the server tolerates a bad entry: skip-and-log,
default optionals, ignore unknown fields ([§3c](#3c-fetch--cache--fallback-fail-open-like-_updatepy)).

### 5.5 Different *source* formats (only if needed — YAGNI)

If skills authored elsewhere are later imported (e.g. Claude-style `SKILL.md`
folders that bundle scripts/assets vs. biopb's single-file `.md`), add a small
**loader registry** keyed by detected shape:

- `skills/<id>.md` — single file (default)
- `skills/<id>/SKILL.md` — folder with assets; the catalog entry gains an `assets` list

Each loader maps its dialect to the one canonical model. The choke-point design means
adding this touches only the build script — don't build it until a second real format
appears.

---

## 6. Phasing

- **P0 — contract. ✅ Delivered.** Schema + stdlib frontmatter contract
  (Appendix A) and the `build_skills_catalog.py` builder/validator (Appendix B) are
  in `biopb-site`; three example skills (`load-tensor-source`, `segment-nuclei`,
  `measure-labels`) generate a real 3-entry `skills/catalog.json`. Both repos can now
  build against it. Remaining: add `PyYAML` to `requirements-docs.txt`.
- **P1 — site.** `build_skills_catalog.py` + `--check` (Appendix B), CI wiring,
  `docs/skills.md` browser.
- **P2 — MCP retrieval.** `_skills.py`: `find_skills` tool + `skill://{id}`
  template + fetch/cache/bundle fallback + config + instruction line.
- **P3 — MCP dynamic resources.** Upgrade the template to dynamically-registered
  `skill://<id>` resources with `list_changed`.
- **P4 — contribution loop.** Wire the "generate a skill?" close-out prompt to emit
  a ready-to-PR `skills/<id>.md`. **The draft lands in the local `skills_dir` first**
  (see [§7.5](#7-open-decisions)) so it is usable *this session*; promotion to the
  curated catalog is the PR, whose payload is that same file byte-for-byte. Without
  the local tier this loop does not close — a generated skill is invisible to
  `find_skills` until merged upstream, so "added to your toolbox" would be false.

---

## 7. Open decisions

1. **Discovery surface** — recommend a `find_skills` **tool** (queryable) as
   primary, with skills also exposed as resources. Alternative: resource-only
   discovery (loses query tailoring).
2. **Dynamic resources vs. template** — recommend template first (P2), dynamic
   concrete resources later (P3). If "everything in `resources/list` from day one"
   is a hard requirement, do dynamic registration in P2.
3. **Skill home** — sources live in **biopb-site** (matches "pulled from
   biopb.org"). Alternative: keep them in the biopb monorepo and have the site build
   fetch them — more moving parts; not recommended.
4. **Tag vocabulary** — resolved: free tags, curated by review. A controlled list
   was tried and removed; enforcement bought nothing the reviewer wasn't already
   doing, and cost an edit per new topic.

### 7.5 Local skills — resolved: a local *source*, not a parallel mechanism

Should the runtime merge locally-stored (uncurated) skills alongside the curated
catalog? The question is two questions with different answers, and conflating them
is what makes it look like scope creep:

- **A — lab customization** ("skills we'll never upstream"). *Already covered* by
  `catalog_url` ([§3d](#3d-config-extend-the-mcpservices-block-in-_configpy)): a lab
  points it at their own `catalog.json` and gets the full pipeline — validation,
  versioning, `find_skills`, `skill://` — for their private set, with the same review
  discipline. The only real gap is that `catalog_url` is singular. The honest fix is
  to make catalog resolution take an **ordered list of sources** (e.g.
  `[biopb.org, lab-site, local-dir]`), merged with a defined collision policy — *not*
  a separate uncurated-file feature. This is a config shape change, not a new
  subsystem.

- **B — the individual draft on-ramp** (the real motivation). This is a **P4
  concern**, not a standalone feature. The server already promises "generate a skill
  and add it to your toolbox," but under curated-only a freshly generated skill is
  useless until merged upstream — it can't appear in `find_skills` or be read as
  `skill://<id>`. A local `skills_dir` is simply *where P4's draft lands so it is
  usable this session*; promotion is the PR, whose payload is the identical file. The
  lifecycle the design gestures at — **local draft → validated in use → promoted via
  PR** — only exists if biopb-mcp owns this local tier.

Why the host's own local-skill mechanism (Claude Code / opencode / Claude Desktop)
does **not** cover B: it splits discovery (host skills never reach `find_skills`); it
can't read biopb's `requires:` capability gating; and it is host-specific, whereas a
biopb-owned local tier is one authoring format (identical to curated `.md`) that is
portable across all three hosts and is exactly the PR payload.

**Decision.** Yes, add it — scoped as *one more source in the resolution list*, not a
parallel discovery path:

1. Catalog resolution takes an **ordered source list**; one entry is a local
   `skills_dir`. (Answers A independently of P4.)
2. Local files are parsed by the **already-vendored** `skill_schema.py`
   ([§5.4](#54-defensive-runtime-belt-and-suspenders)) — same tolerant reader, no new
   parser. Entries flow into the existing in-memory merge ([§3c](#3c-fetch--cache--fallback-fail-open-like-_updatepy))
   and are marked `source: local`.
3. Collision policy is a knob; default **local-wins**, so a site can shadow a curated
   skill.
4. Ship it **with P4**, framed as "close the contribution loop," not as a P2 feature.

It is scope creep only if built as a second discovery mechanism for its own sake —
which the ordered-source-list framing explicitly avoids.

---

## 8. Development & testing

The two repos are decoupled by the `catalog.json` contract, and **the inner dev
loop needs no live site.** Each side tests hermetically; a production URL is an
*integration* concern, not a development dependency.

### 8.1 Fixtures-first (the inner loop)

- **biopb-site side** (P1) is a static generator: test it by running
  `build_skills_catalog.py` locally and `mkdocs build --strict`. The CI gate is
  `--check` on the PR. No URL is involved at any point.
- **biopb-mcp side** (P2/P3) is built not to depend on a live site (fail-open →
  cache → bundled snapshot, [§3c](#3c-fetch--cache--fallback-fail-open-like-_updatepy)).
  Point `catalog_url` at a local source — a `file://` fixture, a
  `python -m http.server` on localhost, or the bundled snapshot — plus the local
  `skills_dir` ([§7.5](#75-local-skills--resolved-a-local-source-not-a-parallel-mechanism)).
  CI should assert the fallback and `catalog_version`-guard branches here, because
  those need *deliberately-broken* catalogs, which belong in local fixtures, not on
  a host.

### 8.2 Prod-unlinked (the integration smoke)

Once P1's `--check` gate is green, let `deploy.yml` publish
`skills/catalog.json` + `skills/*.md` to biopb.org. This is harmless — static files
under `/skills/`, with no shipped consumer pointing at them until `skills.enabled`
flips — and it covers what localhost can't: real TLS, `Content-Type`, CORS (if the
browser page fetches it), httpx-against-real-cert, and the actual rsync deploy path.
Two guardrails:

1. **Publish the data, don't advertise the page.** Hold the `docs/skills.md` nav
   entry (`mkdocs.yml`) until the catalog is real, so humans don't find a
   placeholder. The JSON/MD sit at their URLs unlinked.
2. **Sequence the default flip after the publish.** Keep the shipped MCP build from
   hitting prod by default (`skills.enabled=False`, or a non-live `catalog_url`)
   until the prod catalog exists — otherwise every session start races a 404. It
   fails open silently, but boot-time 404 noise is avoidable. Order: publish the
   static catalog → then ship the consumer defaulting to it.

### 8.3 No separate testing site

A dedicated test host is real ongoing ops (DNS, TLS, another deploy target + rsync
key) for a static-file payload. Its only edge over prod-unlinked is isolating
broken/in-flight catalogs from prod — already covered twice over by the `--check`
gate (a broken catalog never reaches prod) and the server's fail-open + version
guard (a bad prod catalog degrades gracefully). The one case it would serve —
publishing malformed catalogs to exercise the consumer's fallback — is faster and
safer as local fixtures ([§8.1](#81-fixtures-first-the-inner-loop)). Not worth the
standing cost.

---

## Appendix A — canonical frontmatter contract (shipped)

Implemented as **stdlib-only** in `biopb-site/scripts/skill_schema.py` — no pydantic,
so the docs toolchain needs only PyYAML. Holds the version constants, the required
body sections, the `CatalogEntry` dataclass (what the build emits), and the
`coerce_list` helper. Excerpt:

```python
CATALOG_VERSION = 1            # schema of catalog.json; server guards, fails open
CURRENT_SPEC_VERSION = 1       # current authoring dialect (migrate() up-converts older)

# Body structure every skill must carry, as normalized H2 headings. Order is free,
# extra sections allowed. Tags are coerced but deliberately NOT gated.
REQUIRED_SECTIONS = (
    "when to use", "when not to use", "parameters",
    "steps", "failure modes", "next steps",
)
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class CatalogEntry:            # strict, canonical — what the build EMITS per skill
    id: str
    title: str
    description: str
    tags: list
    version: str
    spec_version: int
    requires: list
    updated: str              # ISO date, from git — never author-supplied
    url: str
    sha256: str
    def to_dict(self) -> dict: return asdict(self)


def coerce_list(v) -> list:   # list | "a, b" | scalar | None  ->  list
    if v is None: return []
    if isinstance(v, list): return v
    if isinstance(v, str): return [s.strip() for s in v.split(",") if s.strip()]
    return [v]
```

For defensive runtime parsing on the MCP side ([§5.4](#54-defensive-runtime-belt-and-suspenders)),
`skill_schema.py` is small enough to vendor into `biopb-mcp` as-is.

## Appendix B — build + validator (shipped)

`biopb-site/scripts/build_skills_catalog.py` (~170 lines, stdlib + PyYAML) is the
authoritative source; run it bare to generate `skills/catalog.json`, or with
`--check` in CI to validate only. Warnings never fail the build; **errors** do
(non-zero exit, catalog not written). Its `process(path)` pipeline is the choke
point — one pass per file:

1. **split** frontmatter (normalize CRLF→LF; malformed fence → error).
2. **migrate** the dialect to `CURRENT_SPEC_VERSION`.
3. **infer / coerce** (tolerant read): `id` defaults to the stem and must match it;
   `title` falls back to the first H1 then a humanized id (warn); `tags` coerced +
   lowercased (not gated); `version` checked semver; `updated`
   taken from `git log -1` (author value ignored); `description` required.
4. **emit** a canonical `CatalogEntry` with `sha256` of the raw file, or `None` on
   any error.

`main()` then dedupes by `id`, prints warnings/errors, and — only if error-free —
writes the versioned catalog. The delivered P0 run produced a 3-skill
`catalog.json` with real hashes; `--check` was verified to reject a malformed file
(missing `description`, id/stem mismatch, unknown tag, bad semver) with a non-zero
exit.

> **Note on `generated` / timestamps.** CI stamps `generated` at build time; this is
> fine for the published artifact. (This is a plain CI script — don't carry the
> determinism concern into any workflow-scripting context that forbids wall-clock calls.)
