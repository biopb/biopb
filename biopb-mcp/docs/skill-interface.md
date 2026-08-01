# Skill Interface — Curated Agent Workflows Shipped with biopb-mcp

**Status:** Implemented. The `find_skills` tool, the `skill://{skill_id}`
resource, the `services.skills_*` config, the local skills directory (§3f) and
runtime `requires:` resolution (§3g) are live. The skills themselves and their
authoring gate moved into this repo (§2); the published-catalog design this
document originally described is retired (§1a).
**Component:** `biopb-mcp` — `mcp/_skills.py` (runtime), `mcp/_skills_data/`
(the skills), `_tests/skills/` (the authoring gate).
**Related:** the MCP `guide://*` resources, the `services` config block, the
dynamic op discovery in `mcp/_process_ops.py`, the server's `_BASE_INSTRUCTIONS`
("ask the user whether a new skill should be generated…"),
[`docs/skill-testing.md`](../../docs/skill-testing.md) in the repo root.

---

## Goal

Give the agent a library of **curated, reusable workflows ("skills")** — e.g.
"correct illumination and stitch a tile grid", "measure labeled objects in
physical units", "score a segmentation against ground truth". Each skill is a
markdown file with YAML frontmatter, authored and reviewed through a git
workflow, **shipped inside the package**, and consumed at runtime through:

1. a **discovery tool** (`find_skills`) that filters the catalog, and
2. a **resource** (`skill://<id>`) that returns the full workflow body.

This realizes the loop the server already gestures at in its instructions —
*"after a task, ask whether a new skill should be generated and added to the
agent's toolbox"* — where the toolbox is the shipped set plus the user's own
directory, and "adding" is either a file in `~/.config/biopb/skills` or a PR.

```
   this repo                                  a session
   ┌────────────────────────────┐             ┌────────────────────────────┐
   │ mcp/_skills_data/<id>.md   │             │  find_skills(query) TOOL   │
   │   the shipped skills       │──packaged──▶│    → filters the catalog   │
   │ _tests/skills/             │   in the    │                            │
   │   the authoring gate       │   wheel     │  skill://<id> RESOURCE     │
   └────────────────────────────┘             │    → reads the .md body    │
                                              └──────────┬─────────────────┘
   ~/.config/biopb/skills/*.md ──────merged in──────────┘
     the user's own (§3f)
```

---

## 1a. Why they ship rather than publish

The original design served one `catalog.json` from biopb.org and fetched bodies
over HTTP. That is retired. The reasoning, in full, is
[`docs/skill-testing.md`](../../docs/skill-testing.md) §9; in short:

**A skill is documentation about a specific runtime version.** It quotes an API,
assumes a namespace handle, and depends on packages resolving a particular way.
One served catalog meant every deployed version of biopb read the same text at
once — a body had to be simultaneously correct for every release in the field,
with no migration story, and a bug report could not answer which skill text a
session actually saw.

Shipping them removes that, and removes the machinery with it: the fetch, the
TTL cache, the on-disk cache, atomic writes, corrupt-cache repair, `sha256`
verification, `catalog_version` negotiation, and `catalog.json` itself. The
frontmatter *is* the metadata, and §3f's reader already parsed it, so a generated
index was only a second thing to disagree with the bodies — which is exactly what
the `sha256` check existed to catch.

**The cost: a skill fix needs a release.** §3f is the escape hatch, which makes
it load-bearing rather than a convenience.

`https://biopb.org/skills/` stays served but frozen. Older clients still fetch
it, and they fail open to their own bundled copy when it goes away.

## 1. What ships

`mcp/_skills_data/<id>.md` — frontmatter plus a markdown body written to drop
into the agent's context. There is no index file; the directory is the catalog.

```markdown
---
id: calibrated-measurements
title: Measure labeled objects in physical units, not pixels
description: Report object areas, volumes, and diameters in microns instead of pixels, using the image's real voxel spacing.
tags: [measurement, quantification]
version: 1.0.0
requires: [viewer, tensor, "pkg:biopb-mcp>=0.13.0"]
---

# Measure labeled objects in physical units, not pixels

## When to use
…

## Steps
1. Resolve `requires:` against `server_status` (§3g).
2. Confirm the active labels layer and its spacing with the user.
…
```

`id` must equal the filename stem; `spec_version` defaults to 1 and selects the
migration path (§5.3). Everything else is in the field policy at §5.1.

---

## 2. Authoring and the gate

Skills live beside the runtime they describe, so a skill edit and the code change
it depends on land in the same PR. The gate is `_tests/skills/`, an ordinary part
of the pytest suite:

```
biopb-mcp/src/biopb_mcp/
  mcp/_skills_data/<id>.md     the skills
  mcp/_skills.py               the tolerant runtime reader (§3)
  _tests/skills/
    _schema.py, _validate.py   the strict authoring reader
    test_schema.py             the frontmatter contract
    test_validate.py           which malformations are fatal vs tolerable
    test_shipped_skills.py     rules the real files must satisfy
    test_retrieval.py          do the descriptions answer real phrasings
    test_satisfiability.py     may this skill declare that package at all
    test_packaging.py          do the skills reach the wheel
```

See `_tests/skills/README.md` for what each layer asks. The layer split and what gates a merge is
[`docs/skill-testing.md`](../../docs/skill-testing.md) §1 and §10.

---

## Design principle — two readers, tolerant and strict

Skill files are authored by humans and agents over time; their format *will*
drift (missing fields, `tags` as a string vs. a list, freeform bodies). The
load-bearing decision is that the drift is answered in **two places with
different jobs**, not one:

- **`mcp/_skills.py` is tolerant.** It is on the agent's path, where a malformed
  file must degrade to a skipped entry rather than an error. It infers `id` from
  the filename and `description` from the first H1 or prose line, so a bare
  markdown file with no frontmatter still loads, and it carries no YAML
  dependency.
- **`_tests/skills/_validate.py` is strict.** It is on the author's path, where
  the same file should stop the PR. It uses a real YAML parser precisely so it
  can reject what the other one forgives.

This used to be a *publisher/consumer* split across two repos, with the strict
half at the publish boundary. Collapsing the repos did not collapse the split —
the two jobs are still different — but it did make them checkable against each
other: `test_what_validates_is_what_the_runtime_loads` pins them to the same
answer about which files are skills, so a file only the gate can read cannot pass
review and then be invisible to the agent.

See [§5](#5-handling-format-variation) for the full strategy.

---

## 3. Discovery and retrieval

`mcp/_skills.py`, wired into `_server.py`.

### 3a. Discovery — a tool

A **tool** (not a resource) so it can take a query and return a tailored subset,
mirroring how `query_sources` is preferred over `list_sources`:

```python
@mcp.tool()
def find_skills(query: str = "") -> list[dict]:
    """Discover curated biopb workflows ("skills"). Call at the start of a task.

    `query` filters by id/title/description/tags — every word must appear, in any
    order — and an empty query returns all. Returns metadata including the
    skill://<id> resource URI to read for the full workflow."""
```

**Matching is term-wise, not whole-query.** Every whitespace-separated term must
appear somewhere in the skill's id/title/description/tags; order and adjacency do
not matter. Terms are substrings, so "measure" finds "measurements". The `id` is
in the haystack with hyphens opened out to spaces, because naming a skill
("flatfield") is the most specific request there is. What this rules out is
natural-language sentences — "how do I stitch tiles?" carries terms no
description contains — which is why the tool docstring steers the agent to a few
content words. `_tests/test_skills.py` pins the semantics; `_tests/skills/test_retrieval.py`
pins that the real descriptions answer real phrasings.

### 3b. Full skill files — a resource template

`skill://{skill_id}` is a **template**. It does not appear in `resources/list`
(templates list separately), but `find_skills` hands the agent the exact URIs, so
retrieval works. The read handler strips frontmatter and returns the body.

Upgrading to dynamically-registered concrete resources with
`notifications/resources/list_changed` remains possible and is now cheaper — the
set is known at import time rather than after a fetch — but nothing depends on
it.

### 3c. Loading

`load_catalog()` reads two sources on **every call** and merges them (§3f). There
is no cache: the reads are a handful of small local files, and re-reading is what
makes a local edit live immediately. Loading is **fail-open per file** — an
unreadable or malformed file is skipped and debug-logged, never fatal, and one
bad skill must never sink `find_skills`. A leading `_` marks a file private, as in
the kernel-plugin loader.

### 3d. Config (flat keys on the `services` block in `_config.py`)

```python
"services": {
    "skills_enabled": True,  # on by default
    "skills_local_dir": "",  # empty -> ~/.config/biopb/skills
}
```

`skills_enabled` is **on by default** — a default install discovers skills — and
it is the switch for the *whole* subsystem: false means no scan, an empty
`find_skills`, and no skills directive in the handshake.

`skills_local_dir` is the **personal tier** (§3f). It is deliberately governed by
the same switch: a user who turns skills off is turning the feature off, not just
one source of it.

> `skills_catalog_url` and `skills_cache_ttl` were removed with the fetch (§1a).
> A config carrying them still loads — unknown keys are ignored — but a
> self-hosted catalog URL has no effect.

### 3f. Local (user-authored) skills

`~/.config/biopb/skills/*.md` (`biopb._locations.mcp_skill_dir()`) are merged
into the catalog beside the shipped entries, with local winning a shared id (a
user editing their own copy of a shipped skill expects theirs).

- **Read on every call** — a personal skill is usually one the user is still
  editing, and an authoring loop that needs a restart to see a draft is unusable.
  Deleting the file retracts the skill just as immediately.
- **Body comes from disk**, read fresh at retrieval time.
- **Same reader as the shipped set.** Both are markdown files with frontmatter,
  and reading them two ways would be two things to keep agreeing.
- **Fail-open per file** (§3c).
- **Provenance travels**: every entry carries `origin` (`local`/`catalog`), and
  `find_skills` returns it so the agent can tell a personal draft from a reviewed
  one rather than presenting both as curated.

The sharing tiers are now: **personal** = this directory, **public** = a release.
The middle tier a self-hosted catalog used to serve is gone with the fetch; a lab
wanting a shared set distributes the files, or vendors them into an internal
build. This is the load-bearing cost of §1a, and it is also the only path by
which a skill reaches a machine between releases.

### 3e. Instructions

`_SKILLS_INSTRUCTIONS` in `_server.py` carries the skills guidance:

> "At the start of a task, call `find_skills` to check for a curated workflow before
> improvising; read the matching `skill://<id>` resource for the steps. After
> accomplishing a task, ask the user whether a new skill should be generated…"

It is **appended to `_BASE_INSTRUCTIONS` only when `skills_enabled` is true**
(`set_skills_enabled`, wired from config in the launcher), so an install that
switches skills off is never pointed at a catalog that would come back empty.

### 3g. `requires:` — resolved by the agent, against `server_status`

`requires:` shipped as metadata nothing could act on: emitted by `find_skills`,
answerable nowhere. A skill naming a kernel plugin the install doesn't have read
as available and dead-ended partway through its own steps, and skill bodies
compensated with hand-rolled prose checks — a `dir()` dance in one, a `find_spec`
in another.

The resolution is the agent's, not a function's: it reads `server_status` (which
it already calls before heavy work) and, for a `pkg:` token, tries the import.

| token | resolved from |
|---|---|
| `viewer` | `## Viewer` — including the **window: CLOSED** case, where the Python handle survives but mutations no-op and `screenshot` raises |
| `tensor` | `## Tensor Server` — connected, plus the verbatim connect error when not |
| `dask` | `## Dask`. `da` is always bound, so this never fails; the scheduler behind it (distributed vs. in-process threads) is a *performance* property, and reporting it is more useful than a met/unmet verdict on it |
| `ops:<kind>` | `## Ops` — and what the servers *do* offer falls out of the same line |
| `plugin:<name>` | `## Kernel plugins` — the file stem (`plugin:rolling_ball` ↔ `rolling_ball.py`) or an entry-point name, reported apart |
| `pkg:<name>[>=v\|~=v]` | `## Versions` for `pkg:biopb-mcp` (the token authors actually reach for — see below — so the report carries it), otherwise `import <name>` in `execute_code`: a real ImportError or a real `__version__`, with none of the dev-build/`skimage`-vs-`scikit-image` guesswork a version comparator has to hard-code. A third-party token is `~=` — bounded above as well as below (`docs/skill-testing.md` §3b) — so an installed version *newer* than the range is unmet too, and the answer is the skill's degraded path, never a downgrade: the kernel is live and the package is already imported |

**Why the kernel plugin line has to be reported, not derived.** Every other token
is legible from handles the agent already holds; this one is not, and the
temptation is to answer it by scanning `~/.config/biopb/kernel/` for `<name>.py`.
That is wrong: the loader is **fail-open per file**, so a plugin that raises on
import — or loses its name to the reserved-name guard — is on disk and *not* in
the namespace. Only the loader knows which happened, so it **reports what
survived** (`_requires.record_loaded_plugins`, called from
`_load_namespace_plugins`) and `server_status` prints that record — held in module
state, not `user_ns`, where a plugin could clobber the record of itself.

Since #664 the namespace at least *agrees* with the token: a plugin binds one name
and it is the file stem, so a loaded `segmentation_qc.py` does appear in `dir()` as
`segmentation_qc`. That makes `dir()` a useful cross-check, not a substitute — it
still cannot distinguish "never loaded" from "loaded and then shadowed", and it
says nothing about why.

Guidance lives in the `find_skills` docstring and `guide://kernel`, at the two
moments it is needed, rather than in the handshake instructions (a per-session
context tax for something that matters only once a skill is retrieved).

**It informs, it never gates.** Nothing filters a skill out of `find_skills`, and
no return value invites `if not ok: bail`. Every fix — installing a package,
seeding a plugin, restarting the kernel — needs the user's consent, so the agent's
job is to name the gap and ask, not to decide.

**A `pkg:` token is not only a runtime question, and not every package may be
declared.** `test_satisfiability.py` resolves every `pkg:` a shipped skill
declares against the installed environment and **rejects** the skill when the
package installs only by moving something biopb already depends on. That is not
hypothetical: `basicpy` reverts numpy 2.3.5 → 1.26.4 (and pandas and scipy with
it), and `m2stitch` takes pandas 3.0.3 → 2.3.3. Neither errors, so the agent and
the user get an older stack silently — under a live kernel that already imported
the versions being replaced.

A warning in the body does not fix it: the three options above are general
guidance, and option 2 is the harmful one. Nor does a separate environment —
the kernel's interpreter is the agent's only execution surface, so a package
installed elsewhere cannot be imported. **A package that needs its own
environment is an `ops:<kind>` server**, called rather than imported, which is
what the algorithm plane is for. `flatfield-and-stitch-tiles` declared
`pkg:basicpy` and `pkg:m2stitch` and was dropped rather than shipped with a
workaround.

**`pkg:biopb-mcp>=X` is now an open question.** It made a skill safe to publish
*ahead of* the release carrying the plugin it needs — an older install was told
so up front instead of failing halfway. That was a cross-repo version bound.
Within one release the skill and the runtime ship together, so it is either
redundant or a statement about backwards compatibility, and which is not yet
settled (`docs/skill-testing.md` §11). It is still validated and still reported
from the **kernel's** interpreter, which is the one that will run the skill and
need not be the server process's env.

---

## 4. Curation = git workflow

1. Author (often the agent, per the existing close-out prompt) drafts a skill,
   usually landing it in `~/.config/biopb/skills` first so it is usable *this
   session* (§7.5).
2. Promotion is a PR that moves the identical file into `mcp/_skills_data/`.
   The suite gates it: schema, uniqueness, required sections, `requires:`
   grammar, cross-skill links, phrasing coverage.
3. Human review → merge. Live in the next release.
4. Versioning: author-owned `version` in frontmatter. The repo *is* the source of
   truth — no DB, no admin UI.

Step 3 is the change §1a bought and paid for: skills are no longer live within
one deploy. What used to be a same-day publish is now a release, and the local
directory is what covers the gap.

---

## 5. Handling format variation

The governing rule ([design principle](#design-principle--two-readers-tolerant-and-strict)):
the strict reader rejects at authoring time, the tolerant one degrades at
runtime, and they agree on what counts as a skill.

### 5.1 Frontmatter: tolerant read, strict result

The canonical model (Appendix A) coerces and infers on read; what it yields is
uniform. Field policy:

| Field | Policy on variation |
|---|---|
| `id` | **Inferable** — default to filename stem; if the author supplies one it *must* equal the stem (reject mismatch → avoids drift). |
| `description` | **Required — hard-reject if missing.** The one field discovery actually needs. |
| `title` | Fallback chain: frontmatter → first `#` H1 → humanized `id` (warn). |
| `tags` | Coerce `str → [str]` and lowercase. **Not** validated against a fixed vocabulary: a closed set needs an edit for every new topic and fails the PR introducing it, to enforce a judgment the reviewer curating the catalog is making anyway. |
| `version` | Require semver, else default `0.0.0`. |
| `requires` | Optional; coerce to list. Grammar checked at §3g's vocabulary. |
| `spec_version` | Default `1`; selects the migration path ([§5.3](#53-versioning-the-contract)). |
| `updated` | Optional and rarely used. It was derived from `git log` to stamp the generated index; with no index, a shipped skill's currency is its release. The runtime still reads an explicit `updated:` if present, and a *local* skill takes it from the file mtime. |

Validation failures fail the suite, so the **author** gets the error in the PR —
never the agent at runtime.

### 5.2 Body: opaque, lint lightly

The body is freeform markdown *by design* — it is LLM context, which tolerates
prose. Do not over-constrain it. But the gate:

- Requires the H2 sections at Appendix A: they are what a small model needs and
  cannot infer, especially "when not to use" and the symptom→cause→fix table.
- Checks the authoring guardrails that are mechanically checkable: no
  dataset-specific paths or ids, one-sentence descriptions, `[[wiki-links]]`
  landing on a skill that exists, a declared `plugin:<stem>` actually called
  through its module name, bodies under the ~200-line proxy.
- Records `spec_version` so body *conventions* can evolve.

Bodies are excluded from ruff and from the trailing-whitespace hook: they are
authored prose, two trailing spaces are a markdown hard break, and the contract
layer asserts a fence quotes a third-party call *exactly* as the body claims.

### 5.3 Versioning the contract

One knob now, not two. `catalog_version` described the schema of a file that a
server fetched from a publisher that might be newer than it — there is no such
file and no such skew, so it is gone.

**Per-skill `spec_version`** remains: it lets multiple authoring dialects coexist,
with `migrate()` up-converting older ones. Rule: additive-only within a major;
any new required field ships with a back-fill default.

### 5.4 Defensive runtime

Even with an authoring gate, the runtime tolerates a bad file: skip-and-log,
default optionals, ignore unknown fields (§3c). The gate runs on files in this
repo; the user's directory (§3f) has no gate at all, and that is the case the
tolerance is really for.

### 5.5 Different *source* formats (only if needed — YAGNI)

If skills authored elsewhere are later imported (e.g. Claude-style `SKILL.md`
folders that bundle scripts/assets vs. biopb's single-file `.md`), add a small
**loader registry** keyed by detected shape:

- `<id>.md` — single file (default)
- `<id>/SKILL.md` — folder with assets; the entry gains an `assets` list

Each loader maps its dialect to the one canonical model. Don't build it until a
second real format appears.

---

## 6. Phasing

- **P0 — contract. ✅** Schema + frontmatter contract (Appendix A) and the
  validator (Appendix B).
- **P1 — publishing.** ✅ then retired (§1a). The site build, the generated
  `catalog.json`, and the CI wiring existed and worked; shipping the files
  removed the need for all of it.
- **P2 — MCP retrieval. ✅** `_skills.py`: `find_skills` + `skill://{id}` +
  config + instruction line.
- **P3 — MCP dynamic resources.** Not started; cheaper now (§3b) and still
  optional.
- **P4 — contribution loop.** Partly there: the local tier (§3f) exists, so a
  generated skill is usable this session. Remaining is wiring the close-out
  prompt to emit a ready-to-PR file into it.

---

## 7. Open decisions

1. **Discovery surface** — resolved: a `find_skills` **tool** (queryable) as
   primary, skills also readable as resources.
2. **Dynamic resources vs. template** — template shipped (§3b); dynamic
   registration remains optional.
3. **Skill home** — **resolved, opposite to the original recommendation.**
   Sources were in `biopb-site` to match "pulled from biopb.org". They are now in
   this repo. The deciding argument was not tidiness but testability: a skill's
   claims are about *this* runtime — the API it quotes, the packages it needs
   resolving against this workspace — and no other repo can check them. Resolving
   `biopb-mcp[mcp]` from PyPI answers for the last release, not the branch. See
   `docs/skill-testing.md` §3a and §9.
4. **Tag vocabulary** — resolved: free tags, curated by review. A controlled list
   was tried and removed; enforcement bought nothing the reviewer wasn't already
   doing, and cost an edit per new topic.
5. **`pkg:biopb-mcp>=X`** — open. See §3g.
6. **Release cadence vs. skill churn** — open. §1a trades hot-fix for coherence,
   and §3f only covers it per-machine. If skill edits outpace releases by much,
   revisit.

### 7.5 Local skills — resolved: a local *source*, not a parallel mechanism

Should the runtime merge locally-stored (uncurated) skills alongside the curated
set? Yes, and the reason has strengthened. The original argument was the **draft
on-ramp**: the server promises "generate a skill and add it to your toolbox," and
under curated-only a freshly generated skill is useless until merged upstream —
it can't appear in `find_skills` or be read as `skill://<id>`. The local dir is
where that draft lands so it is usable this session; promotion is the PR, whose
payload is the identical file.

The second argument, **lab customization**, used to be answered by pointing
`catalog_url` at a self-hosted catalog. That answer is gone with the fetch, so
the local dir now carries both cases, and it is the only mechanism by which any
skill reaches a machine outside a release.

Why the host's own local-skill mechanism (Claude Code / opencode / Claude
Desktop) does **not** cover this: it splits discovery (host skills never reach
`find_skills`); it can't read biopb's `requires:` capability gating; and it is
host-specific, whereas a biopb-owned local tier is one authoring format
(identical to a shipped `.md`) that is portable across all three hosts and is
exactly the PR payload.

---

## 8. Development & testing

Hermetic, and now trivially so — there is no site to be up and no URL to point
at. `_tests/test_skills.py` covers the runtime reader against synthetic trees;
`_tests/skills/` is the authoring gate. Both run in the ordinary suite:

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests
```

Two layers are marked and deselected by default — `contract` (needs
basicpy/m2stitch, several GB) and `satisfiability` (a real resolver run per
package, run as its own CI step). `_tests/skills/README.md` has both.

The layer split, what gates a merge, and the agent-facing tiers that are *not*
gates live in [`docs/skill-testing.md`](../../docs/skill-testing.md).

---

## Appendix A — canonical frontmatter contract

**stdlib + PyYAML**, in `_tests/skills/_schema.py`. Holds the version constant,
the required body sections, the `SkillEntry` dataclass, and `coerce_list`:

```python
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
class SkillEntry:              # strict, canonical — what validation yields
    id: str
    title: str
    description: str
    tags: list
    version: str
    spec_version: int
    requires: list
    def to_dict(self) -> dict: return asdict(self)


def coerce_list(v) -> list:   # list | "a, b" | scalar | None  ->  list
    if v is None: return []
    if isinstance(v, list): return v
    if isinstance(v, str): return [s.strip() for s in v.split(",") if s.strip()]
    return [v]
```

`url` and `sha256` are gone: they said where to fetch a body and how to verify
it. `CATALOG_VERSION` went with the file it described. A test asserts the entry
carries no fetch fields, so re-adding one would mean the fetch came back.

The runtime does **not** import this. `mcp/_skills.py` has its own ~25-line
frontmatter reader — deliberately weaker, stdlib-only, and forgiving where this
one rejects (see the design principle above).

## Appendix B — the validator

`_tests/skills/_validate.py` (~170 lines, stdlib + PyYAML). Its `process(path)`
pipeline is the choke point — one pass per file:

1. **split** frontmatter (normalize CRLF→LF; malformed fence → error).
2. **migrate** the dialect to `CURRENT_SPEC_VERSION`.
3. **infer / coerce** (tolerant read): `id` defaults to the stem and must match
   it; `title` falls back to the first H1 then a humanized id (warn); `tags`
   coerced + lowercased (not gated); `version` checked semver; `description`
   required.
4. **emit** a canonical `SkillEntry`, or `None` on any error.

`validate(dir)` dedupes by `id` and returns `(entries, Report)`. Warnings never
fail; **errors** do. It writes nothing — it used to end by generating
`catalog.json`, and pytest is now the gate that used to be a CLI.

`Report` is per-run rather than module state, because the suite calls the
validator many times in one process and a global accumulator would carry one
run's errors into the next.
