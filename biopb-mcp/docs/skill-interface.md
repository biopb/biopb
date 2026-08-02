# Skill Interface — Curated Agent Workflows Shipped with biopb-mcp

**Status:** Implemented — the `find_skills` tool, the `skill://{skill_id}`
resource, the `services.skills_*` config, the local skills directory (§3d) and
runtime `requires:` resolution (§4).
**Component:** `biopb-mcp` — `mcp/_skills.py` (runtime), `mcp/_skills_data/`
(the skills), `_tests/skills/` (the authoring gate).
**Related:** [`skill-testing.md`](skill-testing.md) — how a skill is tested, and
what gates a merge. Also the MCP `guide://*` resources, the `services` config
block, and the server's `_BASE_INSTRUCTIONS`.

---

## 1. What a skill is

A **curated, reusable workflow** — "measure labeled objects in physical units",
"score a segmentation against ground truth" — as one markdown file with YAML
frontmatter, authored through a git workflow, shipped inside the package
(`mcp/_skills_data/*.md`), and consumed at runtime through a discovery **tool**
(`find_skills`) and a **resource** (`skill://<id>`). The user's own directory
(§3d) merges in beside it.

There is no index and no fetch: the directory is the catalog, the frontmatter is
the metadata, and a skill rides the upgrade cycle of the runtime it describes —
it quotes an API, assumes a namespace handle, and depends on packages resolving a
particular way, none of which is true of every release at once. The cost is that
a shipped skill's fix needs a release, which is what makes the local directory
load-bearing rather than a convenience.

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
1. Resolve `requires:` against `server_status` (§4).
…
```

`id` must equal the filename stem. The rest of the contract is §5.

This closes the loop the server's own instructions gesture at — *"after a task,
ask whether a new skill should be generated and added to the agent's toolbox"* —
where the toolbox is the shipped set plus the user's own directory, and "adding"
is either a file in `~/.config/biopb/skills` or a PR.

---

## 2. Two readers, tolerant and strict

Skill files are authored by humans and agents over time, and their format *will*
drift — missing fields, `tags` as a string, freeform bodies. The load-bearing
decision is that the drift is answered in **two places with different jobs**:

- **`mcp/_skills.py` is tolerant.** It is on the agent's path, where a malformed
  file must degrade to a skipped entry rather than an error. It infers `id` from
  the filename and `description` from the first H1 or prose line, so a bare
  markdown file with no frontmatter still loads, and it carries no YAML
  dependency.
- **`_tests/skills/_validate.py` is strict.** It is on the author's path, where
  the same file should stop the PR. It uses a real YAML parser precisely so it
  can reject what the other one forgives.

The two are checkable against each other:
`test_what_validates_is_what_the_runtime_loads` pins them to the same answer
about which files are skills, so a file only the gate can read cannot pass review
and then be invisible to the agent.

---

## 3. Discovery and retrieval

`mcp/_skills.py`, wired into `_server.py`.

### 3a. `find_skills` — a tool, not a resource

A tool, so it can take a query and return a tailored subset — mirroring how
`query_sources` is preferred over `list_sources`. It returns metadata dicts, each
carrying the `skill://<id>` URI to read next.

**Matching is term-wise, not whole-query.** Every whitespace-separated term must
appear somewhere in the skill's id/title/description/tags; order and adjacency do
not matter, and terms are substrings, so "measure" finds "measurements". The `id`
is in the haystack with hyphens opened out to spaces, because naming a skill
("flatfield") is the most specific request there is. Matching the whole query as
one substring could not serve the multi-word queries the tool's own docstring
offers — "stitch tiles" missed a skill whose title said "stitch" and whose
description said "tiles". What stays out of scope is natural-language sentences,
which is why the docstring steers the agent to a few content words.

### 3b. `skill://{skill_id}` — a resource template

A template, so it does not appear in `resources/list` (templates list
separately) — but `find_skills` hands the agent exact URIs, so retrieval works.
The read handler strips frontmatter and returns the body.

### 3c. Loading and config

`load_catalog()` reads both sources on **every call** and merges them. There is
no cache: they are a handful of small local files, and re-reading is what makes a
local edit live immediately. Loading is **fail-open per file** — unreadable or
malformed is skipped and debug-logged, never fatal, and one bad skill must never
sink `find_skills`. A leading `_` marks a file private, as in the kernel-plugin
loader.

```python
"services": {
    "skills_enabled": True,  # on by default
    "skills_local_dir": "",  # empty -> ~/.config/biopb/skills
}
```

`skills_enabled` is the switch for the *whole* subsystem: false means no scan, an
empty `find_skills`, and no skills directive in the handshake. It governs the
local tier too — a user who turns skills off is turning the feature off, not one
source of it.

> A config carrying `skills_catalog_url` or `skills_cache_ttl` still loads —
> unknown keys are ignored — but neither has any effect.

### 3d. Local (user-authored) skills

`~/.config/biopb/skills/*.md` (`biopb._locations.mcp_skill_dir()`) merge into the
catalog beside the shipped entries, with local winning a shared id — a user
editing their own copy of a shipped skill expects theirs. Same reader, same
fail-open, body read fresh from disk at retrieval time; `updated` comes from the
file mtime. Every entry carries `origin` (`local`/`catalog`) and `find_skills`
returns it, so the agent can tell a personal draft from a reviewed one rather
than presenting both as curated.

It carries two cases at once. The **draft on-ramp**: the server promises
"generate a skill and add it to your toolbox", and a freshly generated skill is
useless until it can be retrieved — so it lands here, usable this session, and
promotion is a PR whose payload is the identical file. And **lab customization**,
which a self-hosted catalog used to answer. The tiers are now personal = this
directory, public = a release; there is no middle one, and a lab wanting a shared
set distributes the files or vendors them into an internal build.

A host's own skill mechanism (Claude Code, opencode, Claude Desktop) does not
cover this: it splits discovery (host skills never reach `find_skills`), it
cannot read biopb's `requires:` gating, and it is host-specific — whereas one
biopb-owned local tier is a single authoring format, identical to a shipped
`.md`, portable across all three hosts, and exactly the PR payload.

### 3e. The handshake instruction

`_SKILLS_INSTRUCTIONS` in `_server.py` is appended to `_BASE_INSTRUCTIONS` **only
when `skills_enabled` is true** (`set_skills_enabled`, wired from config in the
launcher), so an install that switches skills off is never pointed at a catalog
that would come back empty.

---

## 4. `requires:` / `suggests:` — resolved by the agent, against `server_status`

`requires:` used to be metadata nothing could act on: emitted by `find_skills`,
answerable nowhere. A skill naming a kernel plugin the install did not have read
as available and dead-ended partway through its own steps, and bodies
compensated with hand-rolled prose checks. The resolution is the agent's, not a
function's: it reads `server_status` — which it already calls before heavy work —
and, for a `pkg:` token, tries the import.

| token | resolved from |
|---|---|
| `viewer` | `## Viewer` — including the **window: CLOSED** case, where the Python handle survives but mutations no-op and `screenshot` raises |
| `tensor` | `## Tensor Server` — connected, plus the verbatim connect error when not |
| `dask` | `## Dask`. `da` is always bound, so this never fails; the scheduler behind it (distributed vs. in-process threads) is a *performance* property, and reporting it beats a met/unmet verdict |
| `ops:<kind>` | `## Ops` — and what the servers *do* offer falls out of the same line |
| `plugin:<name>` | `## Kernel plugins` — the file stem (`plugin:rolling_ball` ↔ `rolling_ball.py`) or an entry-point name, reported apart |
| `pkg:<name>[>=v\|~=v]` | `## Versions` for `pkg:biopb-mcp`, otherwise `execute_code`, in two halves: **present?** is a real `import <name>` and its real ImportError, with none of the dev-build guesswork a version comparator has to hard-code; **which version?** is `importlib.metadata.version("<name>")`, never the module's `__version__` attribute. A third-party token is bounded above as well as below, so an installed version *newer* than the range is unmet too |

**Why the kernel plugin line is reported, not derived.** Every other token is
legible from handles the agent already holds; this one is not, and the temptation
is to answer it by scanning `~/.config/biopb/kernel/` for `<name>.py`. That is
wrong: the loader is fail-open per file, so a plugin that raises on import — or
loses its name to the reserved-name guard — is on disk and *not* in the
namespace. Only the loader knows which happened, so it reports what survived
(`_requires.record_loaded_plugins`, called from `_load_namespace_plugins`) and
`server_status` prints that record, held in module state rather than `user_ns`
where a plugin could clobber the record of itself. Since #664 a plugin binds one
name and it is the file stem, so `dir()` is a useful cross-check — but it still
cannot distinguish "never loaded" from "loaded and then shadowed".

**Why the version comes from metadata and not from `__version__`.** The attribute
is set by hand and drifts: `laptrack` ships `__version__ = "0.17.0"` inside its
0.17.1 distribution, so an agent resolving `pkg:laptrack>=0.17.1` off the
attribute reports the requirement unmet on a correctly installed package — and
then offers to install what is already there. `importlib.metadata.version()`
reads what the resolver actually wrote, which is also what the bound in the token
was stated against, so the two are comparable by construction. The token names a
*distribution*, and that is the argument metadata wants; the import half still
goes by module name, which is why `pkg:scikit-image` is imported as `skimage` and
read as `scikit-image`.

### `suggests:` — the same resolution, a different answer to a gap

A `requires:` gap means the skill cannot run. A `suggests:` gap means the
*preferred* path cannot run, and the body names another one — so the agent takes
the degraded path and says which was used, rather than opening the
install-or-abandon conversation `guide://kernel` prescribes for a missing
requirement. Offering the install is still fine when the user is there and the
difference matters; treating the skill as blocked is not.

The key exists because there was no way to say it. `drift-correction` declared
`pkg:pystackreg` under `requires:` while its own step 1 called the skimage
fallback "a real fallback, not a lesser one" — the frontmatter said mandatory and
the body said optional, and an agent reading the frontmatter first would stop for
a gap the skill had already answered.

It is also what lets the availability gate be per-platform at all
([`skill-testing.md`](skill-testing.md) §4b): a package with no wheel on some
interpreter is a rejection when required and a recorded hole when suggested, and
without a marker for "expected gap" there is nothing for that verdict to key on.
Two rules keep it honest — the workspace floor (`pkg:biopb-mcp>=X`) may never be
suggested, since there is no degraded path from an interface that does not exist
yet, and step 1 of the body must name the suggested package *and* say what
happens without it.

**It informs, it never gates.** Nothing filters a skill out of `find_skills`, and
no return value invites `if not ok: bail`. Every fix — installing a package,
seeding a plugin, restarting the kernel — needs the user's consent, so the
agent's job is to name the gap and ask. Guidance lives in the `find_skills`
docstring and `guide://kernel`, at the two moments it is needed, rather than in
the handshake.

**Not every package may be declared.** A `pkg:` token whose install moves
something biopb already depends on is rejected at authoring time — `basicpy`
reverts numpy 2.3.5 → 1.26.4, and neither that nor `m2stitch`'s pandas downgrade
errors, so the user gets an older stack silently under a live kernel that already
imported the versions being replaced. A package that needs its own environment is
an `ops:<kind>` server, called rather than imported: the kernel's interpreter is
the agent's only execution surface, so installing elsewhere is not a resolution.
[`skill-testing.md`](skill-testing.md) §4a is the gate.

A skill that drives the session declares a `pkg:biopb-mcp>=X` floor — the first
release exposing the interface it is written against — and it is reported from
the **kernel's** interpreter, which is the one that will run the skill and need
not be the server process's env.

---

## 5. The file contract

The strict reader rejects at authoring time, the tolerant one degrades at
runtime, and they agree on what counts as a skill (§2). Validation failures fail
the suite, so the **author** gets the error in the PR — never the agent at
runtime.

### 5a. Frontmatter: tolerant read, strict result

`_tests/skills/_schema.py` holds the canonical model (`SkillEntry`) and
`_validate.py` the one-pass pipeline that produces it: split the frontmatter
(CRLF normalized, malformed fence → error), migrate the dialect, infer and
coerce, emit an entry or `None`. `validate(dir)` dedupes by `id` and returns
`(entries, Report)`; warnings never fail, errors do.

| Field | Policy on variation |
|---|---|
| `id` | **Inferable** — defaults to the filename stem; a supplied one *must* equal the stem (reject mismatch → no drift) |
| `description` | **Required, hard reject if missing.** The one field discovery actually needs |
| `title` | Fallback chain: frontmatter → first `#` H1 → humanized `id` (warn) |
| `tags` | Coerce `str → [str]`, lowercase. **Not** gated against a vocabulary: a closed set needs an edit for every new topic and fails the PR introducing it, to enforce a judgment the reviewer is already making |
| `version` | Semver required, else default `0.0.0` |
| `requires` | Optional; coerced to a list, grammar checked against §4's vocabulary |
| `suggests` | Optional; same coercion and same vocabulary. What the *preferred* path needs. A token may not appear in both lists — the two are answers to one question, and a token in both says the skill needs it and does not |
| `spec_version` | Defaults to `1`; selects the migration path (§5c) |
| `updated` | Optional. A shipped skill's currency is its release; a local one takes it from the file mtime |

A test asserts the entry carries no `url` or `sha256` field: those described
where to fetch a body and how to verify it, and nothing fetches.

### 5b. Body: opaque, linted lightly

Freeform markdown *by design* — it is LLM context, which tolerates prose. The
gate requires the H2 sections `when to use`, `when not to use`, `parameters`,
`steps`, `failure modes`, `next steps` (normalized; order free, extras allowed):
they are what a small model needs and cannot infer, especially "when not to use"
and the symptom→cause→fix table. It also checks the guardrails that are
mechanically checkable — no dataset-specific paths or ids, one-sentence
descriptions, `[[wiki-links]]` that resolve, a declared `plugin:<stem>` actually
called through its module name, bodies under a ~200-line proxy.

Bodies are excluded from ruff and from the trailing-whitespace hook: they are
authored prose, two trailing spaces are a markdown hard break, and the contract
layer asserts a fence quotes a third-party call *exactly* as the body claims.

### 5c. `spec_version`

One knob, per skill: it lets authoring dialects coexist, with `migrate()`
up-converting older ones. Additive-only within a major, and any new required
field ships with a back-fill default.

Even with the gate, the runtime stays defensive: skip-and-log, default optionals,
ignore unknown fields. The gate runs on files in this repo; the local directory
has none, and that is the case the tolerance is really for.

---

## 6. Curation is a git workflow

1. The author — often the agent, per the close-out prompt — drafts a skill,
   usually landing it in `~/.config/biopb/skills` first so it is usable *this
   session*.
2. Promotion is a PR moving the identical file into `mcp/_skills_data/`. The
   suite gates it: schema, uniqueness, required sections, `requires:` grammar,
   cross-skill links, phrasing coverage, package satisfiability.
3. Human review → merge → live in the next release.
4. Versioning is author-owned `version` in frontmatter. The repo *is* the source
   of truth: no DB, no admin UI.

Step 3 is the trade: a shipped skill goes live on a release, and the local
directory covers the gap in between.

Testing is hermetic — there is no site to be up and no URL to point at.
`_tests/test_skills.py` covers the runtime reader against synthetic trees;
`_tests/skills/` is the authoring gate. Both run in the ordinary suite; see
[`skill-testing.md`](skill-testing.md) for the layers and what gates a merge.
