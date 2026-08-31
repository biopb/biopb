# Skills — curated agent workflows, and how they are tested

**Component:** `biopb-mcp` — `mcp/_skills.py` (runtime), `mcp/_skills_data/*.md`
(the skills), `_tests/skills/` (the authoring gate), `_tests/bench/` (the
benchmark),
`.github/workflows/skill-contracts.yaml` (the per-package CI job).
**Related:** [`fixtures.md`](fixtures.md) — what a run is given and how it is
scored, for a skill benchmark and a task benchmark alike. The MCP `guide://*`
resources and the
server's `_BASE_INSTRUCTIONS`.

Part I (§1–§5) is what a skill *is* and how it ships. Part II (§6–§12) is what
is asserted about it, and what gates a merge.

---

# Part I — the skill

## 1. What a skill is

A **curated, reusable workflow** — "measure labeled objects in physical units",
"score a segmentation against ground truth" — as one markdown file with YAML
frontmatter, authored through a git workflow, shipped inside the package
(`mcp/_skills_data/*.md`), and consumed at runtime through a discovery **tool**
(`list_skills`) and a **resource** (`skill://<id>`). The user's own directory
(§2d) merges in beside it.

```markdown
---
id: calibrated-measurements
title: Measure labeled objects in physical units, not pixels
description: Report object areas, volumes, and diameters in microns instead of pixels, using the image's real voxel spacing.
tags: [measurement, quantification]
version: 1.0.0
checklist: [viewer, tensor, "pkg:biopb-mcp>=0.13.0"]
---

# Measure labeled objects in physical units, not pixels

## When to use
…
## Steps
1. Resolve `checklist:` against `server_status` (§3).
…
```

**The directory is the catalog and the frontmatter is the metadata.** A skill
quotes an API, assumes a namespace handle, and depends on packages resolving a
particular way — none of which is true of every release at once — so it rides
the upgrade cycle of the runtime it describes and ships inside the wheel. The
cost is that fixing a shipped skill needs a release, which is what makes the
local directory load-bearing rather than a convenience.

`id` must equal the filename stem. The rest of the contract is §4.

This closes the loop the server's own instructions gesture at — *"after a task,
ask whether a new skill should be generated and added to the agent's toolbox"* —
where the toolbox is the shipped set plus the user's own directory, and "adding"
is either a file in `~/.config/biopb/skills` or a PR.

**Two readers, with different jobs.** Skill files are authored by humans and
agents over time, and their format *will* drift.

- **`mcp/_skills.py` is tolerant.** It is on the agent's path, where a malformed
  file must degrade to a skipped entry rather than an error. It infers `id` from
  the filename and `description` from the first H1 or prose line, so a bare
  markdown file with no frontmatter still loads, and it carries no YAML
  dependency.
- **`_tests/skills/_validate.py` is strict.** It is on the author's path, where
  the same file should stop the PR, and it uses a real YAML parser precisely so
  it can reject what the other one forgives.

`test_what_validates_is_what_the_runtime_loads` pins the two to the same answer
about which files are skills, so a file only the gate can read cannot pass review
and then be invisible to the agent.

## 2. Discovery and retrieval

`mcp/_skills.py`, wired into `_server.py`.

### 2a. `list_skills` — a tool, not a resource

A tool, so it can take a query and return a tailored subset — mirroring how
`query_sources` is preferred over `list_sources`. It returns metadata dicts, each
carrying the `skill://<id>` URI to read next.

**Matching is term-wise, not whole-query.** Every whitespace-separated term must
appear somewhere in the skill's id/title/description/tags; order and adjacency do
not matter, and terms are substrings, so "measure" finds "measurements". The `id`
is in the haystack with hyphens opened out to spaces, because naming a skill
("flatfield") is the most specific request there is. Matching the whole query as
one substring could not serve the multi-word queries the tool's own docstring
offers — "stitch tiles" would miss a skill whose title says "stitch" and whose
description says "tiles". What stays out of scope is natural-language sentences,
which is why the docstring steers the agent to a few content words.

### 2b. `skill://{skill_id}` — a resource template

A template, so it does not appear in `resources/list` (templates list
separately) — but `list_skills` hands the agent exact URIs, so retrieval works.
The read handler strips frontmatter and returns the body.

### 2c. Loading and config

`load_catalog()` reads both sources on **every call** and merges them. There is
no cache: they are a handful of small local files, and re-reading is what makes a
local edit live immediately. Loading is **fail-open per file** — unreadable or
malformed is skipped and debug-logged, never fatal, and one bad skill must never
sink `list_skills`. A leading `_` marks a file private, as in the kernel-plugin
loader.

Which files in a skills directory *are* skills — the `_` marker and the prose
docs beside them — is decided in `mcp/_skills_layout.py`, and nowhere else. Four
readers ask (this loader, the authoring gate, its fixtures, and the CI package
gate), the last of them runs before an env exists and loads that module by path,
and they drifted apart the one time each kept its own copy.

```python
"services": {
    "skills_enabled": True,  # on by default
    "skills_local_dir": "",  # empty -> ~/.config/biopb/skills
}
```

`skills_enabled` is the switch for the *whole* subsystem: false means no scan, an
empty `list_skills`, and no skills directive in the handshake. It governs the
local tier too — a user who turns skills off is turning the feature off, not one
source of it.

### 2d. Local (user-authored) skills

`~/.config/biopb/skills/*.md` (`biopb._locations.mcp_skill_dir()`) merge into the
catalog beside the shipped entries, with local winning a shared id — a user
editing their own copy of a shipped skill expects theirs. Same reader, same
fail-open, body read fresh from disk at retrieval time; `updated` comes from the
file mtime. Every entry carries `origin` (`local`/`catalog`) and `list_skills`
returns it, so the agent can tell a personal draft from a reviewed one rather
than presenting both as curated.

It carries two cases at once. The **draft on-ramp**: the server promises
"generate a skill and add it to your toolbox", and a freshly generated skill is
useless until it can be retrieved — so it lands here, usable this session, and
promotion is a PR whose payload is the identical file. And **lab customization**:
the tiers are personal = this directory, public = a release, with no middle one;
a lab wanting a shared set distributes the files or vendors them into an internal
build.

A host's own skill mechanism (Claude Code, opencode, Claude Desktop) does not
cover this: it splits discovery (host skills never reach `list_skills`), it
cannot read biopb's `checklist:`, and it is host-specific — whereas one
biopb-owned local tier is a single authoring format, identical to a shipped
`.md`, portable across all three hosts, and exactly the PR payload.

### 2e. The handshake instruction

`_SKILLS_INSTRUCTIONS` in `_server.py` is appended to `_BASE_INSTRUCTIONS` **only
when `skills_enabled` is true** (`set_skills_enabled`, wired from config in the
launcher), so an install that switches skills off is never pointed at a catalog
that would come back empty.

## 3. `checklist:` — resolved by the agent, against `server_status`

The agent reads `server_status` — which it already calls before heavy work — and,
for a `pkg:` token, tries the import.

| token | resolved from |
|---|---|
| `viewer` | `## Viewer` — including the **window: CLOSED** case, where the Python handle survives but mutations no-op and `screenshot` raises |
| `tensor` | `## Tensor Server` — connected, plus the verbatim connect error when not |
| `dask` | `## Dask`. `da` is always bound, so this never fails; the scheduler behind it (distributed vs. in-process threads) is a *performance* property, and reporting it beats a met/unmet verdict |
| `ops:<kind>` | `## Ops` — and what the servers *do* offer falls out of the same line |
| `plugin:<name>` | `## Kernel plugins` — the file stem (`plugin:rolling_ball` ↔ `rolling_ball.py`) or an entry-point name, reported apart |
| `pkg:<name>[>=v\|~=v]` | `## Versions` for `pkg:biopb-mcp`, otherwise `execute_code`, in two halves: **present?** is a real `import <name>` and its real ImportError, with none of the dev-build guesswork a version comparator has to hard-code; **which version?** is `importlib.metadata.version("<name>")`, never the module's `__version__` attribute. A third-party token is bounded above as well as below, so an installed version *newer* than the range is unmet too |

**It informs, it never gates.** Nothing filters a skill out of `list_skills`, and
no return value invites `if not ok: bail`. Most of these tokens were never hard
requirements: `viewer` and `tensor` are usually two routes to the same pixels,
`dask` names a scheduler rather than a capability, and even a `pkg:` token
usually has a cruder equivalent in scipy or skimage that a competent agent
reaches for unprompted — which is what an agent is *for*. `drift-correction` says
so in prose: pystackreg is the preferred registration and
`skimage.registration.phase_cross_correlation` is "a real fallback, not a lesser
one".

So the agent checks these against the session before starting, reports what is
missing, and adapts. Every fix — installing a package, seeding a plugin,
restarting the kernel — needs the user's consent, so its job is to name the gap
and ask; the guidance lives in the `list_skills` docstring and `guide://kernel`,
at the two moments it is needed. A body that has a particular fallback in mind
should name it: the agent will improvise one otherwise, and the author's is
usually better than the invented one. That is authoring advice, not a schema
obligation, and it is where `write-a-skill` puts it.

**Why the kernel plugin line is reported, not derived.** Every other token is
legible from handles the agent already holds; this one is not, and the temptation
is to answer it by scanning `~/.config/biopb/kernel/` for `<name>.py`. That is
wrong: the loader is fail-open per file, so a plugin that raises on import — or
loses its name to the reserved-name guard — is on disk and *not* in the
namespace. Only the loader knows which happened, so it reports what survived
(`_requires.record_loaded_plugins`) and `server_status` prints that record, held
in module state rather than `user_ns` where a plugin could clobber the record of
itself. A plugin binds one name and it is the file stem, so `dir()` is a useful
cross-check — but it still cannot distinguish "never loaded" from "loaded and
then shadowed".

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

**Not every package may be declared.** A `pkg:` token whose install moves
something biopb already depends on is rejected at authoring time (§9a) —
`basicpy` reverts numpy 2.3.5 → 1.26.4, and neither that nor `m2stitch`'s pandas
downgrade errors, so the user gets an older stack silently under a live kernel
that already imported the versions being replaced. A package that needs its own
environment is an `ops:<kind>` server, called rather than imported: the kernel's
interpreter is the agent's only execution surface, so installing elsewhere is not
a resolution.

A skill that drives the session declares a `pkg:biopb-mcp>=X` floor — the first
release exposing the interface it is written against — and it is reported from
the **kernel's** interpreter, which is the one that will run the skill and need
not be the server process's env.

## 4. The file contract

The strict reader rejects at authoring time, the tolerant one degrades at
runtime, and they agree on what counts as a skill (§1). Validation failures fail
the suite, so the **author** gets the error in the PR — never the agent at
runtime.

### 4a. Frontmatter: tolerant read, strict result

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
| `checklist` | Optional; coerced to a list, grammar checked against §3's vocabulary. `requires:` is read as an alias by the runtime reader (a user's own older skill keeps its list) and rejected by the strict validator, so nothing shipped drifts back to it |
| `spec_version` | Defaults to `1`; selects the migration path (§4c) |
| `updated` | Optional. A shipped skill's currency is its release; a local one takes it from the file mtime |

### 4b. Body: opaque, linted lightly

Freeform markdown *by design* — it is LLM context, which tolerates prose. The
gate requires the H2 sections `when to use`, `when not to use`, `parameters`,
`steps` (normalized; order free, extras allowed): they are what a small model
needs and cannot infer, especially "when not to use", and every one of them is
answerable from the workflow the author has just run.

`failure modes` and `next steps` are allowed but **not required**, and the
asymmetry is deliberate. Both are only worth writing from evidence, and a
required section gets filled either way — asked for a symptom→cause→fix table
about a workflow that has not failed yet, an author writes what sounds plausible,
and once it is a table row nothing distinguishes that from a failure someone hit.
So they are treated like a regression suite: a row is added when a real failure
is observed — a contract test (§9d), a reference-implementation measurement
(§11b), a benchmark transcript (§10), or a user report — and it names the fix
that was actually applied. An empty table is a real answer. The same rule keeps
`next steps` to handoffs the steps really produce, rather than the adjacent
workflows an author can always imagine.

The gate also checks the guardrails that are mechanically checkable — no
dataset-specific paths or ids, one-sentence descriptions, `[[wiki-links]]` that
resolve, a declared `plugin:<stem>` actually called through its module name,
bodies under a ~200-line proxy.

Bodies are excluded from ruff and from the trailing-whitespace hook: they are
authored prose, two trailing spaces are a markdown hard break, and the contract
layer asserts a fence quotes a third-party call *exactly* as the body claims.

### 4c. `spec_version`

One knob, per skill: it lets authoring dialects coexist, with `migrate()`
up-converting older ones. Additive-only within a major, and any new required
field ships with a back-fill default.

Even with the gate, the runtime stays defensive: skip-and-log, default optionals,
ignore unknown fields. The gate runs on files in this repo; the local directory
has none, and that is the case the tolerance is really for.

## 5. Curation is a git workflow

1. The author — often the agent, per the close-out prompt — drafts a skill,
   usually landing it in `~/.config/biopb/skills` first so it is usable *this
   session*.
2. Promotion is a PR moving the identical file into `mcp/_skills_data/`. The
   suite gates it: schema, uniqueness, required sections, `checklist:` grammar,
   cross-skill links, phrasing coverage, package satisfiability.
3. Human review → merge → live in the next release.
4. Versioning is author-owned `version` in frontmatter. The repo *is* the source
   of truth: no DB, no admin UI.

Step 3 is the trade: a shipped skill goes live on a release, and the local
directory covers the gap in between. Testing is hermetic — there is no site to be
up and no URL to point at.

---

# Part II — testing it

A skill is a prompt fragment published to strangers' agents, and its claim is a
**behavioural delta**: an agent following it does better than one without it. But
most of what goes wrong with a skill is not behavioural — it is a sentence that
stopped being true about somebody's API. So the suite is a pyramid, and the cheap
deterministic layers carry the weight.

The layer that *is* behavioural exploits the thing that looks like an obstacle.
These skills are built around agent–human interaction — blocking confirm-input,
gates before expensive work — which seems untestable unattended. It is not:
build the fixture so the ground truth is **obtainable only by asking**, and a
numeric verifier tests the interaction for free.

## 6. The four layers, and what gates

| Layer | Question | Where it lives | Gates a merge? |
|---|---|---|---|
| **Structure** (§7) | Is the file well-formed, and does it obey the authoring rules? | `test_schema.py`, `test_validate.py`, `test_shipped_skills.py`, `test_packaging.py` | yes, in `mcp-ci` |
| **Retrieval** (§8) | Does `list_skills` surface it for the right request? | `test_retrieval.py` | yes, in `mcp-ci` |
| **Contract** (§9) | Can its packages be installed here, are they available everywhere, do they import, and does the API it quotes still exist? | `test_satisfiability.py`, `test_availability.py`, `test_contracts.py` | yes — damage per matrix cell and availability in one job, both in `mcp-ci`; the rest in `skill-contracts.yaml` |
| **Interaction** (§10) | Does a model following it produce the right numbers? | `_tests/bench/` | **no** — a benchmark; and the case *data* under it does gate |

Everything that gates is in this repo, so a skill edit and the runtime change it
depends on land in the same PR.

Markers hold work back from the default run (`biopb-mcp/pyproject.toml`
`addopts`): `satisfiability` (each token is a real resolver run; `mcp-ci` runs it
as its own step on every matrix cell), `availability` (nine resolver runs per
token; `mcp-ci` runs it once, in a job of its own), and `bench` (needs a
display, API keys and about twenty minutes). Everything else in `_tests/skills/`
and `_tests/bench/` — including every hermetic check on a case (§10d) — runs
with the ordinary suite.

**Stochastic gates get muted within two weeks of the first flake, and then you
have neither the gate nor the trust.** That is why §10's runs report rather than
fail.

## 7. Structure

Ordinary hermetic unit tests over `mcp/_skills_data/*.md`.

- `test_schema.py` — the frontmatter contract itself: list coercion, semver,
  kebab-case ids, the shape of an entry.
- `test_validate.py` — which malformations are fatal (id disagreeing with the
  filename, missing description, non-semver version, a missing required `##`
  section, unparseable YAML, an empty body) and which only warn (a missing
  title, inferred from the H1; a future spec version, clamped).
- `test_shipped_skills.py` — the authoring rules the validator does not express
  generically: the live `checklist:` vocabulary, no duplicate tokens, a
  kernel-driving skill pinning a `pkg:biopb-mcp>=X` floor, a declared
  `plugin:<stem>` actually being called through that module name,
  `[[wiki-links]]` landing on a skill that exists, and the mechanically
  checkable guardrails from `write-a-skill` (one-sentence description,
  imperative title, at least one tag, a length proxy, a first step that checks
  requirements, no body naming a specific dataset).
- `test_packaging.py` — every skill reaches the wheel, none is empty, and the
  test suite does not ship.

## 8. Retrieval

A skill nobody retrieves is not wrong, it is absent. `list_skills` filters, it
does not rank, so this splits in two — both hermetic.

**Matcher semantics**, against synthetic catalogs. What `query` means is a
contract in its own right and must not move when a description is reworded.

**A phrasing table**, against the real skills: (user phrasing → the skill it must
surface), plus **negative** cases that must not surface it. That catches a
description drifting into an implementation summary, and a new skill
cannibalising an existing one's queries. Two invariants need no table and never
go stale: every shipped skill is retrievable by its own name, and every shipped
skill appears in the phrasing table.

## 9. Contract

A skill body is an un-versioned assertion about someone else's API, and this is
where that gets checked. Breakages it exists for: a stitching call whose
`row_col_transpose=` defaults to `True` and silently swaps axes; a retired
singleton-Z axis model still described in two bodies; `np.prod(canvas) *
itemsize` as a memory estimate, ~4× under the real footprint.

The layer asks four questions in order, and the first is cheapest.

### 9a. Damage — would installing it move something already here?

`test_satisfiability.py`, marker `satisfiability`, metadata only (`uv pip install
--dry-run`); nothing is downloaded.

A skill may not declare a `pkg:` token that installs only by **moving something
already in the environment**. `uv pip install --dry-run basicpy` succeeds — and
takes numpy 2.3.5 down to 1.26.4, plus pandas and scipy, under a live kernel that
has already imported numpy 2. Nothing errors; the failure surfaces later,
somewhere else, looking like the agent's fault.

**It is a rejection, not a caveat.** It cannot be fixed in the body — the kernel
is the agent's only execution surface, so "install it in a separate venv" is not
a resolution. A package that genuinely needs its own environment belongs behind
the algorithm plane as an `ops:<kind>` server, called rather than imported. The
gate is unconditional: no allowlist, no xfail, since either would be a place to
record that a known-bad skill ships anyway.

**It runs on every `mcp-ci` matrix cell** (ubuntu 3.10/3.11/3.12, macos 3.12,
windows 3.12), because a resolution depends on the interpreter and the platform.
One red cell rejects the skill. This is the one place a `checklist:` token is
genuinely fatal: the list informs the agent everywhere else, but no amount of
agent adaptability protects a user whose numpy was silently moved under a live
kernel.

Only this repo can ask the question correctly. Resolving against PyPI answers for
the last *release*, not the branch — `biopb-mcp[mcp]` from PyPI yields
`napari==0.8.0` where the source pins `napari[all]==0.7.0`. The workspace's own
distributions are skipped: a floor on one is a statement about this repo's
release history, not about a third party.

### 9b. Availability — can it be installed on every platform we ship to?

`test_availability.py`, marker `availability`, its own `mcp-ci` job. A different
question from §9a with a different verdict.

`uv pip install --dry-run psfmodels` **succeeds** on 3.12, moving nothing — and
`psfmodels` has no cp312 wheel, so that success is a resolution to an sdist that
`--dry-run` never tries to build. The user gets a C++ compile, and on Windows
that means MSVC Build Tools. `--only-binary`, scoped to the declared package, is
what turns that into an answer. It refuses pure-Python sdists too, deliberately:
a pure-Python project essentially always ships a `py3-none-any` wheel, so
requiring one is a good proxy for "no compiler needed".

| | verdict |
|---|---|
| available everywhere | pass |
| a hole on some cells | pass, and print the cells and the resolver's reason |
| installs nowhere | **reject** |

**It reports; it does not reject.** `checklist:` informs the agent and gates
nothing (§3), so a package with no wheel on some interpreter is a gap like any
other: the agent names it and works around it. Rejecting the skill instead would
take a workflow away from the users it works for, to protect the ones an agent
can already serve. What is worth having is the *report* — an author who can see
"3 of 9, and they are all the 3.12 cells, which is what macOS and Windows users
get by default" can decide whether the body needs its fallback spelled out, and
that decision is much better made in the PR than discovered by a user.

The floor is the one case that is not a platform gap: a token no supported
session can ever satisfy means every run improvises past it while the catalog
advertises a path nobody has.

**Nine cells from one Linux job.** `uv pip compile` resolves for an interpreter
and platform that are not present, at about a second a cell, co-resolving the
token with this checkout's `biopb-mcp[mcp]` rather than the last release. That
buys two things the per-cell shape could not have: all nine combinations instead
of the matrix's five (3.10 and 3.11 are Linux-only there, so a macOS-3.10 hole
went unscreened), and **a single place that can render a verdict** — five
independent pytest runs cannot know "failed on 1 of 5", which is why
all-or-nothing was the only rule that shape could express.

What it does not screen: a *transitive* dependency that ships only an sdist.
`--only-binary` is scoped to the declared package because the workspace is a
local source tree and must still be built to co-resolve.

### 9c. Import — does the installed package work at all?

`test_contracts.py::test_every_installed_declared_package_actually_imports`, one
check over whatever the catalog declares. §9a asks whether a package can be
installed without damage, and a package can pass that and still be useless:
`uv pip install --dry-run stardist` resolves clean and moves nothing, because
`csbdeep` declares TensorFlow only under a `[tf1]` extra, and then `import
stardist` raises. The skill dead-ends at step 1 for every user.

Unlike §9d this needs no per-package authoring — it is not a claim about anyone's
API — so it runs over every declared package the env has installed. Which env
that is does not matter: `skill_contracts.py` gives each package its own, so an
absent distribution is legitimate, and a present one that will not import is
fatal on every platform.

### 9d. Signatures — is the API still what the body quotes?

`test_contracts.py`: parameter exists, default is what the prose assumes, return
shape is what the snippet unpacks. Currently manned by `drift-correction` —
`pystackreg`'s modes and `reference="previous"` default, where step 4 reads the
translation out of the transform matrix, and `skimage`'s
`phase_cross_correlation` normalization default for the degraded path.

Three properties keep it honest:

- **The work is derived from the shipped frontmatter.** The packages come out of
  the skills' own `checklist:` lists, so deleting a skill changes what this layer
  does.
- **Coverage is checked in both directions.** A shipped skill declaring a
  third-party package with nothing asserting its surface fails
  `test_every_declared_package_is_covered_here`; a `COVERED` entry naming a
  package no skill declares fails `test_covered_is_not_stale`. This layer once
  sat unmanned for a release because its only skill had been dropped, and that is
  what the pair prevents.
- **The installed version must be inside every declared range**
  (`test_the_installed_version_is_inside_every_declared_range`).

**A third-party token is bounded, not floored** — `pkg:<name>~=X.Y.Z`, PEP 440's
compatible release: a floor plus an upper bound at the next minor. The bound is
what makes the proof transferable — the assertions hold across the declared
range, and the declared range is what the agent resolves — and it is why this
layer needs no cron: the API cannot move under a shipped skill. An exact `==` pin
is wrong for the opposite reason: the agent installs into a *live* kernel, so a
pin against a user who already has a newer version is satisfiable only by the
downgrade §9a rejects. `pkg:biopb-mcp>=X` stays a bare floor; it is a statement
about this repo's own release history.

The grammar accepts no comma pair. `>=0.2.8,<0.3` is unrepresentable: the runtime
reader splits a `[a, b]` frontmatter list on every comma *before* it strips
quotes, so the token would reach the agent as two broken fragments while the
strict parser here reads it correctly — a mis-parse that passes review and
appears only in the field. `~=` says the same thing in one comma-free token.

**Its own workflow, not a step in `mcp-ci`** (`skill-contracts.yaml` +
`skill_contracts.py`): one throwaway env per declared package, over the same
five-cell matrix, behind a `paths` filter. A shared env would force every skill's
package to co-exist with every other's, so the first pair that cannot would break
the whole suite — and the reflex fix would be to drop a skill. Co-resolving one
skill package *with the workspace* is exactly what §9a certifies is safe;
co-resolving skill packages *with each other* is what the per-package envs avoid.

## 10. Interaction — the benchmark

A simulated user — an LLM with a persona and **private facts** — plus a fixture
with those facts stripped from the data. The agent under test drives a **real
biopb-mcp session** and its result is read back out of the kernel and scored by a
programmatic verifier.

> You are the microscopist. One pixel is 0.1 µm across and the z-step is 0.5 µm.
> Answer truthfully and briefly. Never volunteer anything you were not asked.

An agent that assumes cubic 1 µm voxels reports volumes wrong by 200×. An agent
that asks gets it right. **The asking needs no separate assertion** — the
information asymmetry makes correct interaction necessary for the numeric
outcome.

The verifier is **programmatic, never judged**: these skills emit numbers with
knowable right answers, so it computes a number and compares it to a limit, and
nothing reads prose. Skills listed in `cases.NOT_BENCHMARKED` are skipped, e.g.
`write-a-skill` — it emits markdown, and there is no number with a knowable right
answer.

**Withhold something categorically absent from the data** — a unit, a scale, a
provenance, an identity. A voxel size is not recoverable from an array of
numbers by any amount of looking, so a run either asks or invents, and inventing
is visible in the µm³ column by a factor of 200. Defeating the heuristics the
fixture's author thought of is a **weaker construction**:
`drift-correction`'s movie is built so contrast, peak intensity and feature
density all point at the wrong channel, and a capable agent still recovered the
answer by registering on both and keeping the self-consistent one. Such a back
door is not fatal — a run that walks through it has done something defensible
rather than something lucky, and the case then measures whether it got the
answer right by whichever route — but the case module should say which kind it
is, because it changes what a green run means.

The machinery is [`_tests/agentbench/`](../src/biopb_mcp/_tests/agentbench/),
which knows nothing about skills, and the runner is
[`_tests/bench/`](../src/biopb_mcp/_tests/bench/), which knows nothing about
them either. Every case there asks "can an agent do this work"; what a skill was
worth is the delta between two sessions either side of `--bench-skills`, read
afterwards from their `session.json`s rather than declared on the case. What a
run is given and how it is scored is [`fixtures.md`](fixtures.md).

### 10a. The agent matrix

```
BIOPB_AGENT=openai:gpt-5                    # default
BIOPB_RESPONDENT=anthropic:claude-sonnet-5  # default
```

Both sides are `provider:model` and are configured independently, with separate
base-URL overrides (`BIOPB_AGENT_BASE_URL`, `BIOPB_RESPONDENT_BASE_URL`). Known
providers: `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` — each a
`(sdk, base_url, key_env)` triple. Keys may come from the
environment or a `.env` (`BIOPB_ENV_FILE`, the repo root, then
`~/.config/biopb/harness.env`), and are never written to a trace or an artifact.

Anthropic agents were involved in the authoring of the skills and can pass by
recognising their own prose rather than by reading it, so the agent under test is
explicitly gated to be non-Anthropic. That fact lives in the provider table
rather than in prose, so `test_models.py` asserts it.

**The rule constrains the agent and only the agent.** The respondent holds a
persona and answers from a fact table; it is deliberately **skill-blind**, so it
cannot rescue a bad run by paraphrasing step 2 back at the agent, and having
written the skills does not help it do that job. Claude is therefore a fine
respondent, and is the default.

### 10b. A real session, not a stand-in tool surface

The run happens against a real shim-spawned session child: a real IPython kernel,
a real napari viewer, real dask, and the nine real tools reached over real MCP
with their own schemas and the server's own `instructions`. The body arrives
through the real `mcp/_skills.py` — `list_skills` and `skill://<id>`, the same
calls the runtime makes — so editing or deleting a skill changes what a run is
scored against.

The agent reaches `skill://<id>` through a **client-supplied** verb
(`_session.CLIENT_TOOLS`), because a resource is not a tool and the
chat-completions wire carries only tools. Without it `list_skills` returns a uri
nothing can dereference, and a skills-on run works from catalog metadata while
appearing to have the procedure. `test_session_smoke.py` asserts the body arrives
through `call` and not only through the harness's own accessor. The ablation is
unaffected: `skill://` resolves via `load_catalog()`, so an ablated run reading
the uri gets the server's "not in the catalog" answer.

#### The run must not be able to read its own answer

`execute_code` is arbitrary Python by design, so a run *can* open the fixture that
defines its answer — `truth`, the tolerances, the persona's facts — or the skill
markdown an ablated run is meant to lack. This is a validity problem, not a
security one: the agent is curious rather than adversarial, and it says what it
did in the trace. Two measures, because neither is sufficient alone.

**The child imports the shipped wheel, not the checkout.** Running from a source
tree puts `_tests/` inside the installed package, one `os.path.dirname` from any
agent that looks — and one measured run made exactly that walk. `staged_package()`
builds a wheel (which excludes `_tests`) and puts it first on the child's
`PYTHONPATH`, so the answer key is not in the process that could read it. It is
also the more faithful run: it is what a user has. Loud on failure — an unstaged
run is one whose numbers can be read off a file.

**A tripwire records what is left.** The checkout is still on disk and an
absolute path still reaches it, so a `sitecustomize` on the child's path adds an
audit hook for `open`/`os.listdir`/`os.scandir` and records hits on harness-owned
paths. It records rather than refuses: refusing would change the environment under
test, and would break the session child's own reads of `_skills_data`. The
discriminator is the *process* — the session child serves `skill://`, the kernel
is where agent code runs — and it is applied in the parent (`LiveSession.peeked`),
so the hook stays a dumb recorder. `FLAG_PEEKED` carries it onto the sample's row;
unlike the other flags it means the number is void rather than qualified.

The cost is that a red run's cause space includes the kernel, Qt, dask and the
tool schemas. Two things bound it: the trace is written before any assertion runs,
and `test_session_smoke.py` — no model, no key — fails separately, first, and for
free when the stack is what broke. A failed smoke test *skips* the benchmark
rather than merely preceding it, because a run on a broken stack does not produce
a weak result, it produces a meaningless one that reads like a weak one.

Four environment facts are **forced rather than inherited**, because each
silently changes what a run tests:

- **A GL-capable display.** On a display-less box the launcher spawns its own
  `Xvfb` and renders the viewer there, so installing the `xvfb` package is all
  such a box needs; absent both a display and the binary, bring-up probes and
  refuses. `QT_QPA_PLATFORM=offscreen` is never a substitute — napari builds and
  then `add_image` dies in vispy's extension probe.
- **No tensor plane** for an `array`-presented case. `BIOPB_TENSOR_URL` points at
  an unreachable address, so `client` lands as `None` and the agent cannot wander
  into whatever catalog the developer's machine happens to hold. A
  `tensor`-presented case gets the run-scoped plane instead ([`fixtures.md`](fixtures.md) §8).
- **A config tree of its own.** `BIOPB_CONFIG_HOME` points at a temp dir, so the
  catalog under test is the shipped set and not the developer's personal
  `~/.config/biopb/skills/*.md`.
- **Only the kernel plugins the case declares.** That same private tree means an
  empty `~/.config/biopb/kernel/`, so a skill requiring `plugin:segmentation_qc`
  would otherwise be scored where its own `checklist:` cannot be met.
  `Case.plugins` seeds the ones it names, from the copies biopb-mcp ships,
  through the real loader — and nothing else.

The fixture is injected through `session.setup()`, recorded at turn `-1`, so it
never reads as something the agent did.

### 10c. A benchmark, not a gate

A skill's claim is a behavioural delta, so measuring it needs a baseline —
which means **two runs**, not one. The configuration is two switches on the
invocation, and each corner is one command:

| | `--bench-responder=model` | `=silent` | `=briefed` |
|---|---|---|---|
| `--bench-skills=true` | does the whole thing work | does *asking* matter | what the asking cost |
| `=false` | does the *skill* matter | the floor | the fact without the skill |

`briefed` puts every fact the persona holds into the **task prompt** at handover
and answers nothing after. It is the third value because `model` and `silent`
differ in the information *and* in the exchange that obtains it, so their delta
cannot say which one it measured: against `model`, a briefed run holds the
information fixed and removes only the asking, and against `silent` it is the
worth of the fact with no conversation on either side.

Withholding is `services.skills_enabled: false` — a real shipped configuration,
so the kernel, napari, dask and every library stay as they are and only the
curated procedure goes. That the switch took effect is checked on what the
catalog *returns*, not on whether `list_skills` was called: the tool stays
registered either way and `load_catalog()` is what gates.

**One invocation is one configuration**, so no single report contains a delta:
the delta is two session directories, and `session.json` records the switches
that make them comparable. That is deliberate. An arm used to be a harness
configuration the *engine* chose per case, which meant a case's kind decided
what a run cost, and a report had to explain a table whose rows were configured
differently from one another.

**The `silent` column measures the fixture, not the skill.** Whether the
withheld fact is obtainable from the pixels is a property of the construction in
`cases/` — it does not change when a body is edited, and `test_cases.py` already
asserts the cheap half of it hermetically. The delta the layer exists to produce
is the two `--bench-responder=model` runs, so once a fixture's asymmetry is
established the silent pair is what to stop paying for. Re-run them when the
fixture changes, or when a report makes the asymmetry look decorative —
`drift-correction` is the standing reason to keep checking, since a capable agent
recovered its withheld fact anyway.

**No run's outcome fails a test.** Each sample becomes a row with an outcome and a
reason — `ok`, `wrong-answer`, `out-of-turns`, `out-of-tool-calls`, `gave-up`,
`no-result`, `unscorable-result`, `harness-error` — plus flags that change how to
read it: `cut-off-but-scored`, `over-ask-budget(n)`, `never-asked` (not on a
`briefed` row — there is nobody to ask), `asked-but-unanswered`, `stalled`,
`catalog-mismatch`. Ordering matters: a cap beats a bad number, so a run severed
mid-workflow is not reported as a wrong answer, and a *provider* failure beats
everything, because it is not about the skill at all. Every sample runs inside
its own `try`, so one that dies becomes a row instead of an exception that
destroys the other three.

Five properties of the loop, each of which cost a wrong number to find:

- **A budget failure is not a behaviour.** Both models bill their reasoning
  against `max_tokens`, so a budget that comfortably holds the answer can be spent
  before the answer starts, and what comes back is empty. On the respondent that
  looks like a hand-off, on the agent like a model with nothing left to say, and
  read either way it is scored against the skill. So the empty completion is never
  laundered into a reply: the backend raises, the loop stops as
  `respondent-failed` or `agent-truncated`, and both classify as `harness-error`.
  Every agent turn records the provider's own `finish_reason`.
- **A reasoning turn is not just text and tool calls.** Some providers return the
  reasoning alongside them and reject the *next* request if the history is
  inconsistent about it — every assistant message carries the field or none does.
  Both sides that hold history carry it, backfilling turns that predate the
  provider's first use of the key; a provider that never sends the field never
  starts receiving one. The check is flaky rather than deterministic: measured at
  3/5 accepted with the key omitted against 5/5 with it present.
- **A conversation that stops progressing is ended.** `SilentRespondent` answers
  "I don't know" to everything — including a sign-off — so it can never end a run.
  Eight consecutive turns with no tool call stop the run as `stalled`, which is
  flagged as itself rather than as a severance. Healthy runs never exceeded two;
  the two silent runs that motivated the guard trailed 42 and 55 tool-free turns
  past their last real action.
- **A question asked while acting is still a question.** Routing keyed on whether
  the turn called a tool conflates *should the user see this* with *did the agent
  block* — a model that asked its four questions in the same turn as a tool call
  had them swallowed and the run ended with nobody having been asked anything.
  Any turn carrying a question is routed whether or not it acted. The mirror is
  guarded too: `DONE` in reply to a turn that also called tools means "not a
  question to me", never "we are finished".
- **The agent declares when it is done.** The harness appends a completion
  protocol to every task — end the final message with `__BIOPB_TASK_COMPLETE__`
  alone on its own line — and the loop ends on that, before the respondent sees
  the turn. Honoured only on a tool-free turn and only as an exact final line, so
  quoting the protocol while describing it costs nothing. The respondent's `DONE`
  stays as a fallback and the stall guard as the backstop.

Two things *are* asserted, and neither judges a skill: that the report reached
disk with a transcript per sample, and that the catalog matched the switch. The second
is not a finding — if `skills_enabled: false` stopped withholding the catalog,
the delta would read as zero for a reason unrelated to the skill.

`asked` counts blocking questions against the budget `write-a-skill` step 4 sets
(at most three), and the trace records whether a question preceded the first
expensive call. Both are reported, not asserted.

**Outputs.** Per case, under `.bench-outcomes/<namespace>/<case_id>/` — the
namespace being the skill id, or `tasks` for a case that names none (override
the root with `BIOPB_OUTCOME_DIR`, gitignored): `summary.md` and `summary.json`,
and per `sample-N/` a `transcript.md`, a `trace.jsonl`, the verifier's
`summary.json`, and the case's artifacts. A run is bounded at 90 turns and 200 tool calls — generous
on purpose, since these workflows promote compute to background jobs and a cap
that stops a working run only produces unreadable results.

### 10d. One file per skill, and what gates about it

The engine (`bench/_engine.py`) owns the grid, the outcome classification and
the report, and knows no skill. The scoring vocabulary
(`agentbench/_fixture.py`) knows no skill either. A skill contributes exactly one
module under `cases/` exporting a module-level `CASE`:

```python
CASE = Case(
    skill="calibrated-measurements",
    case_id="twelve-nuclei-anisotropic",   # with `skill`, names the run
    task=TASK,                        # the prompt, incl. where results land
    persona=MICROSCOPIST,             # who holds the withheld fact
    fixture=Procedural(Ellipsoids()), # the one fixture this case owns
    layers=(Layer("nuclei", "image"),
            Layer("nuclei_labels", "labels", kind="labels")),
    collect={"volumes_um3": "volumes_um3", "spacing_um": "spacing_um"},
    score=verify,                     # (fixture, attempt) -> Outcome
    save_artifacts=save_artifacts,
    plugins=(),                       # kernel plugins the skill's checklist: names
    persona_must_know=(...), persona_must_not_know=(...),
)
```

Modules are discovered by being there — no registration line, no engine change,
no test code. `test_bench.py` parametrizes over them.

**A case does not name a skill at all.** It once did, and that field fed a
`--bench-cases` filter, a coverage ledger over the shipped catalog and a rule
about which agent could score it; all three are gone, and the bench package no
longer reads `_skills_data`. Whether an invocation is an ablation is decided by
`--bench-skills`, and whether repetition or a control carries the information is
decided by `--bench-samples`.

A **banked** skill's case is written that way too: the `_` marker keeps its
skill out of the catalog, so there is no entry to withhold and a square would be
four copies of one corner — but the work is real, and the case runs, with
`namespace=` carrying the skill's name so nothing on disk moves when it is
promoted. Those cases used to be collected into a tuple that nothing ran.
Withholding is unrelated to any of this: a case declares that it withholds
something by naming it in `persona_must_know`, and several with no `skill` do.

The hermetic checks in `test_cases.py` run in CI over every case, so a case is
checked by arriving rather than by someone remembering to write a test for it:

- **The catalogue is covered** — every shipped skill is either benchmarked or in
  `NOT_BENCHMARKED` with a reason, and nothing names a skill that no longer
  ships. Otherwise "what does this layer cover" is answerable only by reading a
  directory.
- **The persona gives nothing away** — every fact reaches the rendered prompt,
  none leaks into the freely-shared `background`, the "never volunteer"
  instruction survives, and the persona knows what the fixture strips
  (`persona_must_know`) but not the skill's own vocabulary
  (`persona_must_not_know`). A chatty respondent rescues a bad agent and a green
  run looks identical, which is why this is asserted rather than reviewed.
- **The fixture keeps its truth out of its data**, provides every layer the case
  loads, and says where it came from.
- **The verifier refuses an empty attempt** — it must report metrics, all
  unscored, each with a reason. A verifier that passed a run that left nothing
  behind would make every run look fine.
- **The task names what the harness scrapes**, since the collect names are a
  harness convention rather than something the skill asks for.
- **Presentation coverage**, which warns rather than fails
  ([`fixtures.md`](fixtures.md) §9).

`agentbench/test_conversation.py`, `test_report.py` and
`agentbench/test_fixture_protocol.py` cover the loop, the report writer and the
scoring protocol with no model and no session.

## 11. Two authoring tools, neither of them a test

Both run before the body is written, both produce evidence for the PR rather than
a green tick, and a body that has had neither is a claim nobody has tested. §11a
asks *is this content necessary*; §11b asks *is it true*.

### 11a. Ablation

Give a model the task **without** the skill, closed-book, and diff against what
the body says. Cut what it gets right unaided; keep what it gets wrong. It answers
a question about the file rather than about a run, so it is manual and per skill
edit. The procedure is `write-a-skill` step 6; three rules earned in practice:

- **Disclose the environment, withhold only the skill.** A run that withheld the
  third-party packages too had the models hand-roll everything, manufacturing
  evidence for one rule and destroying it for another. §10c's ablated run follows
  the same rule mechanically.
- **Do not ask a model what is obvious.** It introspects badly. Test behaviour,
  not self-report.
- **Use a negative control** — a condition with an irrelevant skill injected. If
  "+skill" wins as much when the skill is nonsense, the measurement is picking up
  "more context → more effort", not content.

Cross-*family* coverage beats cross-*size*: blind spots correlate within a family.
Use a weak model to ask "is this necessary" and the strongest available to ask "is
this redundant or over-constraining".

### 11b. Reference implementation

**A skill body containing a formula is code**, and no layer above runs it.
Structure checks the frontmatter, retrieval checks the description,
satisfiability and availability check the dependencies, and §9d checks *someone
else's* signatures — every one of them looks past the arithmetic the body is
asking an agent to reproduce.

**Before writing a line of the prose, implement the method and measure it against
a fixture with a known answer.** Not afterwards, as a check on what you wrote: the
measurement is what decides what the prose *says*. `flatfield` is the worked
example, and it inverted two of the author's priors — a smoothed median turned out
to be the weakest of the candidates, and the camera offset turned out to dominate
every estimator choice by an order of magnitude, which is why the body spends a
blocking checkpoint on asking for it.

The cost of skipping it is on the record: `flatfield-and-stitch-tiles` shipped a
`smoothed-median` fallback that nothing had ever executed, and a bake-off written
weeks later established it was several times off the achievable error — by which
point it had been in strangers' agents for a release.

- **Quote measured numbers, never plausible ones.** A body that says "this is more
  accurate" is unfalsifiable and ages into folklore. One that says "0.5% against
  1.8%" can be checked, argued with, and found wrong.
- **State the regime.** Every number is conditional on a construction — how many
  frames, what the specimen was doing, how strong the effect was. `flatfield`'s
  hold for a dim acquisition and a low-order field, and say so.
- **Assert the limits too, not just the capability.** The most useful thing
  `flatfield`'s measurement produced was a negative: the residual-spread check
  every author reaches for does not discriminate a good field from one four times
  worse. That became a warning in step 5 rather than a metric.
- **Cite it in the PR.** Paste the script and its output into the issue or the PR
  body, where it lives as long as the repo does. A gist has its own lifetime and
  owner, and the link rots independently of the thing it justifies.
- **A failure row records a failure, it does not predict one.** Everything an
  author can imagine going wrong is the same guesswork the prose is not allowed to
  contain, and it costs more in a table, because a row reads as something that
  happened.
- **The evidence goes in the PR; the row keeps only what outlives it.** The body
  is read by an agent with no repository and no PR — an attribution ("measured in
  the bake-off") is one it can neither verify nor act on. What transfers is the
  number the symptom is compared against and the regime or version it holds for.

**It stays out of the suite deliberately.** Executing prose needs an opt-in fence,
a per-skill checker and a threshold to maintain, and it would gate on a number
whose right value is a judgment call — real cost, carried on every CI run, against
a failure mode that is caught once, at authoring time, by the person who already
has to do the measurement. Keeping the gating layers cheap and deterministic is
what keeps them trusted (§6).

**Where the fixture ends up.** If the skill also gets a benchmark case, put the
generator in `_tests/bench/cases/<skill>.py` and let the same construction back
both, so a number quoted in the body and a tolerance set in the case mean the
same thing. `flatfield`'s does. If it gets no case, the measurement script has
done its job in the PR and does not need a home in the tree.

## 12. Running it

```sh
# everything that gates, and every hermetic check (~1.5 s)
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills

# the damage gate (§9a); CI runs this as its own step, on every matrix cell
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills -m satisfiability

# the availability grid (§9b): 9 cells per declared package, ~1 s each
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills -m availability

# the import + signature layers (§9c, §9d), which need the skill's package
uv run --no-project --python .venv/bin/python --with pystackreg \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/skills/test_contracts.py

# the benchmark (§10): a GL display (or the xvfb package — the session
# brings its own virtual display), two API keys, ~20 min per case
uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/bench -m bench -s

# the other half of a skill's delta: the same cases with the catalog withheld.
# One invocation is one configuration, so a delta is two of these (§10c)
uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/bench -m bench -s \
  --bench-skills=false
```

`-s` is not optional in practice: pytest discards a *passing* test's captured
output, so without it the engine's per-sample progress lines never appear and the
terminal sits blank for the whole run. From a second terminal, the artifact
directory is the other progress view — every sample writes its transcript before it
is scored:

```sh
watch -n5 'find .bench-outcomes -newermt "-1 hour" | sort'
```

**Adding a skill.** Drop the `.md` in `mcp/_skills_data/` — the suite discovers
the directory and applies every rule. It will ask for: a `checklist:` token inside
the vocabulary, a `[[link]]` that resolves, a one-sentence description, a
phrasing-table entry (§8), a contract test for any third-party package (§9d), and
either a benchmark case or a `NOT_BENCHMARKED` reason (§10d).

What it *cannot* ask for is the part that has to happen before the file exists: if
the body states a method, §11b says measure it first and cite the measurement in
the PR. Nothing goes red when that is skipped, which is exactly why it is written
down here and why a reviewer should ask for it.
