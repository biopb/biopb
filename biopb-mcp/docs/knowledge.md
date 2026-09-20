# Knowledge store — one flat set of docs, an agent-edited index

Status: **implemented**. Replaces `skills.md`, which is gone: Part I (what a
skill is and how it ships) is this document, and Part II's surviving layers are
[`_tests/docs/README.md`](../src/biopb_mcp/_tests/docs/README.md) and
[`_tests/bench/README.md`](../src/biopb_mcp/_tests/bench/README.md).

**Component:** `biopb-mcp` — `mcp/_docs.py` (store, index, tools),
`mcp/_docs_data/` (the shipped seed), `~/.config/biopb/docs/` (the user's own).

## 1. Why

Today the agent's knowledge is three namespaces with three access paths:
five `guide://` resources held as Python string constants, fifteen `skill://`
markdown files with a YAML dialect behind a `list_skills` tool, and kernel
plugins surfaced as a third row kind of that tool. Reading goes through MCP
resources, which hosts support unevenly. Writing has no tool at all: the agent
reads a 16 KB authoring skill, resolves a directory from `server_status`, and
writes markdown from inside the kernel through a raw-string literal. The
schema around fifteen files — spec versions, a `checklist:` grammar with its
own guide section, a shared layout module so five readers agree on what a file
is, ~2900 lines of tests and a CI workflow — is out of proportion to the
content it guards.

The content itself is not the problem. The one clean ablation on record
(sessions `20260810-172816` / `20260811-095710`, gpt-5.6-luna, 20 cases) scored
17/3 with the catalog and 12/8 without. So the redesign changes how knowledge
is stored, found and written, and keeps the shipped bodies as the seed.
Re-running that pairing on the new store is the acceptance test (§7).

## 2. The store

A doc is a markdown file. Two directories, read fresh on every access:

| tier | where | writable by the agent |
|---|---|---|
| shipped | `mcp/_docs_data/` inside the wheel | no — copy-on-write (§4) |
| local | `~/.config/biopb/docs/` (`services.docs_local_dir` moves it) | yes |

**Id is the path under the directory, minus `.md`**, and may contain `/`.
Nothing about subdirectories is interpreted today; allowing the slash now
means the store can grow a tree by naming convention later without a
migration. `index` is reserved (§3).

**A `_`-prefixed name is banked: unlisted, not withheld.** It ships, and
`read_doc` returns it by id; what it does not get is a line in the seed index,
so it stays out of the reconciliation tail (§3) and no session is told it
exists. One hand-written entry promotes it, which is what keeps banking cheap to
undo — and is why the file has to be in the wheel. Enforced in exactly one
place, `shipped_ids()`, which only the tail consumes.

This leaves two ways for a doc to go unlisted, and they belong to different
actors: the `_` prefix is *the release does not list this*, the index's
`ignored:` line is *this agent does not list this*. Neither can express the
other's decision, so the seed ships with an empty `ignored:` line and the agent
owns it outright. What a banked doc does *not* get is an assertion — the package
gates read the offered set (§7), since a doc is usually banked because its
evidence is thin, which is exactly when a green gate reads as an endorsement.

**Frontmatter is optional and small.** The reader is the tolerant one that
exists today (`_parse_frontmatter`): scalars and inline lists, unknown keys
ignored, nothing fatal. Recognised keys:

| key | who writes it | meaning |
|---|---|---|
| `description` | author, or inferred from the first paragraph | one line for the reconciliation tail (§3) |
| `kind` | shipped docs only | `reference` or `procedure` (default); the one bit the ablation switch needs (§7) |
| `packages` | shipped docs only | `name~=x.y.z, …` — the third-party APIs the body quotes, read by the contract tests (§7) and by nothing at runtime |
| `updated` | `write_doc` | ISO date of the last write |

Title is the first H1, else the id. Origin (shipped / local / local shadowing
shipped) is derived from where the file sits, never stored.

## 3. The index

The index is a doc — `index.md` in the local directory — that the agent edits
like any other. It is not derived from the files, because a derived listing
cannot carry what makes an index useful: the agent's own hooks, its grouping
and ordering, and "don't use X for Y" notes. It is the pattern harness memory
has already trained the model on.

**Seeded on first run.** The package ships an `index.md` beside the docs; when
the local directory has none, it is copied there. A fresh install starts with
the curated ordering.

**Format is free markdown with two recognised line shapes.**

```markdown
# biopb docs

## Read first
- kernel: namespace, jobs, where computes run, the user shares this kernel
- data: pyramids, laziness, axis order; read a layer with viewer.tensor()

## Procedures
- flatfield: correct uneven illumination from the image itself or a blank
- stitch-tiles: assemble a grid of overlapping tiles into one canvas
- my-lab/nuclei-protocol: the lab's nuclear segmentation, validated on the 2026 set

ignored: measure-smlm-resolution, ratiometric-fret
```

- An **entry** is a list line whose first token is an id: `- <id>: <hook>`.
  The hook is the agent's text and is rendered verbatim.
- The **`ignored:`** line names *shipped* docs the agent has decided not to
  list, so it is bounded by the size of a release. Ignore is explicit rather
  than absence, so a line lost in a sloppy rewrite reappears instead of
  silently retiring a shipped doc (§5). An ignored doc is still readable by id.

**Local docs are indexed from birth and retired by one edit.** `write_doc`
appends an entry for a doc it creates (§4), so every local doc has a line to
remove. A local file with no entry — one the agent retired, or one the user
dropped in by hand — is not surfaced anywhere; it is readable by id and it is
the user's to delete. This is the harness-memory rule: the index is the
catalog, and a file it does not name does not exist to the agent. Keeping
local docs out of the tail and the ignore line is what keeps both bounded.

Headings, prose and order are the agent's. The loader interprets nothing else.

**Rendering.** `read_doc("index")` and the handshake (§4) return the file
plus what the loader knows and the file cannot:

1. an entry whose file does not exist gets the suffix `(missing)`; one that
   shadows a shipped doc gets `(local copy)`;
2. a tail, `New shipped docs: id, id, …`, listing shipped docs with neither an
   entry nor an `ignored:` mention. Not truncated: it is bounded by what one
   release adds, and truncating it would hide the upgrade it exists to report.

The agent cleans up both by editing the file.

**Kernel plugins are not docs.** They are modules already bound in the
namespace: `server_status` lists which ones loaded and `inspect_object` reads
their docstrings, so the store does not mirror them. Today's plugin rows in
`list_skills` go with it. Plugins belong to the planned algorithm-plane
revamp, where the control manages both the `biopb.image` services and the
local Python plugin modules; that plan is not written up yet. One thing to
watch in the acceptance run (§7): those rows were added because two ablated
bench runs saw the bare plugin name in the status output and never followed
it up.

**Bounds.** At most **200 entries**; the write tool refuses an index over that
with a message saying so, which forces condensing rather than silent growth.
At the cap the rendered index is roughly 25 KB, paid once per session start.
The number of files is unbounded; only indexed ones are discoverable, which is
the bound that matters.

## 4. Tools

Two tools replace `list_skills`, `skill://{id}` and the five `guide://`
resources. Tools rather than resources because every host has tools.

**`read_doc(id)`** returns the body under a one-line header (origin, updated,
and `shadows shipped` when it does). `read_doc("index")` returns the rendered
index (§3).

**`write_doc(id, body=None, old=None, new=None)`** — `body` alone, or the
`old`/`new` pair.

- `body` creates or replaces the local file. On a *create*, the tool also
  appends `- <id>: <description>` to the index, under a trailing `## Unfiled`
  heading it adds if absent, so the doc is discoverable without a second call
  and the agent moves the line where it belongs when it next edits the index.
- `old`/`new` replaces one exact occurrence of `old` with `new`; the call is
  refused if `old` is absent or matches more than once. Index edits are the
  common case, and a full rewrite of a long index is both slow and lossy —
  models drop and paraphrase lines past ~100 of verbatim reproduction. With a
  replace, an edit costs a few dozen output tokens whatever the index length.
  This is the primitive every agent harness uses for file edits, so models
  are trained on it; a unified diff was considered and rejected because it
  needs an applier of our own (`difflib` cannot apply a patch) and its hunk
  offsets are the part models get wrong. Several edits are several calls.

There is no delete. Retiring a local doc is removing its index entry, and
retiring a shipped one is putting it on the `ignored:` line (§3): one call,
and no tool that destroys a file.

**Shipped docs are copy-on-write.** A `body` or `old`/`new` write to a shipped
id creates a local file that shadows it (the replace is applied to the shipped
text as its base). The shipped file is never touched, so an upgrade can still
replace it, and the index marks the shadow (§3).

**The result is the diff.** Whichever form was used, the tool returns
`difflib.unified_diff` of before and after, so what actually changed is
visible in the result and a replace that landed somewhere unexpected shows
immediately.

**Body cap.** A doc is read whole into context, so `write_doc` refuses a body
over 300 lines. The shipped procedures sit near 200.

**The handshake carries the index.** The MCP `instructions` field is the
server's `CLAUDE.md`: sent once per session at initialize, read by every host,
and in context before the first tool call. The rendered index goes there,
under one header line saying this is doc `index`, editable with `write_doc`
and re-readable with `read_doc`. The SDK builds the initialization options
per session, so the instructions are recomposed at each initialize and a
session sees the index as it stands. Inlined rather than "now call
`read_doc('index')`": every prompted hop loses agents (#894 is the record of
agents missing the *first* hop).

Not the start tool's return: `start_biopb` (renamed per #894; `start_kernel`
stays as an alias for one release) is also the recovery path and gets called
again after a stuck or dead kernel, and a 25 KB index resent on every retry
is cost with no information. Its return stays what it is today.

Mid-session edits are seen through `read_doc("index")`, and the `write_doc`
result shows the change. Where the host implements resource subscriptions,
the index is additionally exposed as a resource and a `resources/updated`
notification is sent on every write to it — the nearest thing MCP has to a
harness injecting recall. It is a nudge for the hosts that honour it, and
nothing in the design depends on it.

**The rest of the handshake** shrinks to: call the start tool first; the index
above lists what to read, and its *Read first* section is what to read before
non-trivial work; the standing guardrails; one sentence that destructive steps
always ask first. The skills paragraph and the checkpoint vocabulary go — the
checkpoint types move into the authoring doc and the procedures that use them.

## 5. Upgrades

The local index is the user's; the shipped set is the release's. The two never
need merging because the index references docs by id and the loader
reconciles on every render:

- a **new shipped doc** has no entry and no ignore, so it appears in the
  *New shipped docs* tail; the agent files it or ignores it, once;
- a **removed shipped doc** leaves its entry marked `(missing)`; the agent
  deletes the line;
- a **changed shipped doc** is simply read fresh; if the user had shadowed
  it, the shadow wins and the index says so.

No seen-list, no version stamp, no merge step.

## 6. Writing discipline

The authoring gate collapses to the `write_doc` docstring, in five bullets:
write only a validated multi-step procedure, never a dataset-specific one;
phrase the index hook as the user's request; **update an existing doc rather
than write a near-duplicate** — read the index first, and prefer an
`old`/`new` edit to a new file; and **verify a name, flag or call a doc quotes
still exists before relying on it**, since a local doc has no contract test
behind it. The last two are what keep an agent-written store from rotting,
and are lifted from the harness memory prompt that has proven them. A short shipped reference doc,
`authoring`, keeps what a docstring cannot hold — the checkpoint types
(confirm-input, visual check, validate-and-gate) and the derivation-rule
convention for parameters — in well under a hundred lines. `write-a-skill`'s
seven steps and its ablation ritual are dropped; the index cap and the
close-out prompt are the whole discipline for local docs. Promotion to the
shipped seed is still a PR, and that is where review happens.

**Links: one syntax, no policing.** A doc links another as `[[id]]`, and the
link resolves by `read_doc(id)` — nothing renders or rewrites it. The syntax
is fixed so links are machine-visible (search, a later "what links here",
the seed check) instead of four spellings that resolve differently across the
tiers. A dangling link in a local doc is allowed: it marks something worth
writing, and a validator the user does not have is never a condition for a
local doc to load. Shipped docs keep the existing check that every link
resolves to a shipped doc. When to link is the agent's call, with one
docstring sentence: link what this doc depends on or hands off to, not
everything it mentions. Index hooks carry no links.

## 7. Ablation and tests

**The bench switch withholds procedures, not the store.** With guides in the
same store, "docs off" would also withhold the API reference and change the
baseline arm. `--bench-docs=false` (today's `--bench-skills`) hides
`kind: procedure` docs from the rendered index and from `read_doc`; reference
docs stay. That is the only reason `kind` exists.

**What survived from `_tests/skills/`**, now `_tests/docs/`:

| layer | fate |
|---|---|
| structure (`test_schema`, `test_validate`, `_validate.py`, `_skills_layout.py`) | deleted with the schema |
| retrieval (`test_retrieval`) | replaced by `test_seed.py`: every shipped doc is either an entry of the seed index or on its `ignored:` line, every seed entry names a file that ships, every `[[link]]` lands, and nothing is over a cap |
| packaging (`test_packaging`) | kept: every doc, banked ones included, reaches the wheel — a banked doc that did not ship could not be promoted |
| contract (`test_contracts`) | kept: the hand-written API pins need no schema; the coverage fixture reads `packages:` instead of `checklist:` |
| satisfiability / availability | key off `packages:` and survive unchanged in spirit; whether availability keeps earning its CI minutes is a separate call |
| bench cases | unchanged. The switch is `--bench-docs`, and its probe reads the index and each entry's `read_doc` header rather than a `list_skills` reply |

**Acceptance.** Re-run the gpt-5.6-luna pairing with the new store, both arms.
The bar is the recorded 17/3 on the docs-on arm; the docs-off arm should not
move much, since it now keeps the reference docs it used to lose.

## 8. Migration

Content:

- the five guides become `kind: reference` docs `kernel`, `data`, `viewer`,
  `client`, `ops`, listed under *Read first*; the `## Skill requirements`
  section of the kernel guide is dropped;
- the twelve served skills become `kind: procedure` docs; `checklist:` becomes
  one prose *Requirements* line per doc, and a package that must never be
  installed is a sentence there;
- the three banked (`_`) skills stay in the repo, unshipped, as today;
- `write-a-skill` becomes `authoring` (§6).

Code and config:

- `_skills.py`, `_skills_layout.py`, `_resources.py`'s constants,
  `list_skills`, `skill://`, `guide://` are removed; `_docs.py` replaces them.
  The chat loop's synthesized `read_resource` goes with them: it existed only
  because a chat-completions API has no verb for `resources/read`, and the
  store is now ordinary tools;
- `services.skills_enabled / skills_local_dir` become
  `docs_enabled / docs_local_dir`, the old keys read as aliases for one
  release; `skills_index_plugins` is dropped with the plugin rows;
- `biopb._locations.mcp_skill_dir` gains a `mcp_docs_dir` sibling;
- `start_kernel` → `start_biopb` with the alias (#894). **Not done here** -- it
  is its own issue with a thirty-file blast radius and no coupling to the
  store.

Two departures from the plan above, both because the plan was wrong about a
detail rather than about the design:

- **`updated` is derived, not stamped.** An `old`/`new` write that also
  rewrote the frontmatter would be a replace that changed something the agent
  did not name. It is read from `updated:` where an author wrote one, and from
  the local file's mtime otherwise.
- **The kernel guide keeps its requirements section.** The `checklist:` token
  grammar went, but the rest of it -- which `server_status` section answers
  which question, and the three causes of a missing plugin -- is what every
  procedure's first step points at, and dropping it would have left twelve
  dangling pointers.

## 9. Extensibility: what v1 must not preclude

The next class of shipped content is the engine's own design docs
(`ARCHITECTURE.md`, `docs/*.md` across packages), so an agent can diagnose the
engine: 43 files, ~830 KB, 18 of them over the body cap. Not in v1 — they
need pruning first — but shipping them as forty-three index entries would
spend a fifth of the cap, flood the upgrade tail on the release that adds
them, and cost ~16k tokens to read the largest whole. The shape that fits is
**collections**: a first path segment is one unit wherever the loader counts
— one index line, one tail entry, one `ignored:` name — and
`read_doc("internals/")` returns a derived listing (id, lines, first
paragraph) instead of a body; docs over the cap return their heading outline,
and `read_doc(id, section=…)` returns one section. The listing is not the
scaling limit — one line per doc reads fine to a few hundred — but a listing
cannot answer the troubleshooting question, which starts from a literal (an
exception name, a config key, a log line). That is full-text search, and at
under a megabyte of prose it is grep: a `search_docs(pattern)` tool returning
id, line and context, in-process, no index to build. Embeddings would earn
their cost only at tens of megabytes or for questions with no literal anchor,
and design docs share their vocabulary with the errors the code raises.

None of that is built in v1. What v1 does so it stays additive:

- ids may contain `/` and map to subdirectories in both tiers, and the seed
  layout allows them;
- an id ending in `/` is reserved: refused by `write_doc`, and a `- <id>/:`
  index line is kept verbatim and counted as one entry, so a collection line
  in a seed index parses today and means something later;
- reconciliation, the cap and the `ignored:` line are keyed by id in one
  place, so grouping by first segment is a change to that place alone;
- `read_doc` keeps a single positional `id`; `section` is added, never a new
  tool, and a long-doc outline is a change to what the body returns, not to
  the call.

## 10. Open risks

1. **Whether the seed still gets read.** The ablation proved the bodies help
   under tool-then-resource delivery. Under index-in-handshake-then-`read_doc`
   delivery the read is a call the agent chooses to make. Only the acceptance
   run answers this. If it fails, the fallback is inlining the *Read first*
   bodies into the handshake as well, `CLAUDE.md`-style, at ~30 KB per
   session.
2. **Quality of unsupervised local docs.** The bench uses the seed, so it
   cannot see this. The cap bounds the size of the damage, not its rate; only
   dogfooding over weeks will show whether three bullets are enough discipline.
3. **Concurrent hand edits.** The user can edit the local directory, and the
   observe page may grow a way to. An `old`/`new` write fails safe when the
   text has moved; a `body` write from stale state clobbers. Accepted for now.
4. **Two hundred entries is a context-budget guess.** Raising it later touches
   nothing but the constant, since edits are replaces, not rewrites.
