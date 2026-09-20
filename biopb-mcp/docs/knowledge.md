# Knowledge store — one flat set of docs, an agent-edited index

Status: **proposed**. Supersedes [`skills.md`](skills.md) Part I (what a skill is
and how it ships). Part II's contract layer survives, scoped in §7.

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
Re-running that pairing on the new store is the acceptance test (§9).

## 2. The store

A doc is a markdown file. Two directories, read fresh on every access:

| tier | where | writable by the agent |
|---|---|---|
| shipped | `mcp/_docs_data/` inside the wheel | no — copy-on-write (§4) |
| local | `~/.config/biopb/docs/` (`services.docs_local_dir` moves it) | yes |

**Id is the path under the directory, minus `.md`**, and may contain `/`.
Nothing about subdirectories is interpreted today; allowing the slash now
means the store can grow a tree by naming convention later without a
migration. `index` is reserved (§3). A shipped file whose name starts with `_`
is not shipped — the one surviving spelling of "banked, not served".

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
- The **`ignored:`** line names docs, shipped or local, the agent has decided
  not to list. It is the only removal there is: the agent never deletes a
  file, so retiring a doc is one index edit, and a local file the user wants
  gone is theirs to remove. Ignore is explicit rather than absence, so a line
  lost in a sloppy rewrite reappears instead of silently retiring a doc (§5).
  An ignored doc is still readable by id.

Headings, prose and order are the agent's. The loader interprets nothing else.

**Rendering.** `read_doc("index")` and the start tool (§4) return the file
plus what the loader knows and the file cannot:

1. an entry whose file does not exist gets the suffix `(missing)`; one that
   shadows a shipped doc gets `(local copy)`;
2. a tail, `Unindexed (N): id, id, …`, listing docs with neither an entry nor
   an `ignored:` mention — at most the first 10 ids, then the count.

The agent cleans up both by editing the file.

**Kernel plugins are not docs.** They are modules already bound in the
namespace: `server_status` lists which ones loaded and `inspect_object` reads
their docstrings, so the store does not mirror them. Today's plugin rows in
`list_skills` go with it. Plugins belong to the planned algorithm-plane
revamp, where the control manages both the `biopb.image` services and the
local Python plugin modules; that plan is not written up yet. One thing to watch in the acceptance run (§9): those
rows were added because two ablated bench runs saw the bare plugin name in
the status output and never followed it up.

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

**`write_doc(id, body=None, diff=None)`** — exactly one of the two.

- `body` creates or replaces the local file.
- `diff` is a unified diff applied to the current text. Index edits are the
  common case, and a full rewrite of a long index is both slow and lossy —
  models drop and paraphrase lines past ~100 of verbatim reproduction. With a
  diff, an edit costs a few dozen output tokens whatever the index length.

There is no delete. Removing a doc is removing its index entry, or putting it
on the `ignored:` line if it would otherwise resurface in the tail (§3): one
call instead of two, and no tool that destroys a file.

**Shipped docs are copy-on-write.** A `body` or `diff` write to a shipped id
creates a local file that shadows it (the diff is applied to the shipped text
as its base). The shipped file is never touched, so an upgrade can still
replace it, and the index marks the shadow (§3).

**The diff applier is ours.** `difflib` produces and compares diffs but does
not apply them, and `patch` is not portable. The applier parses unified hunks,
**ignores the `@@` line numbers** (models get context right and offsets
wrong), locates each hunk by its context and removed lines, requires the match
to be unique, and applies the whole diff atomically or refuses it naming the
hunk that failed. About fifty lines. The tool returns `difflib.unified_diff`
of before and after, so a mangled hunk is visible in the result.

**Body cap.** A doc is read whole into context, so `write_doc` refuses a body
over 300 lines. The shipped procedures sit near 200.

**The start tool returns the index.** `start_biopb` (renamed per #894;
`start_kernel` stays as an alias for one release) appends the rendered index
to its ready text, after the display warning, under one header line saying
this is doc `index`, editable with `write_doc`. Inlined rather than "now call
`read_doc('index')`": every prompted hop loses agents (#894 is the record of
agents missing the *first* hop), and the index has to enter context anyway,
so inlining costs nothing when the agent complies and saves the round trip.

**The handshake** shrinks to: call the start tool first; the index in its
return lists what to read, and its *Read first* section is what to read before
non-trivial work; the standing guardrails; one sentence that destructive steps
always ask first. The skills paragraph and the checkpoint vocabulary go — the
checkpoint types move into the authoring doc and the procedures that use them.

## 5. Upgrades

The local index is the user's; the shipped set is the release's. The two never
need merging because the index references docs by id and the loader
reconciles on every render:

- a **new shipped doc** has no entry and no ignore, so it appears in the
  unindexed tail; the agent files it or ignores it, once;
- a **removed shipped doc** leaves its entry marked `(missing)`; the agent
  deletes the line;
- a **changed shipped doc** is simply read fresh; if the user had shadowed
  it, the shadow wins and the index says so.

No seen-list, no version stamp, no merge step.

## 6. Writing discipline

The authoring gate collapses to the `write_doc` docstring, in three bullets:
write only a validated multi-step procedure, never a dataset-specific one, and
phrase the index hook as the user's request. A short shipped reference doc,
`authoring`, keeps what a docstring cannot hold — the checkpoint types
(confirm-input, visual check, validate-and-gate) and the derivation-rule
convention for parameters — in well under a hundred lines. `write-a-skill`'s
seven steps and its ablation ritual are dropped; the index cap and the
close-out prompt are the whole discipline for local docs. Promotion to the
shipped seed is still a PR, and that is where review happens.

## 7. Ablation and tests

**The bench switch withholds procedures, not the store.** With guides in the
same store, "docs off" would also withhold the API reference and change the
baseline arm. `--bench-docs=false` (today's `--bench-skills`) hides
`kind: procedure` docs from the rendered index and from `read_doc`; reference
docs stay. That is the only reason `kind` exists.

**What survives from `_tests/skills/`.**

| layer | fate |
|---|---|
| structure (`test_schema`, `test_validate`, `_validate.py`, `_skills_layout.py`) | deleted with the schema |
| retrieval (`test_retrieval`) | replaced by one invariant: every shipped doc is either an entry of the seed index or on its `ignored:` line, and every seed entry names a file that ships |
| packaging (`test_packaging`) | kept: every seed doc reaches the wheel, `_`-prefixed ones do not |
| contract (`test_contracts`) | kept: the hand-written API pins need no schema; the coverage fixture reads `packages:` instead of `checklist:` |
| satisfiability / availability | key off `packages:` and survive unchanged in spirit; whether availability keeps earning its CI minutes is a separate call |
| bench cases | unchanged; the switch is renamed |

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
  `list_skills`, `skill://`, `guide://` are removed; `_docs.py` replaces them;
- `services.skills_enabled / skills_local_dir` become
  `docs_enabled / docs_local_dir`, the old keys read as aliases for one
  release; `skills_index_plugins` is dropped with the plugin rows;
- `biopb._locations.mcp_skill_dir` gains a `mcp_docs_dir` sibling;
- `start_kernel` → `start_biopb` with the alias (#894).

Sequence: one PR for the store, the tools, the seed and the handshake, with
the old surface still registered; a second for the deletions and the test
rewrite; then the acceptance run.

## 9. Open risks

1. **Whether the seed still gets read.** The ablation proved the bodies help
   under tool-then-resource delivery. Under inline-index-then-`read_doc`
   delivery the read is a call the agent chooses to make. Only the acceptance
   run answers this. If it fails, the fallback is inlining the *Read first*
   bodies into the start return, at ~30 KB per session.
2. **Quality of unsupervised local docs.** The bench uses the seed, so it
   cannot see this. The cap bounds the size of the damage, not its rate; only
   dogfooding over weeks will show whether three bullets are enough discipline.
3. **Concurrent hand edits.** The user can edit the local directory, and the
   observe page may grow a way to. A `diff` write fails safe on changed
   context; a `body` write from stale state clobbers. Accepted for now.
4. **Two hundred entries is a context-budget guess.** Raising it later touches
   nothing but the constant, since edits are diffs.
