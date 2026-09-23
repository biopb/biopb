# Knowledge store — one flat set of docs, an agent-edited index

**Component:** `biopb-mcp` — `mcp/_docs.py` (store, index, tools),
`mcp/_docs_data/` (the shipped seed), `~/.config/biopb/docs/` (the user's own).

Two tools, `read_doc`/`write_doc`, reach one flat set of markdown files plus
an agent-edited index. One ablation on record (20 cases, one model) scored
17/3 with the catalog against 12/8 without; §6 is the acceptance test this
store re-runs.

## 1. The store

A doc is a markdown file. Two directories, read fresh on every access:

| tier | where | writable by the agent |
|---|---|---|
| shipped | `mcp/_docs_data/` inside the wheel | no — copy-on-write (§3) |
| local | `~/.config/biopb/docs/` (`services.docs_local_dir` moves it) | yes |

`services.skills_local_dir`, the pre-redesign key, is read as an alias for
one release; the new key wins where both are present.

**Id is the path under the directory, minus `.md`**, and may contain `/`.
Nothing about subdirectories is interpreted today; allowing the slash now
means the store can grow a tree by naming convention later with no
migration. `index` is reserved (§2).

**A `_`-prefixed name is banked: unlisted, not withheld.** It ships, and
`read_doc` returns it by id; what it does not get is a line in the seed
index, so it stays out of the reconciliation tail (§2) and no session is
told it exists. One hand-written entry promotes it, enforced in exactly one
place, `shipped_ids()`, which only the tail consumes. The `_` prefix is *the
release does not list this*; the index's `ignored:` line is *this agent does
not list this* — neither can express the other's decision, and the agent
owns `ignored:` outright. A banked doc gets no assertion from the package
gates (§6), since it's usually banked because its evidence is thin.

**Frontmatter is optional and small**, parsed tolerantly: scalars and inline
lists, unknown keys ignored, nothing fatal.

| key | who writes it | meaning |
|---|---|---|
| `description` | author, or inferred from the first paragraph | the hook `write_doc` files a new doc under (§3) |
| `packages` | shipped docs only | `name~=x.y.z, …` — third-party APIs the body quotes, read by the contract tests (§6) and nothing at runtime |
| `updated` | author, optional | overrides the file mtime the header otherwise reports |

Title is the first H1, else the id. Origin (shipped / local / local shadowing
shipped) is derived from where the file sits, never stored. **The store
classifies nothing else** — which docs are reference and which are
procedures is the index's sectioning, maintained by the agent, not a
runtime `kind:` field.

## 2. The index

The index is a doc — `index.md` in the local directory — that the agent
edits like any other. It is not derived from the files: a derived listing
cannot carry the agent's own hooks, grouping and "don't use X for Y" notes.

**Seeded on first run.** The package ships an `index.md` beside the docs;
when the local directory has none, it is copied there.

**Format is free markdown with two recognised line shapes.**

```markdown
# biopb docs

## References
- kernel: namespace, jobs, where computes run, the user shares this kernel
- data: pyramids, laziness, axis order; read a layer with viewer.tensor()

## Procedures
- flatfield: correct uneven illumination from the image itself or a blank
- stitch-tiles: assemble a grid of overlapping tiles into one canvas
- my-lab/nuclei-protocol: the lab's nuclear segmentation, validated on the 2026 set

ignored: measure-smlm-resolution, ratiometric-fret
```

- An **entry** is a list line whose first token is an id: `- <id>: <hook>`,
  the hook rendered verbatim.
- The **`ignored:`** line names *shipped* docs the agent has decided not to
  list, bounded by the size of a release. Ignore is explicit rather than
  absence, so a line lost in a sloppy rewrite reappears instead of silently
  retiring a shipped doc. An ignored doc is still readable by id.

**Local docs are indexed from birth and retired by one edit.** `write_doc`
appends an entry for a doc it creates (§3), so every local doc has a line to
remove. A local file with no entry — one the agent retired, or one the user
dropped in by hand — is not surfaced anywhere; it is readable by id and is
the user's to delete. Keeping local docs out of the tail and the ignore line
is what keeps both bounded.

Headings, prose and order are the agent's. The loader interprets nothing
else.

**Rendering.** `read_doc("index")` and the handshake (§3) return the file
plus what the loader knows and the file cannot:

1. an entry whose file does not exist gets the suffix `(missing)`; one that
   shadows a shipped doc gets `(local copy)`;
2. a tail, `New shipped docs: id, id, …`, listing shipped docs with neither
   an entry nor an `ignored:` mention — not truncated, since it's bounded by
   what one release adds.

The agent cleans up both by editing the file.

**Kernel plugins are not docs.** They are modules already bound in the
namespace: `server_status` lists which ones loaded and `inspect_object`
reads their docstrings, so the store does not mirror them.

**Bounds.** At most **200 entries**; `write_doc` refuses an index over that
with a message saying so. At the cap the rendered index is roughly 25 KB,
paid once per session start. The number of files is unbounded; only indexed
ones are discoverable, which is the bound that matters.

## 3. Tools

Two tools replace `list_skills`, `skill://{id}` and the five `guide://`
resources. Tools rather than resources because every host has tools.

**`read_doc(id)`** returns the body under a one-line header (origin,
updated, and `shadows shipped` when it does). `read_doc("index")` returns
the rendered index (§2).

**`write_doc(id, body=None, old=None, new=None)`** — `body` alone, or the
`old`/`new` pair.

- `body` creates or replaces the local file. On a *create*, the tool also
  appends `- <id>: <description>` to the index, under a trailing `##
  Unfiled` heading it adds if absent, so the doc is discoverable without a
  second call.
- `old`/`new` replaces one exact occurrence of `old` with `new`; refused if
  `old` is absent or matches more than once. This is the primitive every
  agent harness trains models on for file edits; a full index rewrite is
  both slow and lossy (models drop and paraphrase lines past ~100 of
  verbatim reproduction), where a replace costs a few dozen tokens whatever
  the index length.

There is no delete. Retiring a local doc is removing its index entry, and
retiring a shipped one is putting it on the `ignored:` line (§2).

**Shipped docs are copy-on-write.** A `body` or `old`/`new` write to a
shipped id creates a local file that shadows it (the replace is applied to
the shipped text as its base). The shipped file is never touched, so an
upgrade can still replace it, and the index marks the shadow (§2).

**The result is the diff** — `difflib.unified_diff` of before and after, so
a replace that landed somewhere unexpected shows immediately.

**Body cap.** A doc is read whole into context, so `write_doc` refuses a
body over 300 lines. The shipped procedures sit near 200.

**The handshake carries the index.** The MCP `instructions` field — sent
once per session at initialize, read by every host, in context before the
first tool call — carries the rendered index under one header line saying
this is doc `index`, editable with `write_doc` and re-readable with
`read_doc`. It is inlined rather than "now call `read_doc('index')`", since
a prompted hop loses agents that never take the first one. `start_kernel`'s
own return is unaffected: it also doubles as the recovery path after a
stuck or dead kernel, and a 25 KB index resent on every retry would be cost
with no information.

Mid-session edits are seen through `read_doc("index")`, and the `write_doc`
result shows the change. Where the host implements resource subscriptions,
the index is additionally exposed as a resource with a `resources/updated`
notification on every write — a nudge for hosts that honour it, nothing in
the design depends on it.

**The rest of the handshake** is: call the start tool first; the index
above lists what to read, and its reference docs are what to read before
non-trivial work; the standing guardrails; one sentence that destructive
steps always ask first.

## 4. Upgrades

The local index is the user's; the shipped set is the release's. The two
never need merging because the index references docs by id and the loader
reconciles on every render:

- a **new shipped doc** has no entry and no ignore, so it appears in the
  *New shipped docs* tail; the agent files it or ignores it, once;
- a **removed shipped doc** leaves its entry marked `(missing)`; the agent
  deletes the line;
- a **changed shipped doc** is simply read fresh; if the user had shadowed
  it, the shadow wins and the index says so.

No seen-list, no version stamp, no merge step.

## 5. Writing discipline

The authoring gate is the `write_doc` docstring, in five bullets: write only
a validated multi-step procedure, never a dataset-specific one; phrase the
index hook as the user's request; **update an existing doc rather than
write a near-duplicate** — read the index first, prefer an `old`/`new` edit
to a new file; and **verify a name, flag or call a doc quotes still exists
before relying on it**, since a local doc has no contract test behind it. A
short shipped reference doc, `authoring`, holds what a docstring cannot —
the checkpoint types (confirm-input, visual check, validate-and-gate) and
the derivation-rule convention for parameters — in well under a hundred
lines. Promotion to the shipped seed is a PR, and that is where review
happens.

**Links: one syntax, no policing.** A doc links another as `[[id]]`, and the
link resolves by `read_doc(id)` — nothing renders or rewrites it. A dangling
link in a local doc is allowed: it marks something worth writing. Shipped
docs keep the check that every link resolves to a shipped doc. When to link
is the agent's call — link what this doc depends on or hands off to, not
everything it mentions. Index hooks carry no links.

## 6. Ablation and tests

**The bench switch is an index.** `--bench-docs=false` writes the run's
isolated config tree a local index: the seed with every entry under its
`## Procedures` heading moved to the `ignored:` line. Nothing lists them,
the tail does not resurface them, and the handshake never shows an id the
agent could guess; the reference docs stay listed. That heading is the only
index heading named from outside the index, by the bench alone, and the
seed gate pins it.

`_tests/docs/` covers retrieval (`test_seed.py`: every shipped doc is either
a seed-index entry or on its `ignored:` line, every seed entry names a file
that ships, every `[[link]]` lands, nothing is over a cap), packaging (every
doc, banked ones included, reaches the wheel), and contract (hand-written
API pins, keyed off `packages:`).

**Acceptance.** Re-running the ablation pairing against the new store, both
arms, is the open validation step — §8.

## 7. Extensibility: what the design does not preclude

The next class of shipped content is the engine's own design docs across
packages — larger and more numerous than today's seed. To stay additive
without a later migration: ids may contain `/` and map to subdirectories in
both tiers; an id ending in `/` is reserved — refused by `write_doc`, and a
`- <id>/:` index line is kept verbatim and counted as one entry, so a
collection line in a seed index parses today and means something later;
reconciliation, the cap and the `ignored:` line are keyed by id in one
place, so grouping by first segment is a change to that place alone;
`read_doc` keeps a single positional `id`, leaving room for an optional
`section` later without a new tool or call-site change.

None of that — collection listings, section reads, full-text search over
the doc set — is built. The gap it would close: a listing answers "what
exists", not a troubleshooting question that starts from a literal (an
exception name, a config key, a log line), which is a grep-shaped search
over prose, not a ranking problem, at this corpus size.

## 8. Open risks

1. **Whether the seed still gets read**, now that reading it is a `read_doc`
   call the agent chooses to make rather than a tool/resource the harness
   pushes. Only the acceptance run (§6) answers this; the fallback if it
   fails is inlining reference bodies into the handshake as well.
2. **Quality of unsupervised local docs.** The bench uses the seed, so it
   can't see this. The cap bounds the size of the damage, not its rate.
3. **Concurrent hand edits.** The user can edit the local directory. An
   `old`/`new` write fails safe when the text has moved; a `body` write from
   stale state clobbers. Accepted for now.
4. **Two hundred entries is a context-budget guess.** Raising it later
   touches nothing but the constant, since edits are replaces, not
   rewrites.
