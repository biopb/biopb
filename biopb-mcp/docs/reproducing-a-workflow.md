# Reproducing a workflow — verification in a scratch namespace

**Component:** `biopb-mcp` — `mcp/_jobs.py` (`_Verification`, `_scratch_ns`,
`_exec_cells`), `mcp/_server.py` (`verify_workflow`), `mcp/_notebook.py`
(`build_workflow_notebook`), `mcp/_observe.py` (`/api/notebook?workflow=1`).
**Related:** [`skills.md`](skills.md) — the *other* way a workflow is kept.

## 1. Two artifacts, two axes

Once a workflow is proven, there are two things worth keeping, and they are not
competing answers to one question:

| | A **skill** (`write-a-skill`) | A **workflow notebook** |
|---|---|---|
| What it holds | the procedure, minus the dataset | this run, on this dataset |
| Who re-runs it | an agent, deriving parameters on new data | the user, editing cells |
| Where it lives | `list_skills` catalog | a `.ipynb` on their disk |

`write-a-skill`'s *When NOT to use* already sends dataset-specific work away
from the catalog; this is where it goes.

## 2. Why a filter over the transcript cannot produce the notebook

The obvious design — check some boxes in the observe job list, export those
cells — does not work, and the reason is worth stating because it is not
obvious until you hit it.

Suppose cell 1 is `spacing = (1, 1, 1); labels = label(mask)` and cell 2 is
`spacing = (0.2, 0.1, 0.1)  # forgot the real voxel size`. The notebook needs
cell 1 (it created `labels`) but not its value of `spacing`. **Keeping it is
wrong and dropping it is wrong**; the two have to merge into one cell.

The correct program is a **rewrite** of the transcript, not a subsequence of it.
No selection UI can express a merge, so the rewrite is the agent's judgment and
stays that way. What can be mechanized is *checking* the rewrite.

## 3. A clean namespace, not a clean process

The rewrite is checked by running it. Running it somewhere the session's
leftovers are invisible is what makes the check mean something — a cell that
silently reads a variable it never created is exactly the defect that makes a
transcript unrunnable.

That does **not** need `restart_kernel`. A restart destroys the viewer, the
layers, the dask cluster, and the namespace — i.e. it charges the user the whole
session at the moment their work succeeded, which they will (rightly) refuse.
What is actually needed is a clean *namespace*, and `_exec_capture(code, ns, job)`
already took `ns` as a parameter; only its one call site was pinned to
`_ip.user_ns`.

So `_scratch_ns()` builds a fresh dict holding the bootstrap's names at their
*current* values — `client` and `ops` are the live connected ones, so a
verification runs against the real server — and nothing bound since. The
baseline is recorded by `_jobs.mark_baseline()` at the end of `_bootstrap_impl`,
after the plugins load, so a plugin is in and a session variable is out.

**The residual, named rather than hidden** (the posture of
[`viewer-thread-safety.md`](viewer-thread-safety.md) and
[`agent-fs-guardrail.md`](agent-fs-guardrail.md)): the viewer, `sys.modules`,
and anything mutated in place are shared with the live session. A cell reading
`viewer.layers['nuclei']` still finds a layer this workflow never added, and
layers the run adds are added for real — which is why `added_layers` is
computed and reported instead of left for the user to discover. Variable hygiene
is enforced; layer and module hygiene is not.

## 4. Mechanics

`verify_workflow(cells, title)` submits through the ordinary `_jobs.submit()`
door with `verify_cells=` — same one-agent claim, same one-job-at-a-time rule,
same interrupt. A verification touches the one shared viewer like any other job,
so making it a second kind of thing would mean a second answer to each of those
questions.

The run **stops at the first failure**. Cells after it were written against
state the failed one was supposed to produce, so running them anyway reports a
cascade of consequences as separate defects; they are marked `skipped` so the
report says how far the workflow got.

Output is captured per cell and teed to the job: the notebook needs the split,
`poll_job` on a long verification needs the whole run accumulating where it
always does. Both use the same capped buffer (`_OutputBuffer`).

**The record has a polled shape and a full one**, the way a job already has
`jobs_summary()` and `poll()`. A job snapshot crosses a JSON round trip out of
the kernel every 0.4 s while a verification runs, and carrying every cell's
output there ships the bytes `stdout` already holds, once more per cell — a
20-cell run polled 1.2 MB where an ordinary job polls 200 KB, growing with the
workflow. So `_Cell.snapshot()` carries a one-line head and a length, and only
`verified()` asks for `full=True`, once, when the notebook is built. The ledger
a report prints never needed more than the head.

A verification is kept as *the* workflow (`_jobs.verified()`) only when every
cell ran — a partial run is a report, which the job record already is. A later
failure does not un-verify what passed; a kernel restart does.

## 5. Plugins, and where the two ends have to agree

The scratch namespace holds the user's kernel plugins, because `mark_baseline()`
runs after step 7b and a fresh biopb kernel has them. The notebook's bootstrap
cell **did not rebuild them** — a pre-existing gap that this feature made
expensive: a workflow calling `rolling_ball.subtract_background(...)` verified
green and then died on `NameError` in the saved notebook, with the intro
claiming the cell rebuilt what the run was given.

So `BOOTSTRAP_SRC` now calls the kernel's own `_load_namespace_plugins`, last,
as step 7b is last. **The remaining asymmetry is real and is stated in the
intro**: the plugins come from *the reader's* `~/.config/biopb/kernel`, which
need not be the author's. A missing one binds nothing (the loader is fail-open)
and the cell using it raises `NameError` where it is used. The bootstrap cell
prints what bound, so the reader can tell that case from a bug in the workflow.

The general rule this is an instance of: **whatever the scratch namespace is
seeded with, the bootstrap cell has to rebuild.** They are two halves of one
claim — "this ran against a fresh kernel's namespace, and here is that
namespace" — and a name in one but not the other turns a passing verification
into a false promise.

## 6. Two exports

`/api/notebook` is unchanged: the **audit** export, every retained job in order,
dead ends included, with per-cell provenance headers. `?workflow=1` serves the
**workflow** export: the verified cells, no headers, an intro that states what
the run proved and what it did not. The observe page shows *Save workflow*
beside *Save notebook* only once something is verified, learning that from the
`workflow` key `jobs_view()` added to the poll it already makes.

## 7. Why the retained-job cap went up

`_MAX_RETAINED_JOBS` was 32 and is now 200. The rewrite in §2 is done by reading
the transcript, so eviction takes away the source material for the one step
nothing can automate — in a long session the workflow's opening cells were gone
before it was worth keeping. The raise is safe only because
`_MAX_JOB_OUTPUT_CHARS` now bounds a single record; until then one runaway cell
grew without limit and a record *count* bounded nothing.
