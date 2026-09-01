# Reproducing a workflow — verification in a scratch kernel

**Component:** `biopb-mcp` — `mcp/_scratch.py` (the scratch kernel and the
slot), `mcp/_jobs.py` (`_Verification`, `_exec_cells`), `mcp/_server.py`
(`verify_workflow`), `mcp/_notebook.py` (`build_workflow_notebook`),
`mcp/_observe.py` (`/api/notebook?workflow=1`).
**Related:** [`skills.md`](skills.md) — the *other* way a workflow is kept;
[`../../docs/verification-scratch-kernel.md`](../../docs/verification-scratch-kernel.md)
— why the isolation is a process, what it costs, and which display it takes.

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

## 3. A clean process, not a clean namespace

The rewrite is checked by running it. Running it somewhere the session's
leftovers are invisible is what makes the check mean something — a cell that
silently reads a variable it never created is exactly the defect that makes a
transcript unrunnable.

This started as a clean *namespace*: a dict seeded with the bootstrap's names
and nothing the session had bound. That enforced variable hygiene and nothing
else — the viewer, `sys.modules`, and anything mutated in place stayed shared —
so a workflow leaning on a layer the live session produced **passed verification
and failed on a fresh kernel**, which is the same class of bug in the dimension
most biopb workflows actually depend on.

It is now a clean *process*: a second kernel, spawned per verification and
discarded after it, with its own namespace and **no viewer at all**. It is not a
`restart_kernel` — the live session is untouched, which is what made the
namespace model attractive in the first place — it just costs ~1.5 s of bring-up
instead of nothing.

**Headless is the policy, not a limitation.** The viewer is how an agent shows
something to a person, and a verification has nobody watching; a workflow cell
touching `viewer` raises `NameError` here. That is the two ends agreeing (§5):
the saved workflow notebook has no viewer either, which makes it an ordinary
notebook that runs under `nbconvert --execute`. The audit export keeps one.
[`verification-scratch-kernel.md`](../../docs/verification-scratch-kernel.md)
prices it and records what it cost to learn — a hidden viewer works, but its
screenshots come back black.

**The residual, named rather than hidden** (the posture of
[`viewer-thread-safety.md`](viewer-thread-safety.md) and
[`agent-fs-guardrail.md`](agent-fs-guardrail.md)): a scratch *process* is not a
scratch *world*. It talks to the same tensor server and the same filesystem, so
`client.upload_array` / `add_source` and any cell that writes a file write
through for real — verify three times and you have three uploaded arrays. The
`verify_workflow` docstring says so, because the agent is the one who can warn
the user before running it.

## 4. Mechanics

`verify_workflow(cells, title)` hands the cells to `_scratch.start()`, which
takes the session's one job slot, spawns a kernel, and submits them there
through the ordinary `_jobs.submit()` door with `verify_cells=`. Inside that
kernel the cells run in its *own* namespace: it is fresh, so the filtered dict
the old model needed is exactly the process.

**One slot across both kernels.** Two kernels must not become two schedulers:
the dask cluster is shared and finite, and the agent's whole model is one cell
at a time. So while a verification runs, `execute_code` and the observe console
are refused with the same `busy` they already give — and the running
verification appears in the session's job list as a single row, so the refusal's
"wait for it, or interrupt it from its row" stays true and a person can stop a
verification they did not start.

The two directions are not enforced identically, and `_scratch.start` says so: a
session job is refused while a verification holds the slot exactly, but a
verification checks the session kernel by asking it, which leaves a
millisecond-wide window. Closing that means the session child issuing every job
id.

**`restart_kernel` takes an in-flight verification with it** — not because the
user asked to kill it, but because it holds the slot: leaving it running would
hand back a fresh kernel that can accept nothing, and a wedged verification
would have no escape hatch, which is what `restart_kernel` is documented to be.

**Its death is the verdict.** The scratch host runs with the watchdog off. For
the session kernel a respawn is recovery; here an OOM means "this workflow does
not fit", and respawning would re-run a workflow that just killed a process,
three more times. This is [#900](https://github.com/biopb/biopb/issues/900)
without the retry loop — and without the session dying, which is the containment
the whole design buys.

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
the final collection asks for `full=True`, once, when the run ends.

**Where the record lives.** In the session child (`_scratch._verified`), not in
a kernel — the kernel that produced it no longer exists. There is no
promote-on-success gate and no slot to clear on restart: the answer is "this run
passed, just now".

## 5. Plugins, and where the two ends have to agree

The scratch kernel loads the user's kernel plugins, because it runs the same
bootstrap a session kernel does. The notebook's bootstrap cell **did not rebuild
them** — a pre-existing gap that this feature made expensive: a workflow calling
`rolling_ball.subtract_background(...)` verified green and then died on
`NameError` in the saved notebook, with the intro claiming the cell rebuilt what
the run was given.

So `BOOTSTRAP_SRC` now calls the kernel's own `_load_namespace_plugins`, last,
as step 7b is last. **The remaining asymmetry is real and is stated in the
intro**: the plugins come from *the reader's* `~/.config/biopb/kernel`, which
need not be the author's. A missing one binds nothing (the loader is fail-open)
and the cell using it raises `NameError` where it is used. The bootstrap cell
prints what bound, so the reader can tell that case from a bug in the workflow.

The general rule this is an instance of: **whatever the scratch kernel is built
with, the bootstrap cell has to rebuild.** They are two halves of one claim —
"this ran against a fresh kernel, and here is that kernel" — and a name in one
but not the other turns a passing verification into a false promise. A plugin
that fails to load in the scratch kernel is therefore a *result*, not noise: the
notebook the user is about to save would fail the same way.

## 6. Two exports

`/api/notebook` is unchanged: the **audit** export, every retained job in order,
dead ends included, with per-cell provenance headers. `?workflow=1` serves the
**workflow** export: the verified cells, no headers, an intro that states what
the run proved and what it did not. The observe page shows *Save workflow*
beside *Save notebook* only once something is verified, learning that from the
`workflow` key the session child merges into the poll it already makes
(`_observe._api_jobs`) — the kernel cannot answer it, having never seen the
run.

## 7. Why the retained-job cap went up

`_MAX_RETAINED_JOBS` was 32 and is now 200. The rewrite in §2 is done by reading
the transcript, so eviction takes away the source material for the one step
nothing can automate — in a long session the workflow's opening cells were gone
before it was worth keeping. The raise is safe only because
`_MAX_JOB_OUTPUT_CHARS` now bounds a single record; until then one runaway cell
grew without limit and a record *count* bounded nothing.
