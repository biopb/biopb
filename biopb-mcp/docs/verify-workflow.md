# Verifying a workflow

**Component:** `biopb-mcp` — `mcp/_scratch.py` (the scratch kernel and the
slot), `mcp/_jobs.py` (`_Verification`, `_exec_cells`), `mcp/_server.py`
(`verify_workflow`), `mcp/_notebook.py` (`build_workflow_notebook`),
`mcp/_bootstrap.py`, `mcp/_observe.py` (`/api/notebook?workflow=1`).
**Related:** [`knowledge.md`](knowledge.md) — the *other* way a workflow is
kept; [`viewer-thread-safety.md`](viewer-thread-safety.md) and
[`_agent-fs-guardrail.md`](_agent-fs-guardrail.md) — the same posture of
naming a residual instead of hiding it.

## Two artifacts, two axes

Once a workflow is proven, two things are worth keeping and neither competes
with the other:

| | A **procedure doc** (`authoring`) | A **workflow notebook** |
|---|---|---|
| Holds | the procedure, minus the dataset | this run, on this dataset |
| Re-run by | an agent, deriving parameters on new data | the user, editing cells |
| Lives in | the doc index (`read_doc`) | a `.ipynb` on disk |

## The document is a rewrite, not a selection

`verify_workflow(cells, title)` takes a **document** — markdown with fenced
`python` cells, or a saved `.ipynb` — not a filtered job list. Checking
boxes in the observe job list and exporting those cells doesn't work: a
later cell can depend on an earlier one for a side effect
(`labels = label(mask)`) while overriding an earlier value that was wrong
(`spacing = ...`) — keeping the cell is right, keeping its value is not, so
the two merge into one. No selection UI expresses a merge; only rewriting
the transcript does, and that stays the agent's judgment. What's mechanized
is *checking* the rewrite: each fence runs in order, prose between them
rides into the notebook unmodified, and only the provenance cell and cell
outputs come from the run — an author that wrote both could claim "verified
correct" over a run that only proved the cells execute.

## A clean process, not a clean namespace

Verification runs the candidate in a scratch **kernel** — spawned per run,
discarded after — rather than the live session's. A clean *namespace* (a
dict seeded with the bootstrap's names, nothing else session-bound) was
tried first and enforces variable hygiene only: the viewer, `sys.modules`,
and anything mutated in place stayed shared, so a workflow leaning on a
layer the live session produced passed verification and failed on a fresh
kernel — exactly the bug the tool exists to catch. A second **process**
closes that: its own namespace, no viewer, ~1.5 s bring-up instead of
nothing. It is not `restart_kernel` — the live session is untouched.

**The scratch kernel is given nothing**: no napari viewer, no
`np`/`da`/`client`/`ops`, no user plugins pre-bound, no
`client = _conn.client` refresh prefix. What a document needs, it builds
itself via `biopb_mcp.workflow_env()` (the same code the session bootstrap
uses) — so anything handed to the run for free is something the document
could lean on and fail on later, one level up from variable hygiene.

**No viewer, as policy, not as a gap.** A workflow cell touching `viewer`
raises `NameError`, matching the saved notebook, which also has no viewer
and so runs under plain `nbconvert --execute`. A hidden
`napari.Viewer(show=False)` was considered: it renders with real GPU GL on
the session's own display, but `viewer.screenshot()` comes back solid black
there (`QOpenGLWidget.grabFramebuffer()` needs a shown widget) — a workflow
whose last cell saved a figure would verify green and produce a black
image. An offscreen Qt platform (`QT_QPA_PLATFORM=offscreen`) is worse —
`Viewer(show=False)` there constructs successfully with nothing actually
rendering, a silent trap rather than an alternative. Xvfb renders correctly
but roughly an order of magnitude slower on 3-D. Removing the viewer
sidesteps all three. The audit-export notebook (a session transcript, not a
verification) keeps a best-effort viewer, since it's full of viewer cells
that need somewhere to land.

**Plugins must load the same way on both ends.** The scratch kernel runs
the same bootstrap a session kernel does, so it loads the user's kernel
plugins too — the notebook's own bootstrap cell has to rebuild them
(`_load_namespace_plugins`, last, matching bootstrap step 7b) or a workflow
calling a plugin function verifies green and then dies `NameError` in the
saved notebook. The one asymmetry that's real and stated in the notebook's
intro: plugins come from *the reader's* `~/.config/biopb/kernel`, which
need not be the author's; a missing one binds nothing (fail-open) and the
cell using it raises where it's used. A plugin failing to load in the
scratch kernel is a real result, not noise — the saved notebook would fail
the same way for its next reader.

## What it costs

~1.5 s / ~154 MiB private for the headless scratch kernel (a bare ipykernel
is 0.75 s / 67 MiB) against a 60 s startup timeout, versus ~5-8 s / ~585-595
MiB for a full bootstrap with a viewer. Neither `qtpy` nor `napari` is
imported; dask import (~130 MiB) is the one cost the scratch kernel still
pays, only under a config that spins a cluster. Total memory is **worse**,
not better — the session's data isn't freed, it's duplicated. What changes
is blast radius: an OOM during verification kills a disposable kernel
instead of the user's session.

Two things that sound like costs and aren't. A cold client-side chunk cache
in the second process is not a real re-fetch: most chunks arrive as
weak-referenced mmap views over the server's own segment files, which a
second process shares through the OS page cache; only `chunk_locate`
dispatch (~290 µs) is paid again, and only a remote server with no local
mirror turns this into a real re-fetch. And plugins loading twice costs one
import, not double state — nothing re-executes an already-loaded plugin.

## Its death is the verdict

The scratch host runs with the watchdog off: `KernelHost`'s normal
death-and-respawn recovery is wrong here, since for a scratch kernel death
**is** the verdict (this workflow doesn't fit in memory) and respawning
would silently re-run it several times on a machine already under
pressure. The scratch host reports the death as the result rather than
recovering from it.

The run also **stops at the first failure** — later cells were written
against state the failed one was supposed to produce, so they're marked
`skipped` rather than reported as independent defects.

## Admission: one slot, owned by the session child

The dask cluster is shared and finite, and the kernel model is one cell at
a time, so verification takes the same admission slot ordinary work does
without becoming a second scheduler. The slot lives in the **session
child** (`_scratch.py`), not inside either kernel: `execute_code` checks
`_scratch.running()` before submitting a session job, and `_scratch.start()`
reads the session host's job records (`host.jobs.running()`) before claiming
the slot for a verification. Kernel-side admission (`_jobs.py`'s busy scan) is
unchanged and still handles ordinary job-vs-job contention on its own
kernel; the child's slot layers on top for the cross-kernel case.

**One accepted gap:** the child issues the scratch kernel's own job ids
(`verify-N`, which is what lets `poll_job` route without asking either
kernel), but the session kernel still issues its own `job-N`s, and a
verification checks the session through its host's records, which learn of a
job from iopub a moment after it starts — a check-then-act, so two jobs can
start within roughly a millisecond of each other. Closing it fully means the child issuing every job id, a
larger change than this.

A scratch kernel gets the default in-process scheduler like any other
fresh kernel; a workflow cell that wants a cluster attaches one explicitly,
and the global slot is what keeps two kernels from computing at once even
then — the cost is that a verification's dask workers start cold and can't
reuse the session's warm chunk cache.

`interrupt_kernel` resolves through the kernel's own "single running job"
check, so which kernel to signal is the session child's business. Its
*meaning* changes for a scratch kernel: on the session kernel interrupt is
deliberately best-effort (a hard stop would cost the user their whole
session), but a scratch kernel has nothing to lose, so its interrupt
escalates to a guaranteed kill after a short grace period — no separate
`shutdown_kernel` tool is needed. `restart_kernel` takes an in-flight
verification down with it, since the verification holds the slot and a
freshly restarted kernel could accept nothing while it ran. The job list
shows the running verification as a single row (the run, not its cells), so
a person can interrupt a verification they didn't start.

## What discard does not cover

"Discarded afterwards" describes the process, not its side effects. The
scratch kernel talks to the same tensor server and filesystem as the
session: `client.upload_array`, `client.add_source`, and any cell that
writes a file all go through for real. Verifying a workflow three times
leaves three uploaded arrays and three catalog entries behind — a
disposable kernel doesn't undo that. The `verify_workflow` docstring and
`guide://kernel` say so, so the agent can warn the user before running it.

## Two exports

`/api/notebook` is the **audit** export: every retained job in order, dead
ends included, with per-cell provenance headers. `?workflow=1` serves the
**workflow** export: the verified cells, no headers, an intro stating what
the run proved and what it did not.

A pass is written to `<state>/biopb/mcp/workflows/` before the verdict is
published — every pass, not just a chosen one, since this process only
knows the cells ran, not whether the numbers are right
(`suggested_workflow_filename` keeps a timestamp even with a title;
retention is keep-newest-50). The observe page's Verification pane offers
*Download* for the last run only; an older file stays on disk even after a
later attempt takes the offer away. A failed attempt is written too, to a
single overwritten draft per workflow (`.../workflows/drafts/<title>.md`); a
retry sends the whole document, and once a draft exists the agent re-reads
it before resubmitting, since a user may have edited it in the meantime.
