# Verifying a workflow in a scratch kernel

Status: **implemented** — `mcp/_scratch.py`, with the kernel side in
`mcp/_jobs.py` and `mcp/_bootstrap.py`. `verify_workflow` used to run a
candidate workflow in a scratch *namespace* inside the live kernel; it now runs
it in a scratch *process* — one that is **headless by policy**: no viewer, no
Qt. So is the saved workflow notebook, which is what makes the two agree. This
note is why, what it costs, and the admission rule that keeps two kernels from
becoming two schedulers. Where the implementation stops short of the design, it
says so — see [What was built](#what-was-built).

## What verification is for, and where the namespace model fails it

`verify_workflow` answers one question: **will this program run on its own?**
That is the defect that makes a session transcript unusable as a document — a
cell that reads a variable an earlier, discarded cell created.

A scratch namespace answers it for *bindings* and nothing else. `_scratch_ns`,
the filtered dict this replaced, said so itself:

> The viewer, `sys.modules`, and anything a cell mutates in place are shared
> with the live session. So a cell reading `viewer.layers['nuclei']` still finds
> a layer this workflow never added, an import with side effects has already
> happened, and layers the run adds are added for real. Variable hygiene is
> enforced; layer and module hygiene is not.

So a workflow that leans on a layer the live session produced **passes
verification and fails on a fresh kernel** — precisely the class of bug the tool
exists to catch, in the one dimension most biopb workflows actually depend on.
Verification is not merely incomplete here; on the viewer it is misleading, and
a green result is the reason the user stops checking.

## The bookkeeping the shared kernel forces

Because the run happens in the session's own kernel, through the session's own
job runner, three mechanisms exist to tell it apart afterwards:

- **A per-job record.** `job.verify` holds a `_Verification` on every
  verification job, pass or fail.
- **A session slot.** `_jobs._verified`, module-level, holding only the last
  verification whose every cell ran. The gate is one line in the job finalizer
  (`_jobs.py:746`), and it carries its own rules: a later failure does not
  un-verify, and `reset()` clears it on kernel restart.
- **A filter on the job list.** Verification cells arrive in the session's job
  list looking exactly like ordinary code cells that repeat earlier ones — twice
  over after a failed first attempt. Marking them takes a new field on
  `jobs_summary()` and a view switch on the observe page.

Plus a residual nothing cleans up: layers a verification adds land in the *live*
viewer and stay there. `_Verification.added_layers` keeps their names so the
agent can mention them; nothing removes them.

Four mechanisms, all of them answering "which of these rows was the rehearsal?"
— a question that only exists because the rehearsal shares the stage.

## The model

One scratch kernel, spawned on demand, holding its own namespace, attached to
the session's already-warm dask cluster, and **headless**. A verification is
then:

    run the cells → verdict → if it passed, write the notebook → discard the kernel

There is no slot to remember, because the answer is "this run passed, just now"
rather than "some run passed at some point". There is no residual, because
whatever the run touched goes with the process. There is nothing to filter out
of the session's job list, because the run was never in it.

### Every pass is written

The notebook goes to `<state>/biopb/mcp/workflows/` as soon as the run passes,
before the verdict is published — everything waits on the status, so writing
after it would hand a reader a passed run whose file does not exist yet.

Every pass, because this process knows only that all the cells ran; whether the
*numbers* are right is not a question it can ask. Choosing between passes is
therefore deferred to whoever reads them, and the naming rule was already built
for it: `suggested_workflow_filename` keeps the timestamp even when there is a
title, so repeated verifications of one workflow do not collide. Retention is
keep-newest-50, the shape the per-session logs beside it already use.

The observe page shows the last run and offers *Download* for it. A later
attempt that fails takes that offer away — the page shows one run, and offering
a document it is not showing is the confusing half — but the earlier file is
still on disk. What a failure removes is the offer, not the document.

### No viewer, as policy

The scratch kernel builds **no napari viewer and does not import Qt**. That is a
policy, not a limitation, and it is the one decision the rest of this note
turned on twice before landing on it.

The viewer exists so an agent can show something to the person it is working
with. A verification has no person in it. So a workflow cell that reaches for
`viewer` is a cell that will not run as the document it is about to become — the
saved workflow notebook has no viewer either
(`_notebook.WORKFLOW_BOOTSTRAP_SRC`). Failing in the scratch kernel is the two
ends agreeing, which is the rule the whole feature rests on: *whatever the
scratch kernel is built with, the bootstrap cell has to rebuild.*

It also drops the notebook's dependency on a display, which makes a saved
workflow an ordinary notebook — one that runs under `nbconvert --execute`, in
CI, on a headless box. The audit export keeps its best-effort viewer, because a
session transcript is full of viewer cells that need somewhere to land.

**What it cost to learn.** The design first kept a viewer and hid it with
`napari.Viewer(show=False)`, taking the session's own display so the GL was real
(see [The display](#the-display)). Measured, that works: no window maps, and the
context renders on the GPU. But `viewer.screenshot()` returns **solid black**
there — napari screenshots via `QOpenGLWidget.grabFramebuffer()`, and the widget
is never `isValid()` until it is shown:

```
widget isValid(): False
viewer.screenshot(canvas_only=True)  -> max=0, 0.0% non-black
vispy _scene_canvas.render()         -> max=253, 57.6% non-black
```

So a workflow whose last cell saved a figure would have verified green and
produced a black image — a false pass, the exact thing this feature exists to
prevent. It was fixable (route screenshots through vispy's offscreen render;
verified byte-identical to a shown viewer's, in 2-D and 3-D). But fixing it is
complexity in service of a capability nobody in a verification needs, and asking
"why would a verification screenshot anything?" makes the viewer itself the
thing to remove.

## What it costs

Measured on the dev box (Ryzen 5 5600X, `.venv`, page cache warm):

| | bring-up | RSS | Pss | Private |
|---|---|---|---|---|
| bare ipykernel | 0.75 s | 67 MiB | — | — |
| **headless scratch kernel** | **1.50 s** | **234 MiB** | **187 MiB** | **154 MiB** |
| full bootstrap, `QT_QPA_PLATFORM=offscreen` | 4.88 s | 595 MiB | — | — |
| full bootstrap, real display `:1` | 7.63 s | 584 MiB | — | ~358 MiB |

**1.5 s and ~154 MiB private**, against a 60 s `startup_timeout`. Neither `qtpy`
nor `napari` is imported at all. Dask adds nothing: `DaskClusterHost` is already
decoupled from the kernel and `_launch` injects the scheduler address
(`BIOPB_DASK_ADDRESS`), so a second kernel attaches to the warm cluster rather
than spinning one.

Where the memory goes, cumulative through a *full* bootstrap — the two rows the
headless kernel does not pay are marked:

| step | delta | cumulative |
|---|---|---|
| bare interpreter | — | 11.8 M |
| + numpy | 14.5 M | 26.4 M |
| **+ dask** | **118.0 M** | 144.4 M |
| + distributed | 12.4 M | 156.8 M |
| ~~+ qtpy/PyQt6 (import)~~ | ~~24.0 M~~ | 180.8 M |
| ~~+ QApplication()~~ | ~~3.4 M~~ | 184.1 M |
| ~~+ napari (import)~~ | ~~0.1 M~~ | 184.2 M |
| ~~**+ napari.Viewer()**~~ | ~~**254.8 M**~~ | 439.0 M |
| + biopb.tensor client | 1.5 M | 445.6 M |

Qt was never the cost. PyQt6 plus `QApplication()` is 27 MiB. The two real items
are `napari.Viewer()` at 255 MiB — napari defers nearly everything to
construction, which is why `import napari` itself is 0.1 M — and dask at 130 MiB
of pure import before a single task runs. Dropping the viewer removes the larger
one outright; dask remains, and is the floor.

### Memory: containment, not savings

Worth stating plainly, because the opposite is the tempting claim: a scratch
process still makes total memory **worse** — by ~154 MiB now rather than ~340.
The session holds its data while the scratch kernel holds its own; nothing is
freed by moving.

What changes is the blast radius. Today a verification allocates the workflow's
data a second time *inside the live kernel*, so an OOM kills the session and the
user's work with it — which is exactly what happened in
[#900](https://github.com/biopb/biopb/issues/900). With a scratch process the
kernel that dies is the disposable one, and the failure reads "verification ran
out of memory" instead of "your session is gone".

## The display

**The scratch kernel needs no display.** No Qt, no GL, no `DISPLAY`, no Xvfb.
This section is kept because it is the evidence behind that, and because the
question comes back every time someone proposes putting a viewer in.

Had the scratch kernel kept a viewer, it would have needed a GL context, and the
three obvious ways to avoid borrowing the user's window are not equivalent.
Measured with a bare `QOffscreenSurface` + `QOpenGLContext`, no napari involved:

| | platform | GL renderer |
|---|---|---|
| `QT_QPA_PLATFORM=offscreen` | offscreen | **no context at all** (`create()` returns False) |
| `xvfb-run` | xcb | Mesa llvmpipe (software) |
| `DISPLAY=:0` on this box | xcb | Mesa llvmpipe (software) |
| `DISPLAY=:1` — the session's own | xcb | **NVIDIA Quadro P2000** (hardware) |

**Offscreen is not an option.** It creates no GL context, so nothing renders.
`napari.Viewer(show=False)` still *constructs* there and `add_image` still
returns — napari defers canvas creation — which makes it a trap: a smoke test
passes and rendering is silently absent.

**A virtual display is a real slowdown, not a formality.** Xvfb falls back to
llvmpipe, software rasterisation, measured at roughly an order of magnitude
slower than the GPU on 3-D volumes. Verification would be slowest on exactly the
workflows that are heaviest.

**Taking the session's own display works, and still is not enough.** Measured on
`:1` with `napari.Viewer(show=False)`: hardware GL, and **no window maps** —
confirmed against a live desktop, sampling every 3 s for 24 s with the session's
own window as a positive control. But screenshots come back black (above), so
the viewer that survives this route is one that cannot show anything anyway.

Which is the argument, arrived at the long way: if the only viewer a scratch
kernel can have is one nobody can see and nothing can capture, it is not a
viewer. Leave it out and the display question disappears with it — along with
the reason the end-to-end test had to be gated on a desktop. It runs in CI now.

### Why not share the session's viewer

Raised early, and rejected before the headless decision made it moot. Recorded
because it is the other tempting shortcut.

**It gives up the reason for the design.** A fresh viewer is what closes the hole
this note opens with — a workflow leaning on a layer the live session made.
Share it, and the scratch process buys `sys.modules` isolation plus OOM
containment over what the scratch *namespace* already gave.

**The proxy does not cross a process boundary.** `_viewer_proxy.py` is a
same-process **thread**-marshaling proxy (`run_on_main` → the Qt main thread) for
biopb/biopb#100. Its central move is re-wrapping every napari handle it returns
so handles never leak unwrapped — precisely what cannot be done across
processes, where you would need a server-side handle table and a real protocol.
That surface has already cost a segfault class (#100) and an overlay gap (#840);
it was the wrong place to add an RPC layer.

## What discard does not cover

"Discarded afterwards" describes the *process*, and it is worth being explicit
that this is narrower than "no side effects", because the tidier model invites
the stronger reading.

The scratch kernel talks to the same tensor server and the same filesystem as
the session. `client.upload_array` / `upload_zarr` / `upload_chunk` and
`client.add_source` write through, and so does any cell that writes a file.
Verify a workflow three times and you have three uploaded arrays and three
catalog entries. Nothing about a disposable kernel changes that.

The current model at least states its residual out loud ("layers the run adds
are added for real"). The replacement must do the same, and the place for it is
**agent-facing prose** — the `verify_workflow` docstring and `guide://kernel` —
so the agent knows and says so before running. That docstring is already where
the misleading "**This costs no restart.** … there is nothing to ask the user
about before calling this" line lives (#900).

## The scratch kernel must not be respawned

`KernelHost._handle_unexpected_death` reaps and respawns up to three times in a
60 s window. For the session kernel that is recovery. For a scratch kernel it is
a bug: its death **is** the verdict. An OOM means "this workflow does not fit",
and respawning would silently re-run a workflow that just killed a process,
three more times, each allocating gigabytes on a machine already under pressure
— #900 with a retry loop attached. The scratch host needs the watchdog off, or
an explicit no-respawn mode, and its death reported as the result.

## Admission: one slot, owned by the session child

Two kernels must not become two schedulers. The dask cluster is shared and
finite (12 workers here), `interrupt_current` and `_cancel_dask_futures` assume
one running job, and the agent's whole model of the kernel is "one cell at a
time" (`submit()` returns `{"error": "busy", "running_job_id": ...}`).

So verification takes the same slot ordinary work does. Two details make that
safe rather than a new deadlock.

**Distinguish the two locks that exist today.** `_jobs._lock` (kernel side) is
held only for the admission decision — `submit()` scans for a running job,
starts a thread, and releases on return; `_run` never touches it. That is a
short lock and extending its shape is cheap. `KernelHost._lock` (session child)
is the one held across a kernel RPC, and starving it behind a dead kernel is
[#902](https://github.com/biopb/biopb/pull/902). Do not build the cross-kernel
gate out of the second shape.

**The gate moves up, and stops being a mutex.** Kernel-side admission can only
see its own kernel, so the decision belongs in the session child, which already
owns both `KernelHost`s. It should not be a lock the child *holds* for the
duration of a job: a scratch kernel OOM-killed mid-verification — the exact
scenario this design is for — would die holding it.

Instead the session child owns a **single slot and issues the job ids**. It
never asks a kernel whether it is busy, because it granted the slot. The slot is
freed on completion or on the kernel-death signal the watchdog already produces.
Submissions still fail fast with `busy` rather than blocking. Routing `poll_job`
falls out for free: the child issued the id, so it knows which kernel owns it.

Most of the kernel-side claim machinery (`_owner`, `_owner_label`, the busy
scan) can then shrink, since admission is decided above it.

**The cluster is shared, and the slot is why.** Two clients on one
`LocalCluster` sounds like contention, but under a global slot there is no
second computation to contend with: while a verification runs, the session
kernel is by construction not running a job. The other thing that could plausibly
be computing there — the viewer's own slice reads — is deliberately kept off the
cluster, `ViewerConfig.compute_scheduler` defaulting to `"threads"` so plane
reads use "the one shared client cache (~100% hit on revisit, no worker scatter;
biopb/biopb#8)". Nothing else schedules work on it: outside a job, `_dask_client`
appears only in `server_status`'s `scheduler_info()` and the graceful-close
snippet.

Sharing then pays twice over, because the workers' per-worker chunk caches (the
budget `_make_cache_plugin` divides) are already warm from the session's own
reads — so the verification inherits them, which is most of what the scratch
process's empty client-side cache would otherwise have cost.

What is left is a cache effect in the other direction: a verification can evict
the session's cached chunks from those worker caches, so the session's next run
re-reads them. It is a cache, so nothing is wrong afterwards, only slower —
secondary, and not worth a second cluster or a worker share to avoid.

**Neither interrupt nor restart grows an argument.** `interrupt_kernel()`
resolves through `_running_job()` — "the single running job, or None. One job at
a time" — so while the slot is global that stays well defined and *which* kernel
to signal is the session child's business, not the tool signature's. The rules
around it carry over untouched: a verification the agent started is its own job
to stop, and `requester="user"` still lets a person stop anything running in
their session — which is what makes a verification they did not start
interruptible at all. Ownership is not re-implemented for it: `_scratch.start`
submits with the verifying client's writer, so the scratch kernel's own
`_jobs.submit` claims it and its own `interrupt_current` refuses a stranger, by
the same check that guards any other job.

**But interrupt does change meaning, and it has to.** On the session kernel it
is best-effort by necessity — a KeyboardInterrupt does not reach a blocking C
call, and the guaranteed stop is a group-kill that costs the user their whole
session, so it is theirs to ask for. In a scratch kernel there is nothing to
lose: no variables anyone wants, no layers, no session, and the process was
going to be discarded seconds later anyway. So a verification's interrupt is
best-effort only briefly — long enough for a clean stop to yield the better
record — and then takes the process.

Without that escalation the tool surface has a hole shaped exactly wrong: a
verification wedged in a C call would leave the agent nothing but
`restart_kernel`, i.e. destroying the user's session to end a process built to
be thrown away. There is no `shutdown_kernel` tool and this is why one is not
needed — the guaranteed stop for the scratch kernel is reached through the tool
that already means "stop the running job", rather than through new surface the
agent has to learn when to prefer.

`restart_kernel` needs no argument for a different reason: a scratch kernel is
never restarted, only created and discarded, so "the kernel" is unambiguously
the session one. What restart *must* do is take an in-flight verification with
it. Not because the user asked to kill it, but because the verification holds
the slot: leave it running and the freshly restarted kernel can accept nothing,
and a wedged verification has no escape hatch — which is precisely the job
`restart_kernel` is documented to do ("the guaranteed stop"). Discarding is
cheap; a verification is 5–8 s to start again.

**The human is already inside this rule.** The `origin="user"` exemption in
`submit()` is scoped to the *ownership claim* — it is what lets a person run a
cell without taking the kernel away from the agent that holds it. The
one-at-a-time check runs unconditionally, after it, so a console cell gets
`busy` like anything else and the observe page already words the refusal ("you
already have a cell running (job-N). Wait for it, or interrupt it from its row
above."). Extending the slot across two kernels therefore needs no decision
about the exemption: a verification blocks a console cell exactly as an
`execute_code` job does today.

What it does need is for the refusal to stay *true*. Today the blocking job is
always a row in the list the message points at. A verification running in the
scratch kernel is not, so either the session's job list carries the running
verification as a single row — the run, not its cells — or the message stops
telling the user to look for a row that is not there. The former is better: it
is also the only way a person can interrupt a verification they did not start.

## What was built

The model above, with two deliberate stops short of it.

**The scratch kernel is headless.** Not in the note's first draft, which kept a
hidden viewer; see [No viewer, as policy](#no-viewer-as-policy). The health probe
changed with it — the session kernel's asks for `viewer`, the scratch one's asks
for `_jobs` and `ops`.

**The kernel-side claim machinery stayed.** The note says `_owner`, the busy
scan and the rest "can then shrink, since admission is decided above it". They
did not: the session kernel still runs its own admission for its own jobs, and
the slot in the child is layered on top. That keeps the change additive — the
session kernel behaves exactly as it did — at the cost of two places that both
know a job is running.

**The child does not issue every job id.** It issues the verification's
(`verify-N`, which is what routes `poll_job` without asking either kernel), but
the session kernel still issues `job-N`. The consequence is stated in
`_scratch.start`: a *session* job is refused while a verification holds the slot
exactly, but a *verification* is refused while a session job runs by asking the
session kernel, which leaves a millisecond-wide window in which both can start.
Bounded — two jobs briefly sharing a warm cluster — and closing it means the
child issuing every id, which is the larger change this one deliberately is not.

## Open questions

**Previously "verified" workflows will start failing.** Anything that passed by
leaning on live-session state now fails. That is the feature working, and it
will be reported as a regression. `verify_workflow`'s docstring is written so
the agent can say which it is.

**Workflows that use the viewer will start failing too**, on top of the above —
`viewer.add_image(result)` as a closing "show the user" step is common. That is
the policy working, and the `verify_workflow` docstring tells the agent to
compute and `print` instead, but it is the change most likely to be reported as
a bug.

~~**Lazy `Viewer()`.**~~ Resolved by removal, and more thoroughly than laziness
would have: deferring construction was estimated to save ~110 MiB while still
importing napari and Qt. Not building one saves 255 MiB and imports neither.

## Costs that turn out not to be costs

Two concerns raised against this design do not survive contact with the code,
recorded here so they are not re-raised.

**A cold client-side chunk cache.** The SDK cache is not primarily an LRU. The
strong `cachey` cache holds "only chunks that cost real client RAM: the copies";
the primary path is a `WeakValueDictionary` over mmap'd views that "costs no RAM
and needs no budget" because the server already caches (`biopb/tensor/_pool.py`).
A second process therefore starts with an empty weak cache but mmaps the *same*
server segment files, and the OS page cache is cross-process — so the bytes are
already hot. What is paid again is per-chunk dispatch (~290 µs `chunk_locate`
RTT), not transfer. Only a remote server with no local mirror makes this a real
re-fetch.

**Plugins loading twice.** Kernel plugin loading is namespace binding: names are
merged with reserved-name shadow checks (biopb/biopb#92, #664) behind an import
hook whose contract is "hands back a plugin module that is already loaded,
nothing re-executes". The cost is the plugin's own import, once per process.
And a plugin that fails in the scratch kernel is a *result*, not noise: the
notebook the user is about to save would fail the same way. The only residue is
a plugin that fails and the workflow never uses, which is too narrow to design
around.

## Migration

`_notebook` grew a second bootstrap cell: `WORKFLOW_BOOTSTRAP_SRC` (no viewer,
for the workflow export) beside `BOOTSTRAP_SRC` (viewer, for the audit export).

Deleted with the namespace model: `_jobs._verified` and its
promote-on-success gate, `_jobs.mark_baseline` / `_scratch_ns` / the baseline
name set, `added_layers` and the `_layer_names` bookkeeping behind it, and
`_jobs.verified` / `verified_summary` (the record lives in the child now, as
`_scratch._verified`). `jobs_view()` no longer carries `workflow`; the child
merges it in, because the kernel never saw the run.

Added: `_jobs.running_job()` (the child's cross-kernel busy check) and
`_jobs.verify_record()` (the full record, read once before the kernel holding it
is discarded — the other half of the polled/full split).

[#903](https://github.com/biopb/biopb/pull/903) adds a `verify` marker and a
view filter to the observe job list. It was a patch on the model this note
removes: verification cells no longer appear in the session's job list at all,
because they never ran in the session kernel. What appears instead is one row
for the whole run, so a person can see what the kernel is busy with and
interrupt a verification they did not start. #903 should be closed.
