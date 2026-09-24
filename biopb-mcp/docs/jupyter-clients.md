# Jupyter clients on the session kernel

Status: **implemented**. Decided 2026-09-23.

**Component:** `biopb-mcp` — a kernel subclass in an in-kernel module
(`mcp/_kernel_gate.py`), `mcp/_jobs.py` (the record), `mcp/_kernel.py` and
`mcp/_server.py` (the connection-file hint). Replaces the observe page's user
console.

## Goal

Let a human work in the session kernel from a real Jupyter client — the same
namespace, the same `viewer`, the same `client` — with the two guarantees the
user console gives today: **one writer at a time**, and **the agent is told of
every cell it did not run**. The console's own UI is retired: the gap between
it and a notebook is not worth closing when the notebook can be the UI.

The kernel is an ordinary ipykernel started by `jupyter_client`, so any client
that attaches by connection file works already (`jupyter qtconsole --existing
kernel-<id>.json`). What is missing is the two guarantees, and a way to find
the file.

## What holds

- **ipykernel serializes.** Every execute request runs one at a time on the
  main thread, in arrival order, whatever client sent it -- the agent's cells
  included: `execute_code` sends its code as a plain execute request
  (`KernelHost.run_cell`), the way a notebook does.
- **A `run_async` task is the one thing outside that order**: the agent opts
  into it for a long compute, and it runs on a worker thread beside the cells,
  leaving the main thread -- and so Qt and the screenshots -- free.
- **A client ignores the host's traffic.** qtconsole and Lab drop iopub
  messages whose parent session is not their own (qtconsole's
  `include_other_output` defaults off); only the busy/idle indicator reacts.
- **A task's prints go to iopub under the cell that started it.** ipykernel
  attributes a worker thread's output to the request that started the thread.

The host adds no ordering of its own. Its one threaded client routes every
shell reply and iopub message to the call it answers (`mcp/_kernel_io.py`), so
its round trips overlap and the kernel orders them with everyone else's.

## The rule: queue, as in any notebook

A user's cell sent while the agent's cell runs waits its turn, as it would
behind another cell of their own; one sent while a task runs runs beside it.
The agent is held to one job at a time instead: `execute_code` is refused
while anything runs (the host's records say what), since an agent that queued
would lose track of what ran when. Until the agent's cells moved to the main
thread, a user's cell was refused while an agent job ran; see v3.

## Design

### The kernel class

A kernel subclass (`mcp/_kernel_gate.py`), which `KernelHost` launches every
kernel with (`--IPKernelApp.kernel_class=...`), overrides `do_execute` to hold
each non-empty cell as the running job while it runs (`_jobs.hold_cell`), so
Stop can reach it (see Control channel). It also serves the host's control
requests.

### Foreign is "not the host's session"

The host's own client has one session id for the kernel's life, and it exists
before the kernel does: `KernelManager.client()` inherits the manager's session
id. `KernelHost` hands it over at launch (`BIOPB_HOST_SESSION`), so the kernel
knows its host before the connection file lets anyone connect, and nothing a
cell sends can re-adopt it. A kernel restart is a new manager, so a new id. A
request from any other session is foreign: the host records it as the user's.

The session id is not a secret: it rides in the parent header of every iopub
message the host's requests produce. It tells honest clients apart; anyone
holding the connection file can run anything anyway.

**`silent` does not hide a cell.** `silent` only stops output being broadcast;
silent code runs with full effect, so the kernel class echoes a non-empty
silent foreign cell itself, which ipykernel does not, and the host records it
like any other. What clients actually send silently is empty: qtconsole's
prompt-number request, and `user_expressions` evaluated for its UI. Neither is
recorded.

### Record

The records live in the host (`mcp/_job_log.py`), built from iopub, and the
host names every job (`job-N`, one counter for the host's life, so ids never
repeat across restarts). Every `stream`, `execute_result` and `error` under a
job's request is the job's. Two kinds of job:

- **A cell** is read from the protocol: an iopub `error` fails it
  (`KeyboardInterrupt`: interrupted), and the `status: idle` for its request
  ends it. A foreign cell (`origin="user"`) starts at its `execute_input`. The
  agent's cell is recorded as the host sends it (`JobLog.start_cell`), begins
  at its own `execute_input` -- it may wait behind a user's cell -- and its
  shell reply, which cannot be lost, ends it if its idle never arrives, and
  fails it if its iopub `error` was lost. A verification's cells are these
  too, one request each to the scratch kernel. The host's own snippets and
  empty code are not cells.
- **A task** (`run_async`) runs on a worker thread and outlives the cell that
  started it, so the kernel announces its start and end as `biopb_job` messages
  (`_jobs._publish`). Its thread's prints arrive under that cell's request, and
  from the announced start on they are the task's; its writer is the cell's.

The announcement has no parent header: a client drops iopub from other
sessions, and an unknown message type under its own request would be one more
thing for it to ignore. Streams are flushed before the end is announced, so a
job's output is complete when its record says it ended. iopub is a PUB socket
and can drop under pressure. Cells run one at a time and tasks one at a time,
so a cell's end ends any other cell that had begun, and a task's start any
other task, as "end not recorded"; a cell never ends a task.

Poll, the observe list and detail, the notebook export and the foreign-activity
digest are reads of the host's memory, never a kernel round trip. Records
outlive a kernel restart: one still running when its kernel goes is ended as
interrupted. Display output (`display_data`) is not recorded: the record
says what ran and whether it failed, not what it drew.

### Control channel

What must not wait behind a cell goes to the kernel on the control channel,
which ipykernel serves on its own thread: stopping a job, and the graceful
close before a kill. One custom request, `biopb_request`
(`GatedKernel.biopb_request`), carries an op name and its arguments; the op
runs on a worker thread, bounded by a timeout, since control requests are
handled one at a time and ipykernel's own interrupt and shutdown share that
queue. Refused for any session but the host's, like the gate.

**Stop names its job** (`KernelHost.interrupt_job`): the host takes the id its
records say is running, or the observe row's, and sends the kernel that job's
request (`_jobs.interrupt(request)`), since a cell's id is the host's and
the kernel knows a job by its request. The kernel checks it is still the
running job before touching anything. A stale id stops nothing and the reply
names what runs now. A task gets a `KeyboardInterrupt` raised into its thread;
a cell on the main thread gets a real `SIGINT`
to the kernel process alone (`km.interrupt_kernel` signals the whole process
group, dask workers included), which also wakes a blocking sleep. The check and
the signal happen under the lock `hold_cell` ends the cell under, so the
signal lands in that cell or, at the latest, in the lock wait that ends it, and
never in the next one. One window is ipykernel's: a signal arriving after the
cell's code returned but before the gate settles it surfaces in ipykernel's
wrap-up, as any Jupyter interrupt can; the gate still sends the client a reply.

**The busy/idle around a control request is ignored.** ipykernel publishes a
status for every request, control ones included, and an idle after a control
request would read as idle while a cell still holds the main thread.

What stays on the shell channel needs the main thread or its request: the
agent's cells, screenshot, inspect.

**Who may stop what is the host's decision**: it holds the records (whose job
it is) and the one-agent claim, and the kernel stops what it is told to. A
verification's stop follows the same rules, checked in `_scratch.interrupt`.

### Finding the kernel

The host names the connection file itself, `kernel-biopb-<uuid>.json` in
`jupyter --runtime-dir`: left to `jupyter_client` it is a tempfile no Jupyter
tool looks for. The runtime dir is per-platform (`~/.local/share/jupyter/runtime`,
`~/Library/Jupyter/runtime`, `%APPDATA%\jupyter\runtime`) and private to the
user. `KernelHost.health()` carries `connection_file` and `attach_command`;
`server_status` prints both, and the observe page offers the command to copy:

```
<session python> -m qtconsole --existing <runtime-dir>/kernel-biopb-<uuid>.json
```

The command runs the session child's own interpreter: biopb installs ipykernel,
`jupyter_core` and qtconsole (through napari-console) but no Jupyter front end,
and its uv tool env puts none of their scripts on PATH, so a bare
`jupyter qtconsole` finds nothing or another install's. It is quoted for the
platform's shell (`shlex.join`, or `list2cmdline` on Windows), since a Windows
profile path can contain spaces, and works only on the machine the session runs
on. A frozen (PyInstaller) build reports no command, having no module tree to
run `-m` against. A notebook from any other Jupyter install attaches with the
connection file alone; the protocol does not care which environment the client runs in.

### What retired

The user console, entirely: its execute route, config key, control path root
and editor. The observe page keeps job history, Stop, restart and export — the
parts a notebook cannot do. The control's loopback-only gate guards only the
`chat` root, and `/health` reports it as `loopback_bound`.

### Attribution: `origin` on the job

`_Job` carries `origin` (`"mcp"` | `"user"` | `"chat"`, see
[chat-engines.md](chat-engines.md)), set when the job starts and carried
into the host's record, its snapshot, the job list and the export. A foreign client's cell
is `"user"`.

The writer count is two by construction: the first agent to run a cell claims
the kernel (`_writers.take_claim`, in the host, where every agent's cell enters;
it lasts one kernel) and a second agent's `execute_code` is refused (keyed on
the client id). A human's cell is never gated by that claim — an attached client carries no
identity to gate on — and the observe page's restart is never gated either, so a
stale claim is cleared by the person at the machine rather than seized by a
second agent. Every tool that changes kernel state (`execute_code`,
`interrupt_kernel`, `restart_kernel`) is gated the same way; `interrupt_kernel`
refuses a job the asker did not start.

### Informing the agent: pull, not push

The agent's picture of the namespace goes stale the moment a human runs a cell,
and it cannot be pushed a notice mid-turn — MCP server→client notifications are
not reliably surfaced, and an idle agent has no turn to interrupt. So every
agent-facing round trip (`execute_code`, `poll_job`, `server_status`) appends a
note at return time:

```
[2 cells were run by the user since your last call: job-7 (ok), job-8 (error).
Read them with poll_job('job-7'). Variables and layers may have changed.]
```

Each host record carries a `seen_by_agent` flag, read by `foreign_digest()` and
retired only by a later `ack_foreign_digest(ids)` once the note is rendered into
a result the agent will receive — reading never consumes. Only the kernel's
holder can ack, and only ids reported **terminal** are acked, without
re-reading status. The agent is told that something changed and where to look, never what
changed. `_MAX_RETAINED_JOBS` never evicts an unseen user job.

## Gotchas

- **`stop_on_error`.** Most clients send it true, so a foreign cell that
  errors makes ipykernel abort requests already queued behind it. The host
  retries an `aborted` snippet once after a short pause (`_execute_internal`),
  and sends its agents' cells with it false, so a failing agent cell aborts
  nothing of the user's.
- **A client's Ctrl-C** is a SIGINT to the kernel, which lands in whatever cell
  runs on the main thread -- the agent's included. The cell is recorded as
  interrupted. A task, on its worker thread, is out of its reach.
- **A long cell holds the main thread.** The host's calls queue behind it and
  time out after 120 s. A timeout sends **no interrupt**: it would land in
  whatever holds the main thread, which may be the user's cell. The timed-out
  request stays queued and runs later; its late reply is skipped by message id.
  The tools that need the main thread (`take_screenshot`, `inspect_object`)
  refuse at once instead while the agent's own cell holds it: asking then is a
  mistake in the agent's plan, and the refusal says to use `run_async`.
- **A queued agent cell reads as running.** The host records a cell when it
  sends it; one waiting behind a user's cell is running as far as its record
  says, and is not ended by that cell's end.
- **A hard-killed session leaves its connection file behind** in the runtime
  dir, as any Jupyter kernel does; a clean shutdown removes it. Tests that kill
  a launcher set `JUPYTER_RUNTIME_DIR` so they do not litter the real one.

## Follow-ups (not v1)

**JupyterLab.** Lab's kernel picker shares only kernels its own
`jupyter_server` started; there is no attach-by-connection-file. The fit is a
**kernelspec per live session**, written by the control when a session starts
(`~/.local/share/jupyter/kernels/biopb-<session>/kernel.json`, naming a
provisioner and carrying that session's connection file) and removed when it
ends, so Lab's launcher lists sessions by name. The provisioner (registered on
the `jupyter_client.kernel_provisioners` entry point, on the order of a hundred
lines) hands the server the existing connection info at launch, reports alive
while the kernel pid lives, and **no-ops kill and terminate** — Lab's restart
and close-notebook must not take the session's kernel down. Until it exists,
"connect a notebook" means qtconsole.

## Open: ipykernel version

**Everything here is verified on ipykernel 6.31 only.** `biopb-mcp` declares
`ipykernel` unbounded, but it is capped transitively: `napari==0.7.0` pulls in
`napari-console` 0.1.4, which requires `ipykernel<7`. So the lock, CI (which
installs from it), the nightly fresh resolve (`unlocked-resolve.yaml`) and a
user's `install.sh` all get 6.31, while PyPI has 7.3.0. The cap is someone
else's: a `napari-console` release that lifts it, or a napari upgrade that
brings one, moves every install to 7 with no change here. To do:

1. **Pin** `ipykernel>=6.31,<7` in `biopb-mcp`'s `mcp` extra, so the cap is
   this package's decision and survives a napari upgrade.
2. **Verify against 7.3** once the cap can move: the `test_mcp_*` suites and
   the napari smoke run in a venv with ipykernel 7.3, then move the pin.

What 7 has to be checked for, since the gate and the records lean on ipykernel
internals rather than the protocol:

- **The `do_execute` override.** 6.31 calls `async do_execute(code, silent,
  store_history, user_expressions, allow_stdin, *, cell_meta, cell_id)`, and
  the kernel class reads the request's session and `msg_id` from
  `get_parent("shell")`. Both signature and accessor have to hold, or Stop
  cannot find a cell and silent foreign cells go unrecorded.
- **Worker-thread output attribution.** A `run_async` task's prints reach
  iopub under the cell that started it because 6.31 maps a thread started
  during a request to that request (`_associate_new_top_level_threads_with`).
  The task records depend on it (see Record); without it a task's output is
  filed under no job. Cells do not: theirs is their own request's.
- **Announcements from a worker thread.** A task's start and end, sent on the
  IOPub thread through its private `_really_send`, ordered after a stream flush
  from the task's thread.
- **The control channel.** A custom request type registered through
  `control_msg_types` and served on the control thread (see Control channel),
  and control requests publishing busy/idle under their own parent.
- **Interrupt semantics.** Today: SIGINT is honoured only while a request is
  being serviced (`_jobs._EXTERNAL_INTERRUPT_MSG` leans on this), a client's
  Ctrl-C or Stop lands a main-thread cell's SIGINT in the main thread, and Stop
  raises into a job's worker thread with `PyThreadState_SetAsyncExc`, which
  does not break a blocking C call until it returns. Under 7: whether SIGINT is still gated the same way,
  where it lands with subshells running, and whether an interrupt reaches a
  subshell at all.
- **`stop_on_error`**: an errored or refused cell still aborts what is queued
  behind it with status `aborted`, which the host retries once.

## v2 shape

Two independent changes, either of which can come first.

**Subshells (ipykernel 7).** The host's tools move onto a subshell of their
own, so a foreign cell on the main shell no longer holds them up. That is what
turns the refusal into a bounded wait: the human's cell waits for the job,
costing nobody but the human who chose to wait. Limits: anything touching the
viewer still marshals to the main thread and waits there, and an agent submit
could then overlap a user cell, so the gate needs a lock shared by `submit`
and `record_inline`.

**Blocked by napari, not by verification.** `napari-console` requires
`ipykernel<7` in its latest release (0.1.4), and napari depends on it
unconditionally -- 0.7.0 and the current 0.9.1 alike -- so no napari upgrade
reaches ipykernel 7. The paths: a `napari-console` release that lifts the cap;
or a uv override of it, which every install path (`install.sh`, the lock, the
nightly fresh resolve) would have to carry, against a cap that presumably
guards napari's own console widget. The lifecycle calls that must not queue
behind a user's cell -- Stop and restart's graceful close -- are on the
control channel instead (done, see Control channel). That leaves the other
host calls queueing behind a user's cell, which subshells would not fully fix
either: anything touching the viewer needs the main thread.

**The host subscribes to iopub**, in three stages:

1. *Transport (done).* A threaded client routes every reply and iopub message
   by parent message id (`mcp/_kernel_io.py`). Round trips overlap instead of
   queueing on a host lock, and "busy" is the kernel's own published status.
2. *Records (done).* The kernel announces a job's start and end on iopub, and
   the host keeps the records (see Record). A job's output streams in live,
   the observe page's polling no longer sends execute requests into the
   kernel, records survive a restart, and the in-kernel tee is gone.
3. *Verification (done).* Superseded by v3 stage 3: each cell is its own
   request, recorded like any cell. A scratch kernel runs no watchdog, so
   `_scratch` ends the record itself when the process dies, failing the cell
   it died in.

This is a refactor of how the host talks to the kernel, orthogonal to
subshells and to the gate, which stays in-kernel either way — only kernel code
can refuse a request before it runs.

## Shape of the work

1. `_kernel_gate.py`: `GatedKernel(IPythonKernel)` with `do_execute`; launch
   argument in `__main__`; the host's session id in the launch environment.
   Tests in `_tests/test_mcp_kernel.py` with a second `jupyter_client` attached
   to the real kernel: refused while a job runs, recorded when idle, silent code
   gated, an empty request passing, a poll during a refusal.
2. `_jobs.py`: a `record_inline` context the gate wraps a foreign cell in.
   Tests in `test_mcp_jobs.py`. (The records later moved to the host, v2.)
3. `health()` / `server_status` / observe status: the connection file and the
   attach command.
4. Retire the console (own PR): routes, path root, control gate, page, doc.
