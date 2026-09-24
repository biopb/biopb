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

## What already holds

- **ipykernel serializes.** Every execute request on the shell channel runs
  one at a time on the main thread, in arrival order, whatever client sent it.
  A human cell, the host's poll snippet and a screenshot queue behind each other.
- **The agent's job is the one thing outside that order**, by our design: the
  submit snippet starts a worker thread and returns, so the kernel is idle on
  the shell channel while the job runs. That is what keeps `poll_job`,
  `take_screenshot` and the Qt loop live during a long compute. A foreign cell
  gets the same treatment, since the kernel cannot tell a poll from a cell.
- **A client ignores the host's traffic.** qtconsole and Lab drop iopub
  messages whose parent session is not their own (qtconsole's
  `include_other_output` defaults off); only the busy/idle indicator reacts.
- **A job's prints go to iopub under the host's request.** ipykernel attributes
  a worker thread's output to the request that started the thread, the host's
  submit, so a client sees the agent's code and output only if it opts into
  other-output.

The host adds no ordering of its own. Its one threaded client routes every
shell reply and iopub message to the call it answers (`mcp/_kernel_io.py`), so
its round trips overlap and the kernel orders them with everyone else's.

## The rule: reject, never block, never reroute

While a job thread is live, a foreign cell is **refused** with a reply naming
the job. Otherwise it runs inline on the main thread, as a notebook user
expects, and is **recorded** as a job with `origin="user"`.

Two alternatives were rejected:

- **Blocking the cell until the job ends.** The blocked cell holds the shell
  channel, so the host's polls and screenshots queue behind it for the rest of
  the job — exactly the responsiveness the worker thread was built to buy —
  and the host's execute timeout (120 s) then SIGINTs the kernel, which lands
  in the waiting cell and aborts it. The wait would also have to pump Qt, since
  the job's `run_on_main` calls need the main thread it is blocking. A bounded
  wait is the most this can ever be; v2 (below) is what makes waiting cheap.
- **Rerouting the cell through `_jobs.submit`.** The cell becomes
  asynchronous: the user gets a job id and their output lands in a buffer they
  cannot see. Waiting on it has the deadlock above.

The reverse race needs nothing: an agent submit arriving during a long human
cell is an execute request and queues behind it.

## Design

### The gate lives in `do_execute`, not in an IPython event

A kernel subclass, passed as `--IPKernelApp.kernel_class=...` beside the
existing `exec_lines` argument, overrides `do_execute`. Not `pre_run_cell`:
IPython wraps event callbacks in a try/except that prints and continues, so a
callback cannot refuse a cell, and a Ctrl-C during a wait there is swallowed
and the cell runs anyway. `do_execute` is also where the two facts the gate
needs are at hand: the request's parent session id (`get_parent("shell")`) and
its code.

### Foreign is "not the host's session"

The host's own client has one session id for the kernel's life, and it exists
before the kernel does: `KernelManager.client()` inherits the manager's session
id. `KernelHost` hands it over at launch (`BIOPB_HOST_SESSION`), so the gate is
armed before the connection file lets anyone connect, and nothing a cell sends
can re-adopt it. Learning it from a marker request instead left every request
ungated until that marker ran, and let any client run the marker. A kernel
restart is a new manager, so a new id. A request from any other session is
foreign.

The session id is not a secret: it rides in the parent header of every iopub
message the host's requests produce. The gate keeps honest clients from writing
over each other; anyone holding the connection file can run anything anyway.

**An empty cell passes untouched, foreign or not** — the decision is on the
code, not on `silent`. `silent` only stops output being broadcast; silent code
runs with full effect, so it is gated like any cell. What clients actually send
silently is empty: qtconsole's prompt-number request, and `user_expressions`
evaluated for its UI. Completion and inspection are other message types and
never reach `do_execute`. The remaining gap: `user_expressions` on an empty
request are evaluated even while a job runs. They are expressions, and nothing
records them.

### Refusal

If `_jobs` has a running job, the override returns an `error` reply and
publishes an iopub `error` message under the request's parent (a reply alone
renders nothing in a notebook cell). The text names the job id, its intent and
elapsed time, and says where to stop it: the observe page's Stop, which calls
`interrupt_current` with user origin, is never gated, and reaches the worker
thread that a client's own interrupt cannot. There is no in-kernel escape
hatch; a `_jobs.interrupt_current()` cell would itself be refused, and the
observe page is the tool for it.

### Record

The records live in the host (`mcp/_job_log.py`), built from iopub. The kernel
announces each job's start and end as a `biopb_job` message
(`_jobs._publish`) naming the request its output is published under; every
`stream`, `execute_result` and `error` under that request is the job's. A
foreign cell that runs is announced as `origin="user"` by the gate
(`_jobs.record_inline`): source from the request, status from the reply, its
output going to its own client as usual and filed by the host from the same
iopub. An agent's job is announced by `submit`, its worker thread's prints
attributed to the submit request.

The announcement has no parent header: a client drops iopub from other
sessions, and an unknown message type under its own request would be one more
thing for it to ignore. Streams are flushed before the end is announced, so a
job's output is complete when its record says it ended. iopub is a PUB socket
and can drop under pressure; a start while another record is still running
ends that one as "end not recorded", since only one job runs at a time.

Poll, the observe list and detail, the notebook export and the foreign-activity
digest are reads of the host's memory, never a kernel round trip. Records
outlive a kernel restart: one still running when its kernel goes is ended as
interrupted, and the next kernel's job ids continue from the host's
(`BIOPB_JOB_SEQ`). Display output (`display_data`) is not recorded: the record
says what ran and whether it failed, not what it drew.

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

The user console: its execute route (`/console/execute`,
`observe.console_enabled`), the control's `console` path root, and the editor
on the observe page. The observe page keeps job history, Stop, restart and
export — the parts a notebook cannot do. The control's loopback-only gate now
guards only the `chat` root; `/health` still reports it as `console_enabled`.
A leftover `observe.console_enabled` in a config file is ignored.

### Attribution: `origin` on the job

`_Job` carries `origin` (`"mcp"` | `"user"` | `"chat"`, see
[chat-engines.md](chat-engines.md)), set when the job starts and carried
into the host's record, its snapshot, the job list and the export. A foreign client's cell
is `"user"`.

The writer count is two by construction: the first non-user submitter claims
the kernel and a second agent's `execute_code` is refused (keyed on the client
id). A human's cell is never gated by that claim — an attached client carries no
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
  errors — including the refusal — makes ipykernel abort requests already
  queued behind it. The host retries an `aborted` snippet once after a short
  pause (`_execute_locked`); the gate's test should land a poll on a refusal.
- **A client's Ctrl-C** reaches the main thread only. If the agent's job is
  inside a `run_on_main` slot at that moment, the interrupt lands in the job,
  which the runner labels as an external interrupt naming an attached client.
- **A long human cell holds the shell channel.** The host's calls queue behind
  it and time out after 120 s (the observe page's poll among them). A timeout
  sends **no interrupt**: the host once SIGINTed the kernel on one, from when
  agent code ran on the main thread, and that landed in whatever held it — the
  human's cell. The timed-out request stays queued and runs later; its late reply
  is skipped by message id.
- **A hard-killed session leaves its connection file behind** in the runtime
  dir, as any Jupyter kernel does; a clean shutdown removes it. Tests that kill
  a launcher set `JUPYTER_RUNTIME_DIR` so they do not litter the real one.
- **Busy is not an error.** A refused cell while the agent computes is the
  design working; the message says what is running and how to stop it.
- **`execute_input` is published before `do_execute`**, so a refused cell
  still echoes its input to other-output listeners. Harmless.

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

## v2 shape

Two independent changes, either of which can come first.

**Subshells (ipykernel 7).** The host's tools move onto a subshell of their
own, so a foreign cell on the main shell no longer holds them up. That is what
turns the refusal into a bounded wait: the human's cell waits for the job,
costing nobody but the human who chose to wait. The gate and the record are
unchanged. The lock holds ipykernel 6.31 and the dependency is unpinned;
whether an interrupt reaches a subshell is unverified.

**The host subscribes to iopub**, in three stages:

1. *Transport (done).* A threaded client routes every reply and iopub message
   by parent message id (`mcp/_kernel_io.py`). Round trips overlap instead of
   queueing on a host lock, and "busy" is the kernel's own published status.
2. *Records (done).* The kernel announces a job's start and end on iopub, and
   the host keeps the records (see Record). A job's output streams in live,
   the observe page's polling no longer sends execute requests into the
   kernel, records survive a restart, and the in-kernel tee is gone. Job calls
   that still enter the kernel (submit, interrupt) return their result as a
   `user_expression` rather than a printed line, which a job's output under
   the same request could split.
3. *Verification.* Scratch runs move onto the same records, with per-cell
   boundary events, and the in-kernel output capture is deleted.

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
