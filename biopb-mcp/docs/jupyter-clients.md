# Jupyter clients on the session kernel

Status: **implemented** except the console's retirement (step 4 of *Shape of
the work*), which is its own change. Decided 2026-09-23.

**Component:** `biopb-mcp` — a kernel subclass in a new in-kernel module
(`mcp/_kernel_gate.py`), `mcp/_jobs.py` (the record), `mcp/_kernel.py` and
`mcp/_server.py` (the connection-file hint). Retires `user-console.md`.

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
- **A job's prints never reach iopub.** `_JobStream` diverts a worker thread's
  output into the job's buffer, so a client sees the agent's code only if it
  opts into other-output, and never its output.

The host's `RLock` is not part of this story. It guards the *client*, because
`execute_interactive` discards iopub messages that are not its own; it does not
order anything in the kernel.

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
needs are arguments rather than lookups: the request's parent session id
(`get_parent("shell")`) and its `silent` flag.

### Foreign is "not the host's session"

The host's own client has one session id for the kernel's life. The kernel
learns it from a marker snippet the host runs before readiness, next to the
health probe: `_kernel_gate.adopt_host_session()` records the parent session
of *that* request. No id is passed through the environment and nothing is
generated up front. A request from any other session is foreign. A request
with `silent` set — a client's hidden execution for completion or inspection —
passes through untouched, foreign or not.

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

A foreign cell that runs is wrapped in a `_Job(origin="user")` for its
duration: source from the request, status from the reply, output **teed** from
the main-thread stream — `_JobStream` gains a tee mode for the main thread,
writing both to the record and to the real ipykernel stream, so the notebook
still sees its own output. `foreign_digest`, `ack_foreign_digest`, the observe
history and the notebook export read `origin="user"` jobs already; none of
them change. Rich output (`display_data`, `execute_result`) is not recorded:
the record says what ran and whether it failed, not what it drew.

### Finding the kernel

The host names the connection file itself, `kernel-biopb-<uuid>.json` in
`jupyter --runtime-dir`: left to `jupyter_client` it is a tempfile no Jupyter
tool looks for. The runtime dir is per-platform (`~/.local/share/jupyter/runtime`,
`~/Library/Jupyter/runtime`, `%APPDATA%\jupyter\runtime`) and private to the
user. `KernelHost.health()` carries `connection_file` and `attach_command`;
`server_status` prints both, and the observe page offers the command to copy:

```
jupyter qtconsole --existing <runtime-dir>/kernel-biopb-<uuid>.json
```

The command is built by the session child and quoted for its platform's shell
(`shlex.quote`, or `list2cmdline` on Windows), since a Windows profile path
can contain spaces. It works only on the machine the session runs on.

qtconsole is already in the environment (a napari dependency).

### What retires

The console's execute route, its `console` path root and the public-bind
refusal in the control, the textarea and its rendering, and `user-console.md`.
The observe page keeps job history, Stop, restart and export — the parts a
notebook cannot do. The console doc's Serialization, Attribution and Informing
sections carry over as they stand, because the record is the same `_Job`.

## Gotchas

- **`stop_on_error`.** Most clients send it true, so a foreign cell that
  errors — including the refusal — makes ipykernel abort requests already
  queued behind it. The host retries an `aborted` snippet once after a short
  pause (`_execute_locked`); the gate's test should land a poll on a refusal.
- **A client's Ctrl-C** reaches the main thread only. If the agent's job is
  inside a `run_on_main` slot at that moment, the interrupt lands in the job,
  which the runner labels as an external interrupt. That label names tool
  probes as the likely source and should name a client too.
- **A long human cell still holds the channel.** The host's tools queue behind
  it and its 120 s timeout still applies. Unchanged from today; the gate does
  not make it worse.
- **A hard-killed session leaves its connection file behind** in the runtime
  dir, as any Jupyter kernel does; a clean shutdown removes it. Tests that kill
  a launcher set `JUPYTER_RUNTIME_DIR` so they do not litter the real one.
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

**The host subscribes to iopub.** A reader thread in the host demuxes iopub
by parent message id into records kept in the host process. With the
`_JobStream` diversion removed, ipykernel attributes a worker thread's prints
to the submit request, so a job's output streams in live and `poll_job` becomes
a read of host memory: the observe page's polling stops sending execute
requests into the kernel, the client `RLock` and its busy status lose their
purpose (nothing can eat another request's iopub any more), and a poll can no
longer overrun and SIGINT the kernel. Records survive a kernel restart and can
hold rich outputs, so the in-kernel tee comes out again. This is a refactor of
how the host talks to the kernel, orthogonal to subshells and to the gate,
which stays in-kernel either way — only kernel code can refuse a request
before it runs.

## Shape of the work

1. `_kernel_gate.py`: `GatedKernel(IPythonKernel)` with `do_execute`,
   `adopt_host_session`; launch argument in `__main__`; marker snippet in
   `KernelHost` before readiness. Tests in `_tests/test_mcp_kernel.py` with a
   second `jupyter_client` attached to the real kernel: refused while a job
   runs, recorded when idle, `silent` passes, a poll during a refusal.
2. `_jobs.py`: `_JobStream` tee for the main thread; a `record_inline`
   context the gate wraps a foreign cell in. Tests in `test_mcp_jobs.py`.
3. `health()` / `server_status` / observe status: the connection file and the
   attach command.
4. Retire the console (own PR): routes, path root, control gate, page, doc.
