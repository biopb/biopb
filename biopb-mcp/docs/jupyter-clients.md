# The session kernel — agent, user and host on one ipykernel

The session kernel is an ordinary ipykernel, launched by `KernelHost`
(`mcp/_kernel.py`) with a kernel class of its own (`mcp/_kernel_gate.py`). The
agent, a human in any Jupyter client attached by connection file, and the host's
own tools all talk to it over the Jupyter protocol. Guarantees: **writers take
turns the way notebook cells do**, **the agent is told of every cell it did not
run**, and **Stop and shutdown never wait behind a running cell**.

## Why

A human working in the session wants a real notebook — the same namespace, the
same `viewer`, the same `client` — not a console bolted onto the observe page.
ipykernel already provides most of what that takes: it runs execute requests one
at a time on the main thread in arrival order, whichever client sent them, and a
client ignores iopub traffic from other sessions (qtconsole's
`include_other_output` defaults off). So the design leans on the protocol and
adds only what it lacks: telling the writers apart, recording what each ran,
and a way to stop or close the kernel that does not queue behind a cell.

## Who runs where

- **A cell — anyone's — runs on the kernel's main thread.** `execute_code`
  sends the agent's code as a plain execute request (`KernelHost.run_cell`),
  exactly as a notebook does, so ipykernel orders the agent's and the user's
  cells together. A user's cell sent while the agent's runs waits its turn.
- **A `run_async` task runs on a worker thread**, beside the cells. The agent
  opts into it for a long compute (`_jobs.run_async`), leaving the main thread
  — and so Qt and the screenshots — free. One task at a time. Its viewer calls
  are marshaled to the main thread ([viewer-thread-safety.md](viewer-thread-safety.md)).
- **The host's snippets** (screenshot, inspect, status) are execute requests
  too, queued like any cell. Stop and the graceful close go on the control
  channel instead (below).

**The agent does not queue.** `execute_code` is refused while anything runs —
a cell, anyone's, or a task — since an agent whose cell waited behind others
would lose track of what ran when. The host decides from its records, under one
lock with the send (`_server._submit_job`).

**Main-thread tools fail fast.** While the agent's own cell holds the main
thread, `take_screenshot` and `inspect_object` refuse at once rather than queue
behind it: asking then is a mistake in the agent's plan, and the refusal names
`run_async`.

## Telling writers apart

The host's client has one session id for the kernel's life, and it exists
before the kernel does (`KernelManager.client()` inherits the manager's).
`KernelHost` hands it over at launch in `BIOPB_HOST_SESSION`, so the kernel
knows its host before the connection file lets anyone in, and no cell can
re-adopt it. A request from any other session is **foreign**: recorded as the
user's. The id is not a secret — it rides in the parent header of every iopub
message the host's requests produce — so it keeps honest clients apart and
stops nobody holding the connection file, who can run anything anyway.

**`silent` does not hide a cell.** It only stops output being broadcast; the
code runs with full effect. ipykernel echoes no silent request, so the kernel
class echoes a non-empty silent foreign cell itself (`_publish_execute_input`)
and the host records it. What clients actually send silently is empty —
qtconsole's prompt-number request, `user_expressions` for its UI — and empty
code is never a cell.

**Who wrote it** is `origin` on the record: `"mcp"` (the `execute_code` tool),
`"chat"` (the in-process chat loop, [chat-engines.md](chat-engines.md)) or
`"user"` (a foreign client). A task takes the origin of the cell that started
it.

**One agent per kernel.** The first agent to run a cell claims the kernel
(`_writers.take_claim`, in the host, where every agent cell enters; it lasts one
kernel generation) and a second agent is refused. A human's cell is never gated
— an attached client carries no identity to gate on — and neither is the observe
page's restart, so a stale claim is cleared by the person at the machine, not
seized by a second agent.

## Records

The job records live in the host (`mcp/_job_log.py`), built from the protocol.
The host names every job — `job-N`, one counter for the host's life, so ids
never repeat across restarts — and every `stream`, `execute_result` and `error`
under a job's request is that job's.

- **A cell** starts at its `execute_input` (a foreign cell), or when the host
  sends it (the agent's, `JobLog.start_cell`) — but only *begins* at its own
  `execute_input`, since it may be queued behind a user's cell. An iopub
  `error` fails it (`KeyboardInterrupt`: interrupted), and the `status: idle`
  for its request ends it. The host's own cells also have their shell reply,
  which cannot be lost: it ends a cell whose idle was lost, and fails one whose
  iopub `error` was.
- **A task** outlives the cell that started it, so the kernel announces its
  start and end as `biopb_job` messages (`_jobs._publish`). ipykernel files a
  worker thread's prints under the request that started the thread; from the
  task's announced start they are the task's, not the cell's.

iopub is a PUB socket and can drop. What is lost, the order repairs: cells run
one at a time and tasks one at a time, so a cell's end ends any other cell that
had begun, and a task's start any other task, as "end not recorded". A cell
never ends a task.

Every read — `poll_job`, the observe list and detail, the notebook export, the
foreign-activity digest — is a read of the host's memory, never a kernel round
trip. Records outlive a kernel restart; one still running when its kernel goes
ends as interrupted. `display_data` is not recorded: a record says what ran and
how it ended, not what it drew.

A verification runs the same way in a scratch kernel: each cell its own request,
recorded like any cell ([verify-workflow.md](verify-workflow.md)).

## Informing the agent: pull, not push

The agent's picture of the namespace goes stale the moment a human runs a cell,
and it cannot be pushed a notice mid-turn — MCP server→client notifications are
not reliably surfaced, and an idle agent has no turn to interrupt. So every
agent-facing round trip (`execute_code`, `poll_job`, `server_status`) appends a
note at return time:

```
[2 cells were run by the user since your last call: job-7 (ok), job-8 (error).
Read them with poll_job('job-7'). Variables and layers may have changed.]
```

Each record carries `seen_by_agent`, read by `foreign_digest()` and retired only
by `ack_foreign_digest(ids)` once the note is in a result the agent will
receive — reading never consumes. Only the kernel's holder acks, and only ids
reported terminal. The agent is told that something changed and where to look,
never what changed. Eviction never drops a record the agent has not been told
of.

## Control channel

What must not wait behind a cell goes on the control channel, which ipykernel
serves on its own thread. One custom request, `biopb_request`
(`GatedKernel.biopb_request`), carries an op and its arguments: `interrupt` and
`close` (the graceful close before a kill: the tensor client's Flight
connection shut cleanly). The op runs on a worker thread under a timeout,
because control requests are handled one at a time and ipykernel's own
interrupt and shutdown share that queue. Refused for any session but the
host's.

**Stop names its job.** `KernelHost.interrupt_job` sends the kernel what it
knows the job by — a cell's request (a cell's id is the host's), a task's id —
and the kernel checks, under `_jobs._lock`, that it still runs before touching
anything. A stale id stops nothing, and the reply names what runs now.

- **A cell** gets a real `SIGINT`, to the kernel's main thread only
  (`km.interrupt_kernel` would signal the whole process group, dask workers
  included), which also wakes a blocking sleep or wait. It is sent under the
  lock the cell is ended under (`_jobs.hold_cell`), so it lands in that cell or
  in the lock wait that ends it, never in the next one.
- **A task** gets a `KeyboardInterrupt` raised into its thread
  (`PyThreadState_SetAsyncExc`), which lands at the next bytecode.
- **A blocking dask call** needs nothing more: a dask `Client`'s wait
  (`distributed.utils.sync`) cancels that call's futures on a
  `KeyboardInterrupt`. A cell's stop breaks the wait at once; a task's lands
  when the wait next wakes, within 10 s (hardcoded there).

**Who may stop what is the host's decision** (`_writers.stop_refusal`): a client
that does not hold the kernel is refused, and so is a job another writer
started — the stop would be silent to them. The person at the machine (the
observe page) may stop anything. A verification's stop follows the same rule
(`_scratch.interrupt`).

**The busy/idle around a control request is ignored**: an idle after it would
read as idle while a cell still holds the main thread.

## Finding the kernel

The host names the connection file itself — `kernel-biopb-<uuid>.json` in
`jupyter --runtime-dir`, private to the user — since left to `jupyter_client` it
is a tempfile no Jupyter tool looks for. `KernelHost.health()` carries
`connection_file` and `attach_command`; `server_status` prints both, and the
observe page offers the command to copy:

```
<session python> -m qtconsole --existing <runtime-dir>/kernel-biopb-<uuid>.json
```

It runs the session's own interpreter: biopb installs qtconsole (through
napari) but puts no `jupyter` script on PATH, so a bare `jupyter qtconsole`
finds nothing or another install's. It is quoted for the platform's shell and
works only on the session's machine; a frozen build reports none. A client
from any other Jupyter install attaches with the connection file alone.

**JupyterLab cannot attach**: its kernel picker lists only kernels its own
server started. The fit is a per-session kernelspec naming a provisioner that
hands over the existing connection and no-ops kill and restart; not built.

## Gotchas

- **A long cell holds the main thread.** The host's snippets queue behind it
  and time out after 120 s. A timeout sends **no interrupt** — it would land in
  whatever holds the main thread, likely the user's cell. The request stays
  queued and its late reply is skipped by message id.
- **`stop_on_error`.** Most clients send it true, so a user's failing cell makes
  ipykernel abort what is queued behind it. The host retries an `aborted`
  snippet once, and sends the agent's cells with it false, so a failing agent
  cell aborts nothing of the user's.
- **A client's Ctrl-C** is a SIGINT to the kernel: it lands in whatever cell
  holds the main thread, the agent's included, which is recorded as
  interrupted. A task is out of its reach.
- **The Qt window freezes while a cell runs**, as in any notebook with
  `%gui qt`; `run_async` is the way out. A task can race a user's cell over the
  namespace and the viewer, as in any asynchronous notebook — the agent sees it
  in the records.
- **A late stop.** A `SIGINT` arriving after a cell's code returned, while
  ipykernel wraps it up, surfaces there as any Jupyter interrupt can; the
  kernel class still sends the client a reply.
- **A hard-killed session leaves its connection file** in the runtime dir, as
  any Jupyter kernel does. Tests that kill a launcher set `JUPYTER_RUNTIME_DIR`.

## ipykernel 6.31 only

Every install gets ipykernel 6.31: `napari-console` 0.1.4, which every napari
requires, caps it below 7. This design leans on 6.31 internals, which a move to
7 has to re-verify:

- **`do_execute`'s signature** and `get_parent("shell")` in the kernel class —
  without them Stop cannot find a cell and silent foreign cells go unrecorded.
- **Thread-to-request output attribution**
  (`_associate_new_top_level_threads_with`), which files a task's prints.
- **`IOPubThread._really_send`**, which orders a task's announcements after
  its output.
- **Custom control messages** (`control_msg_types`) and their busy/idle under
  their own parent.
- **SIGINT handling**: honoured only while a request is serviced
  (`_jobs._EXTERNAL_INTERRUPT_MSG` leans on it), and where it lands once
  subshells exist.

ipykernel 7's subshells would let the host's snippets run beside a user's cell
instead of queueing, but not viewer work, which needs the main thread either
way.
