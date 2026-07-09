# Plan: an FS-access guardrail for agent `execute_code` (the audit-hook speed-bump)

Status: **proposed** (design only; not implemented). Scopes a *portable,
cross-platform* filesystem guardrail for the code the agent runs via
`execute_code`, plus the Linux-only path that turns it into a real boundary.

## 1. The problem

`execute_code` runs **arbitrary agent Python** in the child kernel, against a
namespace (`viewer`, `client`, `ops`, `np`, `da`, `skimage`) that includes the
full standard library. Nothing stops that code from `open(...,'w')`,
`os.remove`, `shutil.rmtree`, or reading an unrelated file under `$HOME`. On the
intended **personal / localhost / trusted** deployment the realistic hazard is
not an attacker but the **agent making a mistake** — overwriting the user's data,
deleting the wrong directory, reading something it shouldn't relay.

We want to make that class of accident *hard to do by default*, without
disturbing the parts of the kernel that legitimately touch the disk (napari, Qt,
dask spill, pyarrow, the localhost cache-file `mmap` fast path).

### Why not a "real" sandbox

A genuine FS boundary is an OS kernel feature, and every OS exposes a different,
incompatible one: **Landlock** (Linux 5.13+), **Seatbelt** / `sandbox-exec`
(macOS), **AppContainer** + ACLs (Windows). There is **no simple cross-platform
boundary** — cross-platform and real-boundary are mutually exclusive here. The
only portable layer is an in-process check, which is a *speed-bump*, not a
boundary. That is exactly the right proportion for the accident threat model; the
real boundary is deferred to the hosted case (§6), which is Linux-only anyway.

## 2. Guarantee we are buying

**Honest scope.** With the guardrail on, agent code that does ordinary file I/O
through Python (`open`, `os`, `shutil`, `pathlib`, `io`) is denied any write —
and, optionally, any read — outside a configured allowlist, and fails with a
clear `PermissionError` naming the working dir. It stops **accidents**.

Out of scope (the residual in-process limit, stated plainly): agent code that
*deliberately* evades the hook — `ctypes` into libc, a raw syscall, or a
`subprocess`/child process the parent's audit hook does not cover. Closing that
needs an OS boundary (§6). We accept the gap on desktop and document it.

This is the same honesty posture as `viewer-thread-safety.md`: the portable layer
makes the common failure impossible-by-accident; the adversarial residual is
named, not hidden.

## 3. Mechanism: `sys.addaudithook`

Python's audit hooks (PEP 578) fire on the events that precede real file access —
`open`, `os.rename`, `os.remove`, `os.mkdir`, `shutil.*`, `os.scandir`, etc. A
hook installed at bootstrap inspects the target path and raises `PermissionError`
when it falls outside the allowlist:

```python
def _install_fs_guard(allow_write, allow_read, *, enforce_reads):
    aw = [os.path.realpath(p) for p in allow_write]
    ar = [os.path.realpath(p) for p in allow_read]

    def _hook(event, args):
        if event == "open":
            path, _mode, flags = args
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                                    | os.O_TRUNC | os.O_APPEND))
            _check(path, aw if writing else ar,
                   deny=writing or enforce_reads)
        elif event in _MUTATING_EVENTS:      # os.remove/rename/mkdir/…
            _check(args[0], aw, deny=True)

    sys.addaudithook(_hook)                   # cannot be removed once set
```

Properties that make this the right portable primitive:

- **Fires below the API surface** — one hook covers `open`, `pathlib`, `io`,
  `numpy.save`, `tifffile.imwrite`, … because they all funnel through the same
  audit events. No per-library allowlist to keep in sync.
- **Cannot be unregistered** — `sys.addaudithook` is add-only; agent code cannot
  pop it off.
- **Zero dependencies, every OS** — pure CPython.

Default posture: **enforce writes, log reads** (`enforce_reads=False`). Writes are
where accidents destroy data; blanket read-denial is more likely to break a
legitimate `napari`/`skimage` import path and is lower value for the accident
model. `enforce_reads=True` is a config opt-in.

## 4. The allowlist (biopb-specific — the parts that are easy to miss)

A naive "only the working dir" allowlist silently breaks the kernel. The real
sets:

**Writable (`rw`):**
- the kernel working dir (`KernelHost._launch` passes `cwd`) — where results and
  agent scratch belong;
- the **dask spill dir** (`DaskClusterHost` `local_dir`);
- a private `$TMPDIR` (pyarrow / napari / Qt scratch).

**Readable (only relevant when `enforce_reads=True`):**
- the Python prefix / `site-packages` and napari/Qt resource dirs;
- `/dev/dri` and the X11/Wayland socket (GPU + display);
- **the tensor server's cache directory** — easy to miss. The localhost
  cache-file `mmap` fast path (`chunk_locate`, see the root `CLAUDE.md` §3) has
  the *client* `mmap` the *server's* on-disk segment files directly. Omit the
  server cache root from the read allowlist and every localhost read silently
  falls back to the slower `do_get` socket. So read-enforcement must allowlist
  the server cache root, or accept losing the fast path.

## 5. The two-process-group caveat (do not skip)

The kernel and the dask workers are **different processes in different groups**:
the workers are daemon-owned (`DaskClusterHost` spins `LocalCluster(processes=
True)`), not children of the kernel. An audit hook installed in the kernel does
**not** run in the workers.

So agent file I/O performed *inside a dask task* —
`da.map_blocks(lambda b: open('/etc/…'))`, a custom `da.store` target — executes
on a worker and **bypasses the kernel's hook entirely**. To actually bound FS the
same guard must be installed in each worker via a **`WorkerPlugin` / worker-init**
— the exact mechanism already used to split `cache_budget` across workers
(`CLAUDE.md` §3, `mcp.dask.cache_budget`). Kernel-only enforcement is a partial
guard and must be documented as such.

Both sites read the **same allowlist** from config so they cannot drift.

## 6. Wiring

- **Config:** a new `mcp.sandbox.fs` block in `_config.py` `DEFAULT_CONFIG` —
  `{ "enabled": bool, "enforce_reads": bool, "allow_write": [...],
  "allow_read": [...] }`. Default `enabled` true, `enforce_reads` false. The
  workdir / spill dir / tensor-cache root are resolved and appended at bootstrap,
  not hard-coded in config.
- **Kernel:** install in `mcp/_bootstrap.py`, at the **tail** of bring-up —
  *after* the viewer, dask, and the data connection are wired (those legitimately
  open files during setup), so the guard governs agent code only, never the
  bootstrap itself.
- **Workers:** a `WorkerPlugin` registered where the cluster is created
  (`mcp/_cluster.py`), applying the identical hook + allowlist in each worker's
  `setup`.
- **Errors:** `PermissionError` with a message that names the allowed workdir, so
  the agent gets an actionable signal ("writes are restricted to `<workdir>`;
  save through napari or `client.upload_array` instead") rather than a bare
  denial — steering it back to the sanctioned data-plane I/O path.

## 7. Relationship to the data plane (why this is practical here)

Unlike a generic Python REPL, biopb has a **designed alternative to raw file
I/O**: legitimate data access goes through `client` (→ lazy dask arrays) and
results go back via `client.upload_array` / the napari viewer, per the MCP
operation guardrails ("avoid the filesystem unless the user explicitly asks").
So a tight FS allowlist rarely obstructs a real workflow — the agent was not
supposed to be reading raw files anyway. The guardrail *encodes* an existing norm
rather than imposing a new constraint.

## 8. The upgrade path (out of scope here, noted for continuity)

When the threat model hardens — untrusted agents, or a **hosted / multi-tenant**
deployment — the in-process hook is no longer sufficient and the residual gap in
§2 must be closed with an OS boundary. That case is **Linux-only** (hosted biopb
runs on Linux), so the portability problem disappears: apply **Landlock** as a
self-restriction at the same bootstrap tail (and the same `WorkerPlugin`), reusing
this doc's allowlist verbatim, or rely on the deployment container's bind-mounts.
The audit hook then remains as the friendly-error layer in front of the
kernel-level denial. Consistent with the project's stance (root `CLAUDE.md` §1) of
keeping the services simple and pushing real hardening to the deployment layer.
