---
description: What a procedure's Requirements line names, where to check each, and what to do when one is missing.
---

# Checking what a procedure needs

A procedure doc opens with a **Requirements** line naming what its steps touch.
Resolve it before you start: one that assumes a plugin or package it does not
have fails partway through, after the user has already waited.

**It informs; it does not gate.** A gap is a fact to tell the user and work
around, not a stop sign. Take the doc's fallback where it names one, and where
it does not, say what you are substituting and why before you spend their time
on it — a missing data plane means pixels come from the viewer instead, and a
missing package usually has a slower or cruder equivalent in scipy/skimage. What
you must not do is proceed silently: the user cannot judge a result whose method
they were never told changed.

## Where each one is answered

One `server_status` call answers everything but a third-party package: the
viewer under `## Viewer`, the data plane under `## Tensor Server`, dask under
`## Dask`, the ops under `## Ops`, and a kernel plugin under `## Kernel
plugins` — the only place a plugin can be read, since a plugin contributes its
*function* names and not its own, so `dir()` cannot answer it, and a file that
failed to load is still on disk, so a listing cannot either.

A third-party package you resolve here, in two steps: `import <name>` answers
whether it is present, and `importlib.metadata.version("<name>")` answers
*which version* — never the module's `__version__`, which is hand-maintained and
drifts (`laptrack` ships `__version__ = "0.17.0"` in its 0.17.1 release). A
declared range is bounded at both ends, so an install *newer* than it is unmet
too, and the fix is not another install: say so and offer the degraded path.

## When something is missing

Diagnose, tell the user, let them choose. Installing, seeding and restarting are
all theirs to authorize — but a named gap usually beats abandoning the doc.

**A kernel plugin — three causes, in this order:**
1. **This install predates it.** Seeding cannot conjure a plugin that does not
   ship in the installed version; point at upgrading biopb (rerun the
   installer), and stop there.
2. **Else check the file:** `ls ~/.config/biopb/kernel/<name>.py`. Present, but
   absent from the report → it **failed to load**. The traceback is in the
   session log (`log_file:` under `## System`). Show the user the error; this is
   a bug to report, not something to retry.
3. **Absent → never seeded.** `biopb-mcp-seed-plugins` installs the built-ins,
   then the kernel must restart to load them. **Ask before restarting** — it
   takes the namespace and every layer with it.

**A package — offer three options and let the user pick:**
1. **They install it** — quote the exact command `## Versions` prints (it names
   *this* interpreter). Never a bare `pip install`: it targets whatever env
   their shell has active, which can succeed while the import here still fails.
2. **You install it for them** — same command, run from `execute_code` via
   `subprocess`, **only after they say yes**. Then `importlib.invalidate_caches()`
   and import again; if the module was already half-imported, `restart_kernel`
   (ask — layers are lost).
3. **The doc's degraded path**, if it names one. Often the right answer for a
   one-off run: nothing to undo, and it survives an upgrade.

Whichever of the first two they pick, if `## Versions` says the env is
uv-managed, say so and name `~/.config/biopb/extra-packages.txt`: the install
lands now but is gone at the next biopb upgrade unless the requirement is in
that file.

**The viewer.** `## Viewer` says whether there is a window on screen. *Headless*
means there is none this session; *`window: CLOSED`* means the user closed it,
and `restart_kernel` restores it (ask — layers are lost). Neither is a reason to
stop, and neither means the result cannot be seen: upload it and send a
[[web-viewer]] link, which does not depend on this session having a display at
all. What you lose is `take_screenshot` — unless your host gives you browser
automation, in which case you open that link and look at it yourself.

**The data plane.** `## Tensor Server` names the cause (not connected / auth /
still starting). Check `biopb control status` with the user, or point at
`$BIOPB_TENSOR_URL` if the data lives on a server the control does not own. Do
**not** proceed as if the catalog were empty — that reads to the user as "no
data" rather than "not connected".

**An op.** `## Ops` lists what the servers *do* offer, so say whether one covers
the same need — but **ask before substituting** an op the doc did not name,
since a different model is a different result. Otherwise the user adds a server
to `services.process_image_servers`.

## Related

- [[kernel]] — the namespace these resolve against, and how plugins get loaded.
- [[napari-viewer]] and [[web-viewer]] — the two display surfaces, either of
  which satisfies "somewhere to show the user an image".
