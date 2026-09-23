# Fixtures — what a run is given, and what it has to recover

**Component:** `biopb-mcp` — `_tests/agentbench/_fixture.py` (the vocabulary),
`_tests/agentbench/test_fixture_protocol.py` (its own tests),
`_tests/agentbench/test_fixture_tree.py` (the `-m fixtures` check),
`biopb-mcp/tools/author_*_fixture.py` (authoring a curated case's data).
**Related:** [`../src/biopb_mcp/_tests/bench/README.md`](../src/biopb_mcp/_tests/bench/README.md) — the benchmark written in this
vocabulary and how to run it.

One runner puts a model in front of a real biopb session and scores what
comes back, whether the case is a claim about a skill or about a piece of
work. Every case hands a verifier a `Fixture` and an `Attempt` and reads
back an `Outcome`; nothing in this layer knows what drift is or what a
landmark is.

## A case owns one fixture

**A case is non-decomposable.** Task, persona, fixture, verifier and
tolerances are one artifact, and where the pixels come from — a procedure
or a file on disk — is decided when the case is written, never resolved at
run time. `Case.fixture` is a single `FixtureSpec`: no fallback, no
precedence, no substitution. On a machine that cannot produce it, the case
does not run and says why — the same discipline as a missing API key, never
a pass. `$BIOPB_FIXTURES` is a **root path, not a policy switch**: it says
where a curated case finds its data, never which fixture a case runs.

Substituting the data changes the truth, the achievable accuracy, and can
invert the conclusion: `align-stack-by-features`'s procedural fixture
(every object an identical isotropic Gaussian) ranked two method families
in the **opposite order** from real tissue — 1 cold run in 9 chose
descriptor matching on synthetic content, against 2 in 3 on real sections;
its tolerance was calibrated to 3.0 px where the reference scores 0.56 px
synthetic and 3.69 px real. A skill worth covering both ways is therefore
**two cases**, each its own `case_id`, tolerances and expectations — what
makes `(namespace, case_id)` key the artifacts, the reports and the tree.

## The vocabulary

```python
Fixture(provenance, data, truth, tolerance, about, citation, kind, skill_id, case_id)
Attempt(subject, arrays, notes)
Metric(name, value, limit, unit, unavailable)
Outcome(fixture, attempt, metrics, detail)
```

`data` and `truth` are **separate mappings**, not one object with optional
fields — a truth key appearing in `data` is the one leak the whole layer
depends on not happening, and a test can assert it without knowing the
skill. `kind`, `skill_id` and `case_id` are **stamped by the spec** from the
owning case; anything a builder sets is overwritten, so there is exactly
one place a case's name can drift from.

**Truth is data, not a formula.** A synthetic fixture knows the answer
because it constructed it; a curated one knows whatever a human annotated.
Both hand the verifier a mapping, so one verifier serves either kind.

**A metric that cannot be computed is `unavailable`, never passing.**
`Metric.value is None` means *this run, against this fixture's truth, does
not support this measurement*, and `Outcome.passed` is false when nothing
was scored — covering both a curated fixture whose truth doesn't support a
measurement and an agent that left nothing behind or bound a name to the
wrong shape. Verifiers read a run's leavings through `read_array` /
`read_scalar`, which return `(value, why not)` rather than raising, since an
agent binds a name to the wrong thing about as often as the right one.

## Where the pixels come from

```python
class FixtureSpec(Protocol):
    kind: Kind                                                    # synthetic | curated
    def available(self, skill_id, case_id) -> tuple[bool, str]: ...
    def build(self, skill_id, case_id) -> Fixture: ...
```

Two implementations, no ordering between them:

| | `Procedural(builder)` | `OnDisk(tolerance=...)` |
|---|---|---|
| `kind` | `synthetic` | `curated` |
| Pixels | generated from a seed at run time | read from `$BIOPB_FIXTURES` |
| Truth | exact by construction | whatever was annotated or applied |
| Availability | always | the tree, the manifest entry, and a reader for every file |

`build` takes the owning case's identity rather than the case itself, so
this module imports nothing from the engines above it. Where a second
derivation is cheap, a procedural builder asserts the two agree before
handing the fixture over (`segmentation-qc-metrics` against
`plugin:segmentation_qc`, `calibrated-measurements` against
`regionprops(spacing=)`) — a fixture whose truth is wrong makes every run
scored against it meaningless, so that fails at build time. `OnDisk`'s
`tolerance` lives on the **spec**, not in the tree, so a machine's copy of
the data cannot re-tune what counts as a pass.

## What real data costs is truth

A curated movie can carry a trajectory someone measured off a bead, but not
the un-drifted reference image — no such acquisition exists. Two ways to
close that, per case: the data already carries truth (a segmentation
annotation, a measured trajectory) and the case uses it directly; or a tool
under `biopb-mcp/tools/` perturbs a real acquisition once, at authoring
time, recording the transformation in the manifest's `provenance`
(`tools/author_align_channels_fixture.py`: a real confocal field, a fixed
affine-plus-sinusoid warp, landmarks sampled from the real nuclei, probe
points whose correspondence is known because the warp is invertible).

**The run only ever reads.** The perturbation is a build step whose output
is reviewed data; `kind` stays `curated`, and there is no third provenance
literal. Nothing here validates the **science** of an annotation — that
review belongs in the manifest's `provenance`, since a synthetic seed
doesn't need it.

## Handles, not arrays

A curated case's truth can be a label volume the size of the acquisition,
so a fixture's values may be **refs** rather than repacked arrays:

```python
class ArrayRef(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype
    def __array__(self, dtype=None, copy=None) -> np.ndarray: ...
    def dask(self, chunks="auto") -> "dask.array.Array": ...
```

Every verifier already reads through `np.asarray`, so deferring the read
costs no verifier a line.

- `NpzRef(path, key)` — one array inside an archive; shape and dtype come
  from the member's own header.
- `FileRef(path)` — a whole file as one array, over a small reader
  registry: `.npy` (memory-mapped, so a truth volume larger than the test
  process is addressable), `.tif`/`.tiff`, `.nii`/`.nii.gz`. The non-mmap
  readers defer the read but not the residency.

The registry is small and explicit — a fixture tree is reviewed data, so
its formats are a decision, not whatever the machine happens to import.
`ref_missing()` answers "can this machine open this file" without touching
it, so an unreadable format reports as an availability fact rather than a
crash mid-run. `.npz` is deliberately **not** in the registry, since an
archive holds many arrays: a case's layout maps *key → filename*
(`{"stack": "arrays.npz"}` resolves to the member named `stack`), so one
archive backs several keys with no new syntax.

## The tree

```
$BIOPB_FIXTURES/
├── manifest.json                       # what this machine has, and whose it is
└── <namespace>/<case_id>/
    └── case.json                       # the data/truth partition, and nothing else
```

`<namespace>` is the skill id for a case that names one, and the literal
`tasks` for one that doesn't — `Case.namespace`, also the first half of the
case's label and artifact path.

```json
{
  "about": "one ACDC patient, slices re-placed independently",
  "data":  {"stack":  "patient101_frame01.nii.gz"},
  "truth": {"labels": "patient101_frame01_gt.nii.gz"}
}
```

A key in both mappings is a hard error. Everything *about* the data lives
in the root manifest, so a curated case has exactly one place recording
what was acquired:

```json
{
  "version": 1,
  "fixtures": [
    {
      "skill": "tasks",
      "case_id": "align-channels-from-landmarks",
      "provenance": "channel 3 warped by a fixed affine + sinusoid; 18 landmarks …",
      "citation": "UConn Health, Yu lab -- 4-channel confocal, 2026-07-16",
      "files": {
        "data.npz": {
          "sha256": "e3b0c442…",
          "bytes": 7077888,
          "arrays": {"moving": {"shape": [960, 960], "dtype": "float32"}}
        }
      }
    }
  ]
}
```

The manifest's key is still `skill`, the namespace under the layout above.
`citation` is **required** for a curated fixture, carried into the report
and the artifact directory rather than left to whoever remembers (ACDC
ships a `MANDATORY_CITATION.md`). **A fixture on disk with no manifest
entry does not run** — an acquisition nobody wrote down is not one a
benchmark should score.

## Checking the tree, split by cost

**Shape and dtype, in-band, at build time** (`_agrees_with_manifest`): a
header read, not a pass over the bytes, and a mismatch means the file under
this path is not the file the case was written against.

**The SHA, out-of-band**, behind the `fixtures` marker:

```sh
uv run --no-sync pytest -m fixtures biopb-mcp/src/biopb_mcp/_tests/agentbench
```

Run after syncing a tree, or when a result looks wrong. It walks the
manifest, hashes each file, and reports drift — including a fixture
present on disk but absent from the manifest. Never part of a benchmark run
(hashing multi-gigabyte volumes on a mount costs more than the run it
guards, on every sample). Everything skips on a machine with no tree.

## Presentation — how a fixture reaches the agent

A fixture is not handed to the agent; it is loaded onto a viewer the agent
drives, and *how* is part of the case:

| `presentation` | what the agent finds | cost |
|---|---|---|
| `array` | in-memory numpy on a napari layer, `client is None` | none |
| `tensor` | `client` non-None, `viewer.add_tensor(array_id)`: lazy, pyramided | a data plane for the run |

**Neither is a default or a fallback for the other** — the right one is
whichever the skill was written against. There is deliberately no third,
mmap-backed "lazy but no server" presentation: a local mmap wearing dask's
type would measure the skill off its own route while costing a second
loading mechanism.

```python
Layer(name, key, kind="image", presentation="array", chunks=None, dim_labels=None)
```

`kind` decides which `viewer.add_*` call the harness makes — `points` is
how a person's clicked correspondences actually reach napari, not
cosmetic. `chunks` is explicit rather than left to the uploader's default,
because where laziness is the point the chunking *is* the thing under
test.

### The plane runs for the whole benchmark

Only `tensor` cases need one, so it's conditional — no selected case asks,
nothing starts. When one does, the lifetime is the whole run: one server
with its own temp data dir, started once, serving every case and sample.
`$BIOPB_TENSOR_URL` is exported into the session child's environment and
inherited down to the kernel, but only for a case that actually uploaded
something; an `array` case still gets the unreachable address and a `None`
client. Upload is paid once because a case runs a session per sample and an
invocation runs many cases; a per-session plane would re-upload the same
(often large) fixtures every time. It runs `--writable`, required anyway
since a read-only plane would fail every step that uploads a result. If it
cannot start — or `biopb_tensor_server` isn't installed, which is normal —
`tensor` cases skip with that reason and `array` cases are unaffected. The
developer's own catalog is neither read nor written, since the server gets
a temp data dir of its own. **Only `Fixture.data` is uploaded** — `truth`
never reaches the plane.

### The ids arrive in the namespace, sessions isolated by the id

An uploaded source is not synced to the catalog: `query_sources()` cannot
find it and a task prompt cannot name its id, since it's minted at run
time. The harness binds `fixture_tensors = {layer name: array_id}` in the
kernel namespace as setup, and a `tensor`-presenting case says so in its
prompt.

The plane outlives an individual session, so the isolation is in the id:

```python
source_id = f"cache_{sha256(source_name).hexdigest()[:12]}"   # upload_manager.py
```

The id an agent sees is a one-way hash of a name it is never told — the
harness uploads each fixture under a per-run random name, so an agent
holding the id cannot construct the name that would let it replace the
data. That is also checked, not just assumed: a corner of each fixture is
fingerprinted at upload and again after each sample, and a change flags the
row `fixture-overwritten` rather than silently qualifying the number. An
agent's own uploads persist for the rest of the run — affordable, since
they're bounded by the chunk cache and the plane's whole state is a temp
directory discarded at teardown.

## Coverage is a warning, per skill

A skill declares what it touches in `checklist:`, so presentation coverage
is computed rather than judged — but the unit is the skill, not the case. A
case presenting `array` for a skill that also declares `dask` isn't wrong,
it tests a real branch; it's *incomplete*, and the fix is another case:

```
drift-correction declares ['dask', 'tensor'], but every case presents `array`,
so every run has `client is None` — neither the lazy read path nor any
step that uploads a result has been benchmarked
```

**Never a failure** — a gate here would punish an honest partial benchmark
as hard as a wrong one.

## Artifacts

Every case emits a number *and* an artifact: the number says what happened,
the artifact explains it, and in an imaging project the second is usually
what a person needs. `write_report(outcome, root)` writes `summary.json`
under `root/<subject>/`, keyed on `(namespace, case_id)`, so a second case
for one skill writes beside the first. `save_png` is best-effort — a
missing imageio plugin is swallowed to a `.png.error` file rather than
failing the run.

Pass a shared `vmax` for any pair of images meant to be compared — scaling
each independently is the trap, since a corrected-minus-raw difference near
zero everywhere gets stretched to full range and looks exactly like the
uncorrected one.

`artifact_root()` honours `$BIOPB_OUTCOME_DIR` and otherwise lands inside
the checkout, beside the source, searched for **by marker** rather than
counted to by parent depth.
