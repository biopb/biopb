# Fixtures — what a run is given, and what it has to recover

**Component:** `biopb-mcp` — `_tests/agentbench/_fixture.py` (the vocabulary),
`_tests/agentbench/test_fixture_protocol.py` (its own tests),
`_tests/agentbench/test_fixture_tree.py` (the `-m fixtures` check),
`biopb-mcp/tools/author_*_fixture.py` (authoring a curated case's data).
**Related:** [`skills.md`](skills.md) §10 — the benchmark written in this
vocabulary; `_tests/bench/README.md` — how to run it.

One runner puts a model in front of a real biopb session and scores what comes
back, whether the case is a claim about a skill or about a piece of work. Every
case hands a verifier a `Fixture` and an `Attempt` and reads back an `Outcome`,
and nothing in this layer knows what drift is or what a landmark is.

---

## 1. A case owns one fixture

**A case is non-decomposable.** Task, persona, fixture, verifier and tolerances
are one artifact, and where the pixels come from — a procedure or a file on
disk — is decided *when the case is written*, never resolved at run time.

**Substituting the data makes it a different experiment with the same name.**
The truth changes, the achievable accuracy changes, and the conclusion can
invert. That last one is measured rather than argued: the
`align-stack-by-features` procedural fixture rendered every object as an
identical isotropic Gaussian and ranked two method families in the **opposite
order** from real tissue — 1 cold run in 9 chose descriptor matching on the
synthetic content, against 2 in 3 on real sections. Its tolerances were
calibrated to 3.0 px where the reference scored 0.56 px; the same reference on
real sections scores 3.69 px and fails that gate. A harness that swapped one for
the other per machine would publish both results as one number.

So there is no fallback, no precedence and no substitution. `Case.fixture` is a
single `FixtureSpec`. On a machine that cannot produce it, the case does not
run and says why — the same discipline as a missing API key, and never a pass.

`$BIOPB_FIXTURES` is a **root path, not a policy switch**: it says where a
curated case finds its data, never which fixture a case runs.

**A skill worth covering both ways is two cases**, each with its own `case_id`,
its own tolerances and its own expectations. That is what makes
`(namespace, case_id)` load-bearing rather than decorative — it keys the
artifacts, the reports and the tree.

## 2. The vocabulary

```python
Fixture(provenance, data, truth, tolerance, about, citation, kind, skill_id, case_id)
Attempt(subject, arrays, notes)
Metric(name, value, limit, unit, unavailable)
Outcome(fixture, attempt, metrics, detail)
```

`data` and `truth` are **separate mappings** rather than one object with
optional fields, which makes the leak the whole layer depends on not happening —
a truth key appearing in `data` — a thing a test can assert about any fixture
without knowing the skill.

`kind`, `skill_id` and `case_id` are **stamped by the spec** from the case that
owns them; anything a builder sets is overwritten. A second place to name a case
is a second place for the name to drift, with the report saying one thing and
the artifact directory another.

**Truth is data, not a formula.** A synthetic fixture knows the answer because
it constructed it; a curated one knows whatever a human annotated. Both hand the
verifier a mapping, so one verifier serves either kind.

**A metric that cannot be computed is `unavailable`, never passing.**
`Metric.value is None` says *this run, against this fixture's truth, does not
support this measurement*, and `Outcome.passed` is false when **nothing** was
scored. That covers both halves of one problem: a curated fixture whose truth
does not support a measurement, and an agent that left nothing behind or bound a
name to the wrong shape. Without it every silent-responder run would read as a
clean run.

Verifiers read a run's leavings through `read_array` / `read_scalar`, which
return `(value, why not)` rather than raising. An agent binds a name to the
wrong thing about as often as it binds it to the right one, and crashing on that
is worse than scoring it.

## 3. Where the pixels come from

```python
class FixtureSpec(Protocol):
    kind: Kind                                                    # synthetic | curated
    def available(self, skill_id, case_id) -> tuple[bool, str]: ...
    def build(self, skill_id, case_id) -> Fixture: ...
```

Two implementations, with **no ordering between them**:

| | `Procedural(builder)` | `OnDisk(tolerance=...)` |
|---|---|---|
| `kind` | `synthetic` | `curated` |
| Pixels | generated from a seed at run time | read from `$BIOPB_FIXTURES` |
| Truth | exact by construction | whatever was annotated or applied |
| Availability | always | the tree, the manifest entry, and a reader for every file |

`build` takes the owning case's identity rather than the case itself, which
keeps this module free of any import from the engines above it.

Where a second derivation is cheap, a procedural builder asserts the two agree
before handing the fixture over: `segmentation-qc-metrics` checks its closed-form
TP/FP/FN against what `plugin:segmentation_qc` actually matches, and
`calibrated-measurements` checks its voxel-count truth against
`regionprops(spacing=)`. A fixture whose truth is wrong makes every run scored
against it meaningless, so that fails at build time rather than reporting a quiet
zero later.

`OnDisk`'s `tolerance` lives on the **spec**, not in the tree: limits belong
beside the verifier that reports against them, where a machine's copy of the
data cannot quietly re-tune what counts as a pass.

## 4. What real data costs is truth

A curated movie can carry a trajectory someone measured off a bead, but not the
un-drifted reference image — no such acquisition exists. Two ways to close that,
and the choice is per case:

- **The data already carries truth** (a segmentation annotation, a measured
  trajectory). The case uses it directly and needs no authoring step.
- **Perturb once, at authoring time.** A tool under `biopb-mcp/tools/` takes a
  real acquisition, applies a known transformation, and writes the result into
  the tree with the transformation recorded in the manifest's `provenance`.
  `tools/author_align_channels_fixture.py` is the worked example: a real confocal
  field, a fixed affine-plus-sinusoid warp, landmarks detected and sampled from
  the real nuclei, and probe points whose true correspondence is known because
  the warp is invertible.

**The run only ever reads.** A benchmark that re-derives its data every run
cannot notice that its data changed, and a transformation applied *during* a run
is a knob someone can turn between two results that later get compared. So the
perturbation is a build step whose output is reviewed data, `kind` stays
`curated`, and there is no third provenance literal to reason about.

Nothing here validates the **science** of an annotation. Whether someone's
fiducial trajectory is right is a review question — the review a synthetic seed
does not need — and that asymmetry belongs in the manifest's `provenance`.

## 5. Handles, not arrays

A curated case's truth can be a label volume the size of the acquisition, and
repacking that into an npz to hand a verifier one array is both wasteful and
lossy. So a fixture's values may be **refs**:

```python
class ArrayRef(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype
    def __array__(self, dtype=None, copy=None) -> np.ndarray: ...
    def dask(self, chunks="auto") -> "dask.array.Array": ...
```

Every verifier already reads through `np.asarray`, so deferring the read costs
no verifier a line.

- `NpzRef(path, key)` — one array inside an archive. Shape and dtype come from
  the member's own header, so checking a file against the manifest is a seek.
- `FileRef(path)` — a whole file as one array, over a small reader registry:
  `.npy` (memory-mapped, so a truth volume larger than the test process is
  addressable rather than resident), `.tif`/`.tiff`, `.nii`/`.nii.gz`. The
  non-mmap readers defer the read but not the residency.

The registry is small and explicit: a fixture tree is reviewed data, so the
formats it may arrive in are a decision rather than whatever the machine happens
to import. `ref_missing()` answers "can this machine open this file" without
touching it, so a tree half of whose formats are unreadable here reports as an
availability fact rather than crashing halfway through a paid run.

`.npz` is deliberately **not** in the registry — an archive holds many arrays, so
naming the file does not name an array. That is why a case's layout maps *key →
filename* rather than the reverse: `{"stack": "arrays.npz"}` resolves to the
member named `stack`, so one archive backs several keys with no new syntax.

## 6. The tree

```
$BIOPB_FIXTURES/
├── manifest.json                       # what this machine has, and whose it is
└── <namespace>/<case_id>/
    └── case.json                       # the data/truth partition, and nothing else
```

`<namespace>` is the **skill id** for a case that names one, and the literal
`tasks` for a case that does not — `Case.namespace`, which is also the first
half of the case's label and of its artifact path. The case's identity locates
its data, so there is nothing to select and nothing to sort.

```json
{
  "about": "one ACDC patient, slices re-placed independently",
  "data":  {"stack":  "patient101_frame01.nii.gz"},
  "truth": {"labels": "patient101_frame01_gt.nii.gz"}
}
```

A key in both mappings is a hard error: a truth the run can see is not a truth.

Everything *about* the data lives in the root manifest instead, so a curated case
has exactly one place recording what was acquired:

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

The manifest's key is still `skill`, which is the namespace under the layout
above. `citation` is **required** for a curated fixture and is read from here
rather than from whoever remembers — ACDC ships a `MANDATORY_CITATION.md`, and
that obligation belongs to the harness. It is carried into the report and the
artifact directory.

**A fixture on disk with no manifest entry does not run.** An acquisition nobody
wrote down is not one a benchmark should score, and this is how an unreviewed
copy would otherwise slip into a run.

## 7. Checking the tree, split by cost

**Shape and dtype, in-band, at build time** (`_agrees_with_manifest`). The
manifest already records them, reading them is a header read rather than a pass
over the bytes, and a mismatch means the file under this path is not the file the
case was written against — the one remaining way a case name could quietly
denote two experiments.

**The SHA, out-of-band**, behind the `fixtures` marker:

```sh
uv run --no-sync pytest -m fixtures biopb-mcp/src/biopb_mcp/_tests/agentbench
```

Run after syncing a tree, or when a result looks wrong. It walks the manifest,
hashes each file, and reports drift — including a fixture present on disk but
absent from the manifest. Hashing multi-gigabyte volumes on a mount would cost
more than the run it guards, and would do it again on every sample, so it is never
part of a benchmark run. Everything skips on a machine with no tree.

## 8. Presentation — how a fixture reaches the agent

A fixture is not handed to the agent; it is loaded onto a viewer the agent
drives, and *how* is part of the case:

| `presentation` | what the agent finds | cost |
|---|---|---|
| `array` | in-memory numpy on a napari layer, `client is None` | none |
| `tensor` | `client` non-None, `viewer.add_tensor(array_id)`: lazy, pyramided | a data plane for the run |

**Neither is a default and neither is a fallback for the other.** They are two
legitimate environments, and the right one is whichever the skill was written
against — a viewer layer holding a plain array is a real thing an agent meets,
not a concession. There is deliberately no third, mmap-backed "lazy but no
server" presentation: the lazy path a skill is written for is `client.get_tensor`
over Flight, and a local mmap wearing dask's type would measure the skill off its
own route while costing a second loading mechanism.

`Layer` is `bench/_case.py`, and `kind` is what decides which `viewer.add_*`
call the harness makes. `points` is not cosmetic among them: it is how a
person's clicked correspondences actually reach napari, and a landmark task
handed a raw `(N, 2)` array would be testing a different route.

```python
Layer(name, key, kind="image", presentation="array", chunks=None, dim_labels=None)
```

`chunks` is explicit rather than left to the uploader's default, because where
laziness is the point the chunking *is* the thing under test: a route that only
fails at a chunk boundary is not exercised by a single-chunk array.

### The plane runs for the whole benchmark

Only `tensor` cases need one, so it is conditional — no selected case asks,
nothing starts. When one does, the lifetime is the **whole run**: one server with
its own temp data dir, started once, serving every case and every sample.
`$BIOPB_TENSOR_URL` is exported into the session child's environment and
inherited down to the kernel — but only for a case that actually uploaded
something. An `array` case still gets the unreachable address and a `None`
client, in the same invocation, after the plane is up.

- **Upload is paid once.** A case runs a session per sample and an invocation
  runs many cases; a per-session plane would upload the same volumes again every
  time, and these are the large fixtures.
- **It is the production shape.** A durable plane that sessions come and go
  against is the runtime tree in `biopb-mcp/CLAUDE.md`.
- **The developer's own catalog is neither read nor written**, because the
  server gets a temp data dir of its own.

It runs `--writable`, which is required anyway: a read-only plane would fail
every step that uploads a result rather than measure it. If it cannot start — or
`biopb_tensor_server` is not installed, which is normal, since biopb-mcp cannot
depend on a package that is never on PyPI — `tensor` cases skip with that reason
and `array` cases are unaffected.

**Only `Fixture.data` is uploaded.** `truth` never reaches the plane, so the leak
the whole layer depends on not happening is prevented structurally.

### The ids arrive in the namespace

An uploaded source is deliberately not synced to the catalog, so `query_sources()`
cannot find a fixture and the agent cannot browse to it; nor can a task prompt
name the id, since it is minted at run time. The harness binds
`fixture_tensors = {layer name: array_id}` in the kernel namespace as setup, and
a case presenting `tensor` says so in its prompt — the same kind of harness
convention as the `collect` names, and asserted the same way.

### Sessions are isolated by the id, not by cleanup

The plane outlives the individual session, so in principle one session's uploads are
visible to the next. The isolation is in the id:

```python
source_id = f"cache_{sha256(source_name).hexdigest()[:12]}"   # upload_manager.py
```

**The id an agent sees is a one-way hash of a name it is never told.** The name
appears in no descriptor, no layer and no catalog entry, so the harness uploads
each fixture under a per-run random name and an agent holding the id cannot
construct the name that would let it replace the data. Collision is
unrepresentable rather than unlikely, which matters because the natural name for
a skill's own output is derived from the task.

That is an argument about a surface running arbitrary Python, so it is also
checked: a corner of each fixture is fingerprinted at upload and again after each
sample, and a change flags the row `fixture-overwritten`, voiding the number rather
than qualifying it. Prevention is structural; the check is there so that
defeating it cannot be quiet.

An agent's own uploads persist for the rest of the run. That is affordable —
they are bounded by the chunk cache, and the plane's whole state is a temp
directory discarded at teardown.

## 9. Coverage is a warning, per skill

A skill declares what it touches in `checklist:`, so presentation coverage can be
computed rather than judged — but the unit is **the skill, not the case**. A case
presenting `array` for a skill that declares `dask` is not wrong; it tests the
in-memory branch, which is a real branch. What it is, is *incomplete*, and since
a skill may have several cases the fix is another case rather than a correction
to this one.

```
drift-correction declares ['dask', 'tensor'], but every case presents `array`,
so every run has `client is None` — neither the lazy read path nor any
step that uploads a result has been benchmarked
```

**Never a failure.** A gate here would punish an honest partial benchmark exactly
as hard as a wrong one. The ledger belongs beside `NOT_BENCHMARKED`, which
records "this skill is outside the layer, and why"; this records "this skill is
*partly* inside it, and which part".

## 10. Artifacts

Every case emits a number *and* an artifact: the number says what happened, the
artifact explains it, and in an imaging project the second is usually what a
person needs. Imaging failures are recognisable in a second by eye and awkward
to characterise in an assertion.

`write_report(outcome, root)` writes `summary.json` under
`root/<subject>/`, where *root* is the case's own directory keyed on
`(namespace, case_id)` — so a second case for one skill writes beside the first
rather than over it. `save_png` is best-effort: an artifact explains a failure,
it never causes one, so a missing imageio plugin is swallowed to a
`.png.error` file.

Pass a shared `vmax` for any pair of images meant to be compared. Scaling each
independently is the trap — a corrected-minus-raw difference that is near zero
everywhere gets stretched to full range and looks exactly like the uncorrected
one, so the artifact would be most misleading precisely where it is most wanted.

`artifact_root()` honours `$BIOPB_OUTCOME_DIR` and otherwise lands inside the
checkout, beside the source: these are meant to be opened and paged through, not
hunted for in a temp dir. The checkout is **searched for by marker**, never
counted to by parent depth — a depth is right until the module moves, and
nothing asserts on it.
