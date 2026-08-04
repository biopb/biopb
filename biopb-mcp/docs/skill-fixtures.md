# Fixtures for the interaction layer — a design

Status: **implemented**. Supersedes PR #704, and corrects the framing
`biopb-mcp/docs/skill-testing.md` §5d used to carry. The table below is what
each section became; the deviations after it are where building the thing
changed the design.

| Section | State |
|---|---|
| The principle; what it deletes | **landed** — `curated_for()` and its precedence are gone |
| The shape (`FixtureSpec`, `Procedural`, `OnDisk`) | **landed** |
| Data and truth as references (`ArrayRef`) | **landed**, as `NpzRef` + `FileRef` |
| Identity — `(skill, case_id)` everywhere | **landed** |
| The tree carries a manifest; out-of-band hashing | **landed** (`-m fixtures`) |
| Presentation — **`array`/`tensor`**, `dask` dropped | **landed** |
| The run-scoped data plane | **landed** |
| Per-skill presentation coverage warning | **landed**, and warns for all 5 benchmarked skills today |

Four deviations from what is written below, each argued where it applies:

- **`MemmapRef` was not built.** `FileRef` memory-maps `.npy` itself, so a
  separate class would have been the same object under a second name.
- **`.npz` is not in `FileRef`'s reader registry.** An archive holds many
  arrays, so naming the file does not name an array; a `.npz` in a case's
  layout resolves through `NpzRef` on the key that referenced it, which is why
  the layout below maps *key → filename* rather than the reverse.
- **The `dask` presentation was dropped**, leaving `array` and `tensor`. A local
  mmap wearing dask's type is not an environment any session has either, so it
  would have measured a lazy skill off its own path at the cost of a second
  loading mechanism.
- **Arm isolation is not the scheme split described below**, which rested on a
  cleanup call that does not exist. It is the fixture id being a one-way hash of
  a per-run secret name, plus a fingerprint check that flags any change.

## The principle

**A case is non-decomposable.** Task, persona, fixture, verifier and tolerances
are one artifact, and where the fixture's pixels come from — a procedure or a file
on disk — is decided *when the case is written*, not resolved at run time.

Today's model contradicts this. `Case.build_fixture()` is
`curated_for(skill) or build()`: every case is silently overridable by whatever
happens to sit under `$BIOPB_SKILL_FIXTURES` on the machine running it. That is
wrong, and not merely untidy — **substituting the data makes it a different
experiment with the same name.** The truth changes, the achievable accuracy
changes, and the conclusion can invert.

That last risk is measured rather than hypothetical. The `align-stack-by-features`
procedural fixture rendered every object as an identical isotropic Gaussian, and
ranked two method families in the *opposite order* from real tissue: 1 cold arm in
9 chose descriptor matching on the synthetic content against 2 in 3 on real
sections. Its tolerances were calibrated to 3.0 px where the reference scored
0.56 px; the same reference on real sections scores 3.69 px and fails that gate. A
harness that swapped one for the other at run time would publish both results as
one number.

So: **no substitution, no fallback, no precedence.** A case has one fixture. On a
machine that cannot produce it, the case does not run, and says so.

## What the principle deletes

- `curated_for()`'s precedence, entirely.
- `$BIOPB_SKILL_FIXTURES` as a policy switch. It is demoted to what it should
  always have been: **the root path under which on-disk fixtures live**. Setting it
  changes where data is found, never which experiment runs.
- The "tolerances do not transfer between synthetic and curated" problem, which was
  an artifact of substitution. Limits are calibrated once, against the one fixture
  the case owns.
- `no_synthetic_reason` from PR #704. A case does not owe an explanation for not
  having a second, hypothetical fixture; it has the fixture its author chose.

**Want a skill covered both ways? That is two cases**, each non-decomposable, each
with its own id, tolerances and expectations — which is also what makes the
`(skill, case_id)` keying below load-bearing rather than hypothetical.

## The shape

```python
@dataclass(frozen=True)
class Case:
    skill: str
    case_id: str                 # identity: names the run, the artifacts, the data
    fixture: FixtureSpec         # exactly one, fixed at authoring time
    task: str
    persona: Persona
    layers: Sequence[Layer]
    collect: Mapping[str, str]
    score: Callable[[Fixture, Attempt], Outcome]
    tolerance: Mapping[str, float]
    ...
```

```python
@runtime_checkable
class FixtureSpec(Protocol):
    """Where this case's pixels come from. One per case, not a preference."""

    kind: Kind                                    # "synthetic" | "curated"

    def available(self) -> tuple[bool, str]:      # (runnable here, why not)
        ...

    def build(self, case: Case) -> Fixture:
        ...
```

Two implementations, and no ordering between them:

```python
@dataclass(frozen=True)
class Procedural:
    """Generated from a seed. Always available; truth exact by construction."""
    builder: Callable[[], Fixture]
    kind: Kind = "synthetic"

@dataclass(frozen=True)
class OnDisk:
    """Real data the case was written against.

    Rooted at $BIOPB_SKILL_FIXTURES/<skill>/<case_id>/ — the case's own identity
    locates its data, so there is nothing to select and nothing to sort.
    """
    kind: Kind = "curated"
```

`OnDisk()` needs no arguments in the ordinary case: the case already knows what it
is called, and its data lives under that name. The alphabetical-first-directory
selection in today's `curated_for()` disappears along with the choice it was
making.

A case whose `fixture.available()` is false becomes an `unavailable()` skip naming
the missing path — the same discipline as a missing API key, and never a pass.

## Data and truth as references, not arrays

Independent of the framing above, and still needed: a verifier must be able to
reach real ground truth that is not already packed into an npz, and a case must be
able to carry truth larger than the test process.

```python
class ArrayRef(Protocol):
    """A handle to an array that has not been read yet."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    def __array__(self, dtype=None) -> np.ndarray: ...     # np.asarray(ref) works
    def dask(self, chunks="auto") -> "dask.array.Array": ...
```

with `NpzRef(path, key)`, `FileRef(path)` over a small reader registry
(`.npy/.npz/.tif/.nii/.nii.gz`), and `MemmapRef(path)`. `Fixture.data` and
`Fixture.truth` values become `ArrayLike | ArrayRef`.

**Every existing verifier keeps working unchanged**, because they already read
through `np.asarray(...)`; what changes is that reading defers to the point of use.
So ground truth can be the `_gt.nii.gz` sitting beside its image, or a label volume
the size of the acquisition, without repacking and without residency.

An on-disk case's `case.json` then describes only the *data layout* — not limits,
which belong to the case module beside the verifier that reports against them,
and not provenance or citation, which belong to the manifest below so that a
curated case has exactly one place recording what was acquired:

```json
{
  "about": "one ACDC patient, slices re-placed independently",
  "data":  {"stack":  "patient101_frame01.nii.gz"},
  "truth": {"labels": "patient101_frame01_gt.nii.gz"}
}
```

The mapping is *key → filename* rather than filename → keys because a `.npz`
carries many arrays: `{"stack": "arrays.npz"}` resolves to the member named
`stack`, so one archive can back several keys with no new syntax.

The data/truth partition rule — a key in both is an error, because a truth the run
can see is not a truth — is unchanged and still checkable on keys alone.

## Two presentations, because a faked-lazy one measures nothing

The largest gap, and it is not in the fixture but in the path from fixture to
agent:

```python
# _benchmark.py, loading a case's layers
session.put_array("_fixture_array", np.asarray(fixture.data[layer.key]))
```

`np.asarray` materialises, `put_array` round-trips through `np.save`/`np.load`, and
the agent finds a plain in-memory numpy array with `client is None` — which every
task prompt states aloud ("There is no tensor server in this session"). The
benchmark presents a data environment **no production session has**, so the lazy
half of the catalogue cannot be measured on the path it was written for.

```python
@dataclass(frozen=True)
class Layer:
    name: str
    key: str
    kind: str = "image"            # image | labels
    presentation: str = "array"    # array | tensor
    chunks: tuple[int, ...] | None = None
    dim_labels: tuple[str, ...] | None = None
```

| presentation | what the agent finds | cost |
|---|---|---|
| `array` | in-memory numpy on a viewer layer, `client is None` | none |
| `tensor` | `client` non-None, `viewer.add_tensor(array_id)`: lazy, pyramided | a data plane for the run (below) |

**An earlier draft of this document had a third, `dask`** — a `da.from_array`
over a memory-mapped file, lazy but with no server. It is not here, and the
reason generalises: **that would have been a data environment no production
session has either.** The lazy path a skill is actually written for is
`client.get_tensor`, which returns a dask array over Flight; a local mmap wearing
the same type would still have measured the skill off its own route, while
costing a second loading mechanism to build and maintain. Two presentations that
are each real beat three where the middle one is a prop.

**Neither is the default and neither is a fallback for the other.** They are two
legitimate environments, and the right one is whichever the skill was written
against. A skill about in-memory numpy is correctly tested with `array`; a viewer
layer holding a plain array is a real thing an agent meets, not a concession.
Presentation is part of the case for the same reason the fixture is — changing it
changes what is being measured.

`chunks` is explicit rather than left to the uploader's default because where
laziness is the point the chunking *is* the thing under test: a route that only
fails at a chunk boundary — the out-of-core measurement route, `chunked_label` —
is not exercised by a single-chunk array. The upload takes the case's chunk shape
verbatim, so this is a real control rather than a hint.

### The ids arrive in the namespace, because the catalog will not carry them

An uploaded source is deliberately **not synced to the catalog**
(`upload_manager.py`), so `query_sources()` cannot find a fixture and the agent
cannot browse to it. Nor can a task prompt name the id: it is minted at run time.
So the harness binds `fixture_tensors = {layer name: array_id}` in the kernel
namespace as setup, and a case presenting `tensor` must say so in its prompt —
the same kind of harness convention as the `collect` names, and asserted the same
way.

### The skill's own `checklist:` says what is left uncovered

A skill already declares what it touches, so coverage can be computed rather than
judged — but the unit is **the skill, not the case**. A case presenting `array`
for a skill that declares `dask` is not wrong; it tests the in-memory branch,
which is a real branch. What it is, is *incomplete*: the lazy branch has no case
yet, and since a skill may have several cases, the fix is another case rather
than a correction to this one.

So this reports a **coverage gap, and warns**:

```
drift-correction declares ['dask', 'tensor'], but every case presents `array`,
so every arm runs with `client is None` — neither the lazy read path nor any
step that uploads a result has been benchmarked
```

Never a failure. A gate here would demand cases be written before the design
that makes them possible has landed, and would punish an honest partial
benchmark exactly as hard as a wrong one. The ledger belongs beside
`NOT_BENCHMARKED`, which already records "this skill is outside the layer, and
why" — this records "this skill is *partly* inside it, and which part".

## When a plane is needed, it runs for the whole benchmark

Only cases presenting `tensor` need one, so the plane is **conditional** — no
selected case asks for it, nothing starts, and the suite behaves exactly as it
does today. When one does, the right lifetime is the **whole run**: one server,
started once, serving every case and every arm that wants it.

```
session-scoped fixture
  ├── start a tensor server on a free port, with its own temp data dir
  ├── upload every `tensor`-presented layer of every selected case  → array_ids
  ├── export $BIOPB_TENSOR_URL into each session child's environment
  │      (inherited by the shim → session → kernel; `_connection` documents this
  │       as the escape hatch "for a plane nothing supervises", so no control
  │       plane is needed and invariant I2 is untouched)
  └── teardown: stop the server, discard the store
```

Why run-scoped rather than per-case or per-arm:

- **Upload is paid once.** A case runs four arms; a per-arm plane would upload the
  same volume four times, and these fixtures are the large ones by construction.
- **It is the production shape.** A durable plane that sessions come and go against
  is exactly the runtime tree in `biopb-mcp/CLAUDE.md`; a plane per session is not
  a thing that happens in production.
- **Isolation stays intact** by giving the server its own temp data dir, the same
  discipline `_write_config` already applies to the config tree — the developer's
  own catalog is neither read nor written.

Only `Fixture.data` is uploaded. `truth` never reaches the plane, so the leak the
whole layer depends on not happening is prevented structurally rather than by
care.

### Isolation between arms comes from the id, not from cleanup

The plane outlives the individual arm, so in principle an agent's uploads would be
visible to the arms after it — an isolation property today's kernel-scoped model
gives for free.

**An earlier draft of this section was wrong about how to get it back.** It
proposed declaring fixtures as file-backed `sources:` so that agent writes, which
can only ever be `cache://`, were distinguishable by scheme, and then "dropping
the cache sources" between arms. Two facts kill that:

- **There is no API to drop a cache source.** `remove_source` raises unless the
  url starts with `dnd://` (`source_manager.py`), and cache adapters are expected
  to accumulate until the server stops (`cached_source.py`). The cleanup step the
  scheme split existed to enable does not exist.
- **Declaring them file-backed is not cheap either.** There is no npy/npz adapter,
  so a procedurally generated fixture would have to be written as zarr first — a
  second serialisation path, to enable a cleanup that cannot be performed.

The isolation is already there, in the id:

```python
source_id = f"cache_{sha256(source_name).hexdigest()[:12]}"   # upload_manager.py
```

**The id an agent sees is a one-way hash of a name it is never told.** The adapter
keeps the id and the url (`cache://<source_id>`); the name appears in no
descriptor, no layer and no catalog entry. So the harness uploads each fixture
under a **per-run random name**, and an agent holding the id cannot construct the
name that would let it replace the data. Accidental collision is impossible rather
than unlikely, which matters because the natural name for a skill's own output
(`drift-correction` step 7, `stitch-tiles` step 7) is derived from the task.

That is an argument about a surface that runs arbitrary Python, so it is also
checked: the harness fingerprints a corner of each fixture at upload and again
after every arm, and a change **flags the row** — `fixture-overwritten`, the same
shape as `read-harness-internals`, voiding the number rather than qualifying it,
and voiding every later arm too. Prevention is structural; the check is there so
that defeating it cannot be quiet.

What follows from having no per-arm sweep is that an agent's own uploads persist
for the rest of the run. That is affordable — they are bounded by the chunk cache,
and the plane's whole state is a temp directory discarded at teardown.

The plane must run **`--writable`** (the server defaults to read-only), which is
required anyway: a read-only plane would fail every skill step that uploads a
result rather than measure it.

If the plane cannot start — or `biopb_tensor_server` is not installed, which is
normal, since biopb-mcp cannot depend on a package that is never on PyPI — cases
presenting `tensor` skip through `unavailable()` with that reason, and `array`
cases are unaffected.

## Identity

`case_id` is promoted from "whatever the builder happened to set on the Fixture" to
part of the `Case`. Everything currently keyed on `skill` alone is keyed on
`(skill, case_id)`:

- `where_for(case)` → `interaction/<skill>/<case_id>`
- `test_benchmark._RUNS`, `test_cases._FIXTURES`
- a test asserting `(skill, case_id)` is unique

Today four structures assume one case per skill with nothing checking it, so a
second case for one skill would overwrite reports, hand back the first run's
results, and read the wrong fixture — silently.

## The tree carries a manifest

`$BIOPB_SKILL_FIXTURES/manifest.json` describes what a machine actually has, so
the fixtures present can be listed, their citations collected, and their contents
checked — none of which is possible against a bare convention.

```json
{
  "version": 1,
  "fixtures": [
    {
      "skill": "align-stack-by-features",
      "case_id": "ovule-serial-sections",
      "provenance": "PlantSeg Arabidopsis ovules, N_425_ds2x.tif, slices 10-21",
      "citation": "Wolny et al., eLife 2020;9:e57613",
      "licence": "CC BY 4.0",
      "files": {
        "arrays.npz": {
          "sha256": "e3b0c442...",
          "bytes": 7077888,
          "arrays": {"stack": {"shape": [12, 384, 384], "dtype": "float32"}}
        }
      }
    }
  ]
}
```

`Fixture.citation` is read from here and required for `kind == "curated"`, then
carried into the report and the artifact directory. ACDC ships a
`MANDATORY_CITATION.md`; that obligation belongs to the harness rather than to
whoever remembers it.

### Content checking is out-of-band

**The SHA is in the manifest; verifying it is not part of a benchmark run.**
Hashing multi-gigabyte volumes on a mount would cost more than the run it guards,
and would do it on every arm.

Instead a marker-gated check, matching how `satisfiability` and `availability`
are already held back from the default suite:

```sh
uv run --no-sync pytest -m fixtures biopb-mcp/src/biopb_mcp/_tests/skills
```

Run after syncing the tree, or when a result looks wrong. It walks the manifest,
hashes each file, and reports drift — including a fixture present on disk but
absent from the manifest, which is how an unreviewed acquisition would otherwise
slip into a run.

**Shape and dtype are checked in-band**, at fixture build time. The manifest
already records them, reading them is a header read rather than a pass over the
bytes, and a mismatch means the file under this path is not the file the case was
written against — which is the failure that matters most under
non-decomposability, since it is the one remaining way a case name could quietly
denote two experiments. It does not catch altered pixels; that is what the
out-of-band hash is for.

## Migration

Mechanical, and no existing case changes behaviour — the five shipped cases are
`Procedural`, `presentation="array"`, and gain an explicit `case_id` they already
had inside their builders:

```python
# before
build=AmbiguousChannels(),
# after
case_id="two-channels-one-structural",
fixture=Procedural(AmbiguousChannels()),
```

PR #704 is superseded rather than rebased: its `build=None` +
`no_synthetic_reason` is the override model patched, and this removes the model.

Done, and it went as described: the five builders also stopped restating
`skill_id`/`case_id`/`kind`, which the spec now stamps from the case that owns
them — a second place to name a case is a second place for the name to drift,
with the report saying one thing and the artifact directory another.

`skill-testing.md` §5d has been corrected in the same change; it used to say real
data "can replace a synthetic case without touching the verifier", which is
precisely the framing being retired.

## Settled

- **`array` and `tensor` are both legal**, chosen per case to match what the
  skill was written against — no default and no ranking. When `tensor` is chosen,
  one data plane serves the whole run.
- **The tree carries a root manifest**, with provenance, citation, licence and
  per-file SHA.
- **Hash verification is out-of-band**, behind a `fixtures` marker, never inside a
  benchmark run.
- **Presentation coverage is reported per skill and only warns.** A skill whose
  cases do not exercise everything its `checklist:` declares is incompletely
  benchmarked, which is a backlog entry, not an error.
- **Arms are isolated by id**: a fixture's id is a one-way hash of a per-run
  secret name, so it cannot be reached from anything an agent is given; a
  fingerprint check flags any change anyway.
- **Shape and dtype are checked in-band** at build time; the SHA stays
  out-of-band.

Nothing is left open by design; what follows is what landing it implies.

## What this implies

The five existing cases need **no migration**: `array` is a legal presentation, so
they stay exactly as written. Only their `case_id` becomes explicit.

What the coverage warning says on day one, and it is worse than predicted:
**all five benchmarked skills** declare `dask` or `tensor` and none has a case
that presents one, so every arm ever run has had `client is None`. That is a
backlog entry this design creates the means to close, not a defect it introduces —
and it is now a line of pytest output rather than something nobody had counted.
