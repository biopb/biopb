"""What a run is given, what it has to recover, and how that is scored.

Skill-agnostic on purpose: nothing here knows what drift is. A case module
supplies a builder that returns a :class:`Fixture` and a verifier that turns
``(fixture, attempt)`` into an :class:`Outcome`; this file is the vocabulary
they are both written in.

Two properties shape it, and they are the only non-obvious parts.

*Truth is data, not a formula.* A synthetic fixture knows the answer because it
constructed it; a curated one knows whatever a human annotated. Both hand the
verifier a mapping, and the verifier reads the keys it needs — so one verifier
serves either kind. What that does *not* license is swapping one for the other
under a running case: a case owns exactly one fixture, fixed when it was
written (`docs/skill-fixtures.md`).

*A metric it cannot compute is unavailable, not passing.* A real movie has no
un-drifted reference image, so any metric needing one is absent from the report.
:attr:`Outcome.passed` is false when *nothing* was scored, which is what stops a
fixture with an empty truth from reading as a clean run — and, just as often
here, what stops an agent that left nothing behind from reading as one.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

Kind = Literal["synthetic", "curated"]


@dataclass(frozen=True)
class Fixture:
    """One case: what a run is given, and what it has to recover.

    ``data`` and ``truth`` are deliberately separate mappings rather than one
    object with optional fields. It makes the leak the whole layer depends on
    not happening — a truth key appearing in ``data`` — a thing a test can
    assert about any fixture, synthetic or curated, without knowing the skill.

    Values may be arrays or :class:`ArrayRef` handles; verifiers read through
    ``np.asarray`` either way, so a truth volume the size of the acquisition
    costs nothing until something looks at it.
    """

    provenance: str
    data: Mapping[str, Any]
    truth: Mapping[str, Any]
    tolerance: Mapping[str, float]
    #: Prose for the artifact directory: what this case is meant to be hard about.
    about: str = ""
    #: Whose data this is. Required of a curated fixture, read from the tree's
    #: manifest rather than from whoever remembers — ACDC ships a
    #: `MANDATORY_CITATION.md`, and that obligation belongs to the harness.
    citation: str = ""
    #: Identity and kind are **stamped by the :class:`FixtureSpec`** from the
    #: `Case` that owns it, so a builder does not restate what the case already
    #: declares. Anything set here by a builder is overwritten.
    kind: Kind = "synthetic"
    skill_id: str = ""
    case_id: str = ""

    @property
    def label(self) -> str:
        return f"{self.skill_id}/{self.case_id}"


@dataclass(frozen=True)
class Attempt:
    """What a run left behind, in the terms a verifier can score.

    ``arrays`` is whatever the harness could scrape out of the kernel, so it is
    routinely partial: an agent may bind one of the two names the task asked
    for, or bind a name to something of the wrong shape. A verifier reports what
    it could not score rather than raising.
    """

    subject: str
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class Metric:
    """One number, its limit, and whether it could be computed at all.

    ``value is None`` is the load-bearing state: it says *this run, against this
    fixture's truth, does not support this measurement*, which must never be
    confused with a pass.
    """

    name: str
    value: float | None
    limit: float
    unit: str = ""
    unavailable: str = ""

    @property
    def scored(self) -> bool:
        return self.value is not None

    @property
    def passed(self) -> bool:
        return self.value is not None and self.value <= self.limit

    def __str__(self) -> str:
        if not self.scored:
            return f"{self.name}: not scored ({self.unavailable})"
        verdict = "ok" if self.passed else "FAIL"
        return (
            f"{self.name}: {self.value:.4g}{self.unit} (limit {self.limit:g}) {verdict}"
        )


@dataclass(frozen=True)
class Outcome:
    """A verifier's report on one (fixture, attempt) pair."""

    fixture: Fixture
    attempt: Attempt
    metrics: Sequence[Metric]
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> list[Metric]:
        return [m for m in self.metrics if m.scored]

    @property
    def failures(self) -> list[Metric]:
        return [m for m in self.scored if not m.passed]

    @property
    def passed(self) -> bool:
        """Every metric that could be computed is within its limit — and at
        least one could be. A run that scored nothing has not passed; it has not
        been tested."""
        return bool(self.scored) and not self.failures

    def summary(self) -> str:
        head = (
            f"{self.fixture.label} [{self.fixture.kind}] via {self.attempt.subject}: "
            f"{'PASS' if self.passed else 'FAIL'}"
        )
        return "\n  ".join([head, *(str(m) for m in self.metrics)])


# --- reading a run's leavings ----------------------------------------------
#
# Every verifier starts the same way, and it is not the part worth writing
# three times: an agent binds a name to the wrong thing about as often as it
# binds it to the right one, and *crashing* on that is worse than scoring it.
# An unscorable result is an ordinary outcome of an agent run and has to arrive
# as "not measured", the same as a truth the fixture cannot supply.


def read_array(
    attempt: Attempt, key: str, shape: tuple[int, ...]
) -> tuple[np.ndarray | None, str]:
    """``(array, why not)`` for *key*, required to be exactly *shape*."""
    got = attempt.arrays.get(key)
    if got is None:
        return None, f"the run left no `{key}`"
    got = np.asarray(got, float)
    if got.shape != shape:
        return None, f"the run's `{key}` is {got.shape}, not {shape}"
    return got, ""


def read_scalar(attempt: Attempt, key: str) -> tuple[float | None, str]:
    """``(number, why not)`` for a *key* the run should have bound to one."""
    got = attempt.arrays.get(key)
    if got is None:
        return None, f"the run left no `{key}`"
    got = np.asarray(got, float)
    if got.size != 1 or not np.isfinite(got).all():
        return None, f"the run's `{key}` is not a finite single number ({got.shape})"
    return float(got.reshape(())), ""


def relative_error(got, want) -> float:
    """Worst elementwise ``|got - want| / |want|``. The comparison for a
    quantity whose scale is the point — a volume in µm³ against one in voxels
    is wrong by a factor, not by an amount."""
    got = np.asarray(got, float)
    want = np.asarray(want, float)
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)))


# --- arrays that have not been read yet ------------------------------------
#
# A curated case's truth can be a label volume the size of the acquisition, and
# repacking that into an npz to hand a verifier one array is both wasteful and
# lossy. So a fixture's values may be *handles*: everything downstream already
# reads through `np.asarray`, so deferring the read costs no verifier a line.


@runtime_checkable
class ArrayRef(Protocol):
    """A handle to an array that has not been read yet."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __array__(self, dtype=None, copy=None) -> np.ndarray: ...

    def dask(self, chunks: Any = "auto") -> Any: ...


def _npy_header(stream) -> tuple[tuple[int, ...], np.dtype]:
    """``(shape, dtype)`` from a ``.npy`` stream, without reading the data."""
    version = np.lib.format.read_magic(stream)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ValueError(f"unsupported .npy format {version}")
    shape, _fortran, dtype = readers[version](stream)
    return tuple(shape), dtype


@dataclass(frozen=True)
class NpzRef:
    """One array inside an ``.npz``, read on use.

    Shape and dtype come from the member's own header, so checking a file
    against the manifest is a seek rather than a pass over the bytes.
    """

    path: Path
    key: str

    def _header(self) -> tuple[tuple[int, ...], np.dtype]:
        with (
            zipfile.ZipFile(self.path) as archive,
            archive.open(f"{self.key}.npy") as member,
        ):
            return _npy_header(member)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._header()[0]

    @property
    def dtype(self) -> np.dtype:
        return self._header()[1]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        with np.load(self.path) as arrays:
            got = arrays[self.key]
        return got if dtype is None else got.astype(dtype)

    def dask(self, chunks: Any = "auto") -> Any:
        import dask.array as da

        return da.from_array(np.asarray(self), chunks=chunks)


def _read_npy(path: Path):
    # The memmap *is* the deferral: a chunked read through it touches only the
    # pages it needs, which is what makes a `.npy` truth volume free to carry.
    return np.load(path, mmap_mode="r")


def _read_tif(path: Path):
    import tifffile

    return tifffile.imread(path)


def _header_tif(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        return tuple(series.shape), np.dtype(series.dtype)


def _read_nifti(path: Path):
    import nibabel

    return np.asarray(nibabel.load(path).dataobj)


def _header_nifti(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    import nibabel

    image = nibabel.load(path)
    return tuple(image.shape), np.dtype(image.get_data_dtype())


def _header_from_read(read: Callable[[Path], Any]) -> Callable[[Path], Any]:
    def header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
        got = read(path)
        return tuple(got.shape), np.dtype(got.dtype)

    return header


#: ``suffix -> (module the reader needs, header reader, array reader)``. Small
#: and explicit: a fixture tree is reviewed data, so the formats it may arrive
#: in are a decision rather than whatever the machine happens to import.
#:
#: ``.npz`` is absent on purpose — an archive holds many arrays, so naming one
#: file does not name an array. A `.npz` in a case's layout resolves through
#: :class:`NpzRef` on the key that referenced it.
_READERS: dict[str, tuple[str, Callable, Callable]] = {
    ".npy": ("numpy", _header_from_read(_read_npy), _read_npy),
    ".tif": ("tifffile", _header_tif, _read_tif),
    ".tiff": ("tifffile", _header_tif, _read_tif),
    ".nii": ("nibabel", _header_nifti, _read_nifti),
    ".nii.gz": ("nibabel", _header_nifti, _read_nifti),
}


def reader_suffix(path: Path | str) -> str:
    """The suffix `_READERS` is keyed on — ``.nii.gz`` is one suffix, not two."""
    name = str(path).lower()
    return ".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix


def ref_missing(path: Path | str) -> str:
    """Why this file cannot become an :class:`ArrayRef` here, or ``""``.

    Answerable without touching the file, so a tree half of whose formats this
    machine cannot open reports that as an availability fact rather than as a
    crash halfway through a paid run.
    """
    if Path(path).suffix.lower() == ".npz":
        return ""  # numpy is always here, and the key names the array
    return reader_missing(path)


def reader_missing(path: Path | str) -> str:
    """Why :class:`FileRef` cannot read this file here, or ``""``."""
    suffix = reader_suffix(path)
    if suffix not in _READERS:
        return f"no reader for {suffix or 'a file with no suffix'} ({path})"
    module, _, _ = _READERS[suffix]
    if importlib.util.find_spec(module) is None:
        return f"reading {path} needs {module}, which is not installed"
    return ""


@dataclass(frozen=True)
class FileRef:
    """A whole file as one array, read on use.

    Formats come from :data:`_READERS`; ``.npy`` is memory-mapped, so a truth
    volume larger than the test process is addressable rather than resident.
    The others defer the read but not the residency — deferral is what a ref
    promises, and only the mmap path delivers partial IO.
    """

    path: Path

    @property
    def _reader(self) -> tuple[str, Callable, Callable]:
        suffix = reader_suffix(self.path)
        if suffix not in _READERS:
            raise ValueError(f"no reader for {suffix!r} ({self.path})")
        return _READERS[suffix]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._reader[1](self.path)[0]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._reader[1](self.path)[1])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        got = np.asarray(self._reader[2](self.path))
        return got if dtype is None else got.astype(dtype)

    def dask(self, chunks: Any = "auto") -> Any:
        import dask.array as da

        return da.from_array(self._reader[2](self.path), chunks=chunks)


# --- where a case's data comes from ----------------------------------------

#: The **root path** under which on-disk fixtures live. Not a policy switch:
#: setting it changes where a curated case finds its data, never which fixture
#: a case runs. A case owns one fixture, chosen when it was written.
FIXTURE_DIR_ENV = "BIOPB_FIXTURES"

#: What a machine actually has, and where a curated fixture's provenance,
#: citation and per-file record live. A fixture directory with no entry here is
#: an acquisition nobody wrote down, and does not run.
MANIFEST_NAME = "manifest.json"


def fixture_root() -> Path | None:
    raw = os.environ.get(FIXTURE_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def read_manifest() -> Mapping[str, Any]:
    """The tree's manifest, or ``{}`` if this machine has no tree."""
    root = fixture_root()
    if root is None or not (root / MANIFEST_NAME).is_file():
        return {}
    return json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))


def manifest_entry(skill_id: str, case_id: str) -> Mapping[str, Any] | None:
    for entry in read_manifest().get("fixtures") or ():
        if entry.get("skill") == skill_id and entry.get("case_id") == case_id:
            return entry
    return None


@runtime_checkable
class FixtureSpec(Protocol):
    """Where this case's pixels come from. One per case, not a preference.

    There is no ordering between implementations and no fallback between them.
    Substituting the data makes it a different experiment with the same name:
    the truth changes, the achievable accuracy changes, and the conclusion can
    invert — measured, not hypothetical (`docs/skill-fixtures.md`). A skill
    worth covering both ways gets **two cases**, each with its own id and its
    own tolerances.

    ``build`` takes the owning case's identity rather than the case itself,
    which keeps this module free of any import from the engine above it.
    """

    kind: Kind

    def available(self, skill_id: str, case_id: str) -> tuple[bool, str]: ...

    def build(self, skill_id: str, case_id: str) -> Fixture: ...


def _stamp(fixture: Fixture, kind: Kind, skill_id: str, case_id: str) -> Fixture:
    """Identity onto a built fixture. The case declares it; nothing else may."""
    return replace(fixture, kind=kind, skill_id=skill_id, case_id=case_id)


@dataclass(frozen=True)
class Procedural:
    """Generated from a seed. Always available, and its truth is exact because
    the builder constructed it."""

    builder: Callable[[], Fixture]
    kind: Kind = "synthetic"

    def available(self, skill_id: str = "", case_id: str = "") -> tuple[bool, str]:
        return True, ""

    def build(self, skill_id: str, case_id: str) -> Fixture:
        return _stamp(self.builder(), self.kind, skill_id, case_id)


@dataclass(frozen=True)
class OnDisk:
    """Real data the case was written against.

    A synthetic fixture is not a microscope — no vendor metadata, no genuine
    vignetting, no real stage error — and for some skills that gap changes the
    answer rather than the difficulty. Such a case is written against an
    acquisition from the start.

    Rooted at ``$BIOPB_FIXTURES/<skill_id>/<case_id>/``: the case's own
    identity locates its data, so there is nothing to select and nothing to
    sort. It holds one file:

    ``case.json``
        ``{"about": ..., "data": {key: filename}, "truth": {key: filename}}``.
        The two mappings partition what the case reads, and the split is what
        the verifier trusts — a key in both is an error, because a truth the
        run can see is not a truth. A ``.npz`` filename resolves to the member
        named by its key.

    Everything *about* the data — provenance, citation, licence, per-file
    shape, dtype and hash — lives in the tree's manifest instead, so a curated
    case has exactly one place that records what was acquired.

    What real data costs is **truth**: a curated movie can carry a trajectory
    someone measured off a bead, but not the un-drifted reference image,
    because no such acquisition exists. A metric the fixture cannot support
    reports as unavailable, never as passing. And nothing here validates the
    *science* of the annotation — whether someone's fiducial trajectory is
    right is a review question, and it is the review a synthetic seed does not
    need. That asymmetry belongs in the manifest's ``provenance``.
    """

    #: Limits belong to the case module, beside the verifier that reports
    #: against them — never to the tree, where a machine's copy of the data
    #: could quietly re-tune what counts as a pass.
    tolerance: Mapping[str, float] = field(default_factory=dict)
    kind: Kind = "curated"

    def where(self, skill_id: str, case_id: str) -> Path | None:
        root = fixture_root()
        return None if root is None else root / skill_id / case_id

    def available(self, skill_id: str = "", case_id: str = "") -> tuple[bool, str]:
        """Whether this machine can run the case, and why not.

        The identity is passed in because the spec does not hold it; the engine
        calls this with the case's own, and the argument-free form exists only
        so the protocol stays uniform.
        """
        root = fixture_root()
        if root is None:
            return False, f"{FIXTURE_DIR_ENV} is not set, and this case's data is real"
        here = self.where(skill_id, case_id)
        assert here is not None
        if not (here / "case.json").is_file():
            return False, f"{here / 'case.json'} is missing"
        if manifest_entry(skill_id, case_id) is None:
            return False, (
                f"{skill_id}/{case_id} is on disk but not in {root / MANIFEST_NAME} — "
                "an acquisition nothing recorded is not one a run should score"
            )
        spec = json.loads((here / "case.json").read_text(encoding="utf-8"))
        for name in {**(spec.get("data") or {}), **(spec.get("truth") or {})}.values():
            if not (here / name).is_file():
                return False, f"{here / name} is missing"
            if why := ref_missing(here / name):
                return False, why
        return True, ""

    def build(self, skill_id: str, case_id: str) -> Fixture:
        here = self.where(skill_id, case_id)
        assert here is not None, f"{FIXTURE_DIR_ENV} is not set"
        spec = json.loads((here / "case.json").read_text(encoding="utf-8"))
        data_files = dict(spec.get("data") or {})
        truth_files = dict(spec.get("truth") or {})
        if overlap := set(data_files) & set(truth_files):
            raise ValueError(
                f"{here}: {sorted(overlap)} listed as both data and truth -- a "
                "truth the run can see is not a truth"
            )
        entry = manifest_entry(skill_id, case_id)
        if entry is None:
            raise ValueError(f"{skill_id}/{case_id} is not in {MANIFEST_NAME}")
        if not str(entry.get("citation", "")).strip():
            raise ValueError(
                f"{skill_id}/{case_id} records no citation. Real data comes from "
                "someone, and saying so is not optional."
            )
        refs = {
            key: _ref(here, name, key)
            for key, name in {**data_files, **truth_files}.items()
        }
        _agrees_with_manifest(entry, {**data_files, **truth_files}, refs)
        return Fixture(
            provenance=entry.get("provenance", str(here)),
            citation=str(entry["citation"]),
            about=spec.get("about", ""),
            data={k: refs[k] for k in data_files},
            truth={k: refs[k] for k in truth_files},
            tolerance=dict(self.tolerance),
            kind=self.kind,
            skill_id=skill_id,
            case_id=case_id,
        )


def _ref(here: Path, name: str, key: str):
    path = here / name
    return NpzRef(path, key) if path.suffix.lower() == ".npz" else FileRef(path)


def recorded_layout(
    entry: Mapping[str, Any], filename: str, key: str
) -> Mapping[str, Any] | None:
    """What the manifest says this array is, or ``None`` if it says nothing."""
    record = (entry.get("files") or {}).get(filename)
    if not isinstance(record, Mapping):
        return None
    if arrays := record.get("arrays"):
        got = arrays.get(key)
        return got if isinstance(got, Mapping) else None
    return record if "shape" in record else None


def _agrees_with_manifest(entry, files: Mapping[str, str], refs: Mapping[str, Any]):
    """Shape and dtype, in-band, at build time.

    A header read rather than a pass over the bytes, and it catches the one
    remaining way a case name could quietly denote two experiments: the file
    under this path not being the file the case was written against. It does
    not catch altered pixels — that is what the out-of-band hash is for.
    """
    for key, filename in files.items():
        layout = recorded_layout(entry, filename, key)
        if layout is None:
            raise ValueError(
                f"{entry.get('skill')}/{entry.get('case_id')}: {MANIFEST_NAME} "
                f"records no shape for `{key}` in {filename}"
            )
        want_shape = tuple(layout["shape"])
        got_shape = tuple(refs[key].shape)
        if got_shape != want_shape:
            raise ValueError(
                f"{filename}[{key}] is {got_shape}, but {MANIFEST_NAME} records "
                f"{want_shape} — this is not the data the case was written against"
            )
        if want_dtype := layout.get("dtype"):
            got_dtype = np.dtype(refs[key].dtype)
            if got_dtype != np.dtype(want_dtype):
                raise ValueError(
                    f"{filename}[{key}] is {got_dtype}, but {MANIFEST_NAME} "
                    f"records {want_dtype}"
                )


# --- artifacts -------------------------------------------------------------

#: Every case emits a number *and* an artifact: the number says what happened,
#: the artifact explains it. Imaging failures are recognisable in a second by
#: eye and awkward to characterise in an assertion.
ARTIFACT_DIR_ENV = "BIOPB_OUTCOME_DIR"


def artifact_root() -> Path:
    """Where a suite's reports and transcripts land, by default in the checkout.

    Landing beside the source is deliberate: these are meant to be opened and
    paged through, not hunted for in a temp dir.

    The root is *searched for* rather than counted to. It used to be
    ``parents[6]``, right for one location and silently wrong the moment this
    module moved a directory -- it then resolved above the checkout, where the
    gitignore does not reach and nobody would think to look for a report. A
    depth that only works from one path is a landmine for the next move, so the
    marker is what gets looked for.
    """
    raw = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent / ".skill-outcomes"
    # An installed copy with no checkout around it: keep artifacts beside
    # whoever ran the thing rather than somewhere up the filesystem.
    return Path.cwd() / ".skill-outcomes"


def write_report(outcome: Outcome, root: Path) -> Path:
    """Write *outcome* to ``root/<subject>/`` and return that directory.

    *root* is already the case's own directory (`where_for`), which is keyed on
    ``(skill, case_id)`` — so a second case for one skill writes beside the
    first rather than over it.
    """
    here = root / outcome.attempt.subject
    here.mkdir(parents=True, exist_ok=True)
    (here / "summary.json").write_text(
        json.dumps(
            {
                "skill": outcome.fixture.skill_id,
                "case": outcome.fixture.case_id,
                "kind": outcome.fixture.kind,
                "provenance": outcome.fixture.provenance,
                "citation": outcome.fixture.citation,
                "about": outcome.fixture.about,
                "subject": outcome.attempt.subject,
                "notes": outcome.attempt.notes,
                "passed": outcome.passed,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "limit": m.limit,
                        "unit": m.unit,
                        "scored": m.scored,
                        "passed": m.passed if m.scored else None,
                        "unavailable": m.unavailable,
                    }
                    for m in outcome.metrics
                ],
                "detail": {k: _plain(v) for k, v in outcome.detail.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return here


def save_png(image: np.ndarray, path: Path, *, vmax: float | None = None) -> None:
    """Best-effort 8-bit PNG, scaled to ``[0, vmax]`` when *vmax* is given.

    Pass a shared *vmax* for any pair of images meant to be compared. Scaling
    each independently is the trap: a corrected-minus-raw difference that is
    near zero everywhere gets stretched to full range and looks exactly like
    the uncorrected one — the artifact would be most misleading precisely where
    it is most wanted.

    An artifact explains a failure; it never causes one, so a missing imageio
    plugin is swallowed.
    """
    try:
        from skimage.io import imsave

        finite = np.nan_to_num(np.asarray(image, float))
        if vmax is None:
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, float(vmax)
        scaled = (
            np.clip((finite - lo) / (hi - lo), 0.0, 1.0)
            if hi > lo
            else np.zeros_like(finite)
        )
        imsave(path, (scaled * 255).astype(np.uint8), check_contrast=False)
    except Exception as exc:  # noqa: BLE001 - artifacts must not break a run
        path.with_suffix(".png.error").write_text(repr(exc), encoding="utf-8")


def _plain(value: Any) -> Any:
    """JSON-able view of a value that may be a numpy scalar or array."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
