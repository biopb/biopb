"""What a run is given, what it has to recover, and how that is scored.

Skill-agnostic on purpose: nothing here knows what drift is. A case module
supplies a builder that returns a :class:`Fixture` and a verifier that turns
``(fixture, attempt)`` into an :class:`Outcome`; this file is the vocabulary
they are both written in.

Two properties shape it, and they are the only non-obvious parts.

*Truth is data, not a formula.* A synthetic fixture knows the answer because it
constructed it; a curated one knows whatever a human annotated. Both hand the
verifier a mapping, and the verifier reads the keys it needs — which is what
lets real data replace a synthetic case without touching the verifier (§5a).

*A metric it cannot compute is unavailable, not passing.* A real movie has no
un-drifted reference image, so any metric needing one is absent from the report.
:attr:`Outcome.passed` is false when *nothing* was scored, which is what stops a
fixture with an empty truth from reading as a clean run — and, just as often
here, what stops an agent that left nothing behind from reading as one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Kind = Literal["synthetic", "curated"]


@dataclass(frozen=True)
class Fixture:
    """One case: what a run is given, and what it has to recover.

    ``data`` and ``truth`` are deliberately separate mappings rather than one
    object with optional fields. It makes the leak the whole layer depends on
    not happening — a truth key appearing in ``data`` — a thing a test can
    assert about any fixture, synthetic or curated, without knowing the skill.
    """

    skill_id: str
    case_id: str
    kind: Kind
    provenance: str
    data: Mapping[str, Any]
    truth: Mapping[str, Any]
    tolerance: Mapping[str, float]
    #: Prose for the artifact directory: what this case is meant to be hard about.
    about: str = ""

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


# --- the curated path ------------------------------------------------------

#: Where a curated fixture tree lives. Unset on most machines, and that is the
#: normal state -- a case falls back to its own procedural builder.
FIXTURE_DIR_ENV = "BIOPB_SKILL_FIXTURES"


def curated_root() -> Path | None:
    raw = os.environ.get(FIXTURE_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class CuratedNpz:
    """A fixture read off disk: real data someone acquired and annotated.

    A synthetic fixture is not a microscope — no vendor metadata, no genuine
    vignetting, no real stage error. This exists so substituting real data is
    *putting a directory somewhere*, not writing code. Under
    ``$BIOPB_SKILL_FIXTURES/<skill_id>/<case_id>/``:

    ``case.json``
        ``{"provenance": ..., "about": ..., "data": [key, ...],
        "truth": [key, ...], "tolerance": {metric: limit}}`` — the two key
        lists partition ``arrays.npz``, and the split is what the verifier
        trusts. A key in neither list is ignored; a key in both is an error,
        because a truth the run can see is not a truth.

    ``arrays.npz``
        every array named by those lists.

    What the substitution costs is **truth**: a curated movie can carry a
    trajectory someone measured off a bead, but not the un-drifted reference
    image, because no such acquisition exists. A metric the fixture cannot
    support reports as unavailable, never as passing.

    Nothing here validates the *science* of the annotation. It cannot: whether
    someone's fiducial trajectory is right is a review question, and it is the
    review a synthetic seed does not need. That asymmetry is the real cost of
    real data, and it belongs in the case's ``provenance``.
    """

    skill_id: str
    case_id: str
    kind: Kind = "curated"

    @property
    def _dir(self) -> Path | None:
        root = curated_root()
        return None if root is None else root / self.skill_id / self.case_id

    def available(self) -> tuple[bool, str]:
        root = curated_root()
        if root is None:
            return False, f"{FIXTURE_DIR_ENV} is not set"
        here = self._dir
        assert here is not None
        missing = [n for n in ("case.json", "arrays.npz") if not (here / n).is_file()]
        if missing:
            return False, f"{here} is missing {', '.join(missing)}"
        return True, ""

    def build(self) -> Fixture:
        here = self._dir
        assert here is not None, f"{FIXTURE_DIR_ENV} is not set"
        spec = json.loads((here / "case.json").read_text(encoding="utf-8"))
        data_keys = list(spec.get("data") or ())
        truth_keys = list(spec.get("truth") or ())
        if overlap := set(data_keys) & set(truth_keys):
            raise ValueError(
                f"{here}: {sorted(overlap)} listed as both data and truth -- a "
                "truth the run can see is not a truth"
            )
        with np.load(here / "arrays.npz") as arrays:
            data = {k: arrays[k] for k in data_keys}
            truth = {k: arrays[k] for k in truth_keys}
        return Fixture(
            skill_id=self.skill_id,
            case_id=self.case_id,
            kind="curated",
            provenance=spec.get("provenance", str(here)),
            about=spec.get("about", ""),
            data=data,
            truth=truth,
            tolerance=dict(spec.get("tolerance") or {}),
        )


def curated_for(skill_id: str) -> Fixture | None:
    """The curated fixture for *skill_id*, if this machine has one.

    Checked before a case builds its own, so pointing `$BIOPB_SKILL_FIXTURES`
    at a tree of real acquisitions substitutes them for the procedural fixtures
    without editing anything. The first case directory wins: a curated tree is a
    deliberate act, and one skill's benchmark runs one fixture.
    """
    root = curated_root()
    if root is None or not (root / skill_id).is_dir():
        return None
    for case in sorted((root / skill_id).iterdir()):
        if case.is_dir() and (case / "case.json").is_file():
            return CuratedNpz(skill_id=skill_id, case_id=case.name).build()
    return None


# --- artifacts -------------------------------------------------------------

#: Every case emits a number *and* an artifact: the number says what happened,
#: the artifact explains it. Imaging failures are recognisable in a second by
#: eye and awkward to characterise in an assertion.
ARTIFACT_DIR_ENV = "BIOPB_SKILL_OUTCOME_DIR"


def artifact_root() -> Path:
    raw = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    # .../biopb-mcp/src/biopb_mcp/_tests/skills/interaction/_fixture.py -> the
    # checkout root. Landing beside the source is deliberate: these are meant to
    # be opened and paged through, not hunted for in a temp dir.
    return Path(__file__).resolve().parents[6] / ".skill-outcomes"


def write_report(outcome: Outcome, root: Path) -> Path:
    """Write *outcome* to ``root/<case>/<subject>/`` and return that directory."""
    here = root / outcome.fixture.case_id / outcome.attempt.subject
    here.mkdir(parents=True, exist_ok=True)
    (here / "summary.json").write_text(
        json.dumps(
            {
                "skill": outcome.fixture.skill_id,
                "case": outcome.fixture.case_id,
                "kind": outcome.fixture.kind,
                "provenance": outcome.fixture.provenance,
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
