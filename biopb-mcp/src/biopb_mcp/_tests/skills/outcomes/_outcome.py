"""The fixture protocol every outcome case is written against.

Skill-agnostic on purpose: nothing here knows what drift is. A skill's module
supplies a provider that builds a :class:`Fixture`, a runner that produces an
:class:`Attempt`, and a verifier that turns the pair into an :class:`Outcome`.

The shape is chosen so a **curated fixture of real data can replace a synthetic
one without touching the verifier**. Two things follow from that, and they are
the only non-obvious parts of this file:

*Truth is data, not a formula.* A synthetic fixture knows the answer because it
constructed it; a curated one knows whatever a human annotated. Both hand the
verifier a mapping, and the verifier reads the keys it needs.

*A metric it cannot compute is unavailable, not passing.* A real movie has no
un-drifted reference image, so any metric needing one is simply absent from the
report. :attr:`Outcome.passed` is false when *nothing* was scored, which is what
stops a fixture with an empty truth from reading as a clean run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

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

    Fields are optional because a run may legitimately produce only some of
    them — the degraded path of a skill can yield a trajectory without ever
    materialising a corrected array, and an agent may leave one and not the
    other in its namespace. A verifier reports what it could not score.
    """

    subject: str
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class Metric:
    """One number, its limit, and whether it could be computed at all.

    ``value is None`` is the load-bearing state: it says *this fixture's truth
    does not support this measurement*, which is the normal condition for a
    curated real fixture and must never be confused with a pass.
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
        least one could be. A fixture whose truth supports nothing has not
        passed; it has not been tested."""
        return bool(self.scored) and not self.failures

    def summary(self) -> str:
        head = (
            f"{self.fixture.label} [{self.fixture.kind}] via {self.attempt.subject}: "
            f"{'PASS' if self.passed else 'FAIL'}"
        )
        return "\n  ".join([head, *(str(m) for m in self.metrics)])


# --- providers -------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """A source of one fixture. The seam a curated fixture substitutes at."""

    skill_id: str
    case_id: str
    kind: Kind

    def available(self) -> tuple[bool, str]:
        """``(usable, why not)``. A synthetic provider is always usable; a
        curated one answers for whether its data is on this machine."""

    def build(self) -> Fixture: ...


_PROVIDERS: dict[str, list[Provider]] = {}


def register(provider: Provider) -> Provider:
    """Add *provider* to the registry, newest last. Returns it, so it can be
    used as a decorator on a provider class instance."""
    cases = _PROVIDERS.setdefault(provider.skill_id, [])
    if any(p.case_id == provider.case_id for p in cases):
        raise ValueError(
            f"duplicate fixture case {provider.skill_id}/{provider.case_id}"
        )
    cases.append(provider)
    return provider


def providers_for(skill_id: str) -> list[Provider]:
    return list(_PROVIDERS.get(skill_id, ()))


def registered_skills() -> list[str]:
    return sorted(_PROVIDERS)


# --- the curated path ------------------------------------------------------

#: Where a curated fixture tree lives. Unset on most machines, and that is the
#: normal state -- curated cases skip rather than fail.
FIXTURE_DIR_ENV = "BIOPB_SKILL_FIXTURES"


def curated_root() -> Path | None:
    raw = os.environ.get(FIXTURE_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class CuratedNpz:
    """A fixture read off disk: real data someone acquired and annotated.

    This exists so that substituting real data for a synthetic case is *putting
    a directory somewhere*, not writing code. Under
    ``$BIOPB_SKILL_FIXTURES/<skill_id>/<case_id>/``:

    ``case.json``
        ``{"provenance": ..., "about": ..., "data": [key, ...],
        "truth": [key, ...], "tolerance": {metric: limit}}`` — the two key
        lists partition ``arrays.npz``, and the split is what the verifier
        trusts. A key in neither list is ignored; a key in both is an error,
        because a truth the run can see is not a truth.

    ``arrays.npz``
        every array named by those lists.

    Nothing here validates the *science* of the annotation. It cannot: whether
    someone's fiducial trajectory is right is a review question, and this is the
    same review a synthetic seed does not need. That asymmetry is the real cost
    of real data, and it belongs in the case's ``provenance``.
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


def register_curated(skill_id: str) -> list[CuratedNpz]:
    """Register every curated case present for *skill_id*, if any.

    Called at import time by a skill's fixture module. A tree that is not there
    registers nothing: no phantom case appears for data that was never going to
    be on this machine. The *tier* still advertises itself, as a single skip
    carrying the name of this env var — see `test_drift_correction.CURATED`.
    """
    root = curated_root()
    if root is None or not (root / skill_id).is_dir():
        return []
    added = []
    for case in sorted((root / skill_id).iterdir()):
        if case.is_dir() and (case / "case.json").is_file():
            added.append(register(CuratedNpz(skill_id=skill_id, case_id=case.name)))
    return added


# --- artifacts -------------------------------------------------------------

#: Per §2, every outcome case emits a number *and* an artifact: the number
#: gates, the artifact explains. Imaging failures are recognisable in a second
#: by eye and awkward to characterise in an assertion.
ARTIFACT_DIR_ENV = "BIOPB_SKILL_OUTCOME_DIR"


def artifact_root() -> Path:
    raw = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    # .../biopb-mcp/src/biopb_mcp/_tests/skills/outcomes/_outcome.py -> checkout
    # root. Landing beside the source is deliberate: these are meant to be
    # opened and paged through, not hunted for in a temp dir.
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
