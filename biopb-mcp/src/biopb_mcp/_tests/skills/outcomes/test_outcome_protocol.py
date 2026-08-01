"""The fixture protocol itself — including the curated path, which no machine
in CI has data for.

Without these, `CuratedNpz` is a promise rather than a mechanism: the door is
only open if something has walked through it. So these tests build a curated
tree in a temp dir and read it back, which exercises every line of the
substitution path using arrays that are three floats long.

Hermetic and instant, so unlike the rest of this package they are *not* marked
`outcome` — they run in the ordinary suite, where a regression in the protocol
is a normal test failure rather than something noticed on a workstation later.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ._outcome import (
    ARTIFACT_DIR_ENV,
    FIXTURE_DIR_ENV,
    Attempt,
    CuratedNpz,
    Fixture,
    Metric,
    Outcome,
    artifact_root,
    register,
    register_curated,
    write_report,
)


def _fixture(**kw) -> Fixture:
    base = {
        "skill_id": "a-skill",
        "case_id": "a-case",
        "kind": "synthetic",
        "provenance": "test",
        "data": {},
        "truth": {},
        "tolerance": {},
    }
    return Fixture(**{**base, **kw})


def _outcome(*metrics: Metric) -> Outcome:
    return Outcome(fixture=_fixture(), attempt=Attempt(subject="s"), metrics=metrics)


# --- the anti-vacuous rules ------------------------------------------------


def test_an_unavailable_metric_is_not_a_passing_one():
    """The distinction the curated path depends on. A real movie has no
    un-drifted reference, so a metric needing one is absent -- and absent must
    not read as satisfied."""
    m = Metric("x", None, 1.0, unavailable="no reference exists")
    assert not m.scored
    assert not m.passed
    assert "not scored" in str(m)


def test_an_outcome_that_scored_nothing_has_not_passed():
    """A fixture whose truth supports no metric has not been tested. Reporting
    that as a pass is how a substituted fixture with a mis-declared truth would
    turn the whole layer green while measuring nothing."""
    empty = _outcome(Metric("x", None, 1.0, unavailable="why"))
    assert not empty.passed
    assert empty.scored == []


def test_an_outcome_passes_only_when_every_scored_metric_is_within_its_limit():
    assert _outcome(Metric("x", 0.5, 1.0)).passed
    assert not _outcome(Metric("x", 1.5, 1.0)).passed
    # One unavailable metric does not spoil an otherwise scored run.
    assert _outcome(
        Metric("x", 0.5, 1.0), Metric("y", None, 1.0, unavailable="n/a")
    ).passed
    assert _outcome(Metric("x", 1.5, 1.0)).failures[0].name == "x"


def test_the_limit_is_inclusive():
    """A value exactly at the limit passes. Stated because it is the kind of
    boundary that gets flipped silently while re-tuning a tolerance."""
    assert Metric("x", 1.0, 1.0).passed


# --- the registry ----------------------------------------------------------


def test_registering_two_cases_under_one_id_is_an_error():
    """A duplicate case_id would make the second provider invisible: it would
    register, never be selected by id, and its assertions would silently be
    someone else's."""
    provider = CuratedNpz(skill_id="dupe-skill", case_id="dupe-case")
    register(provider)
    with pytest.raises(ValueError, match="duplicate fixture case"):
        register(CuratedNpz(skill_id="dupe-skill", case_id="dupe-case"))


# --- the curated path ------------------------------------------------------


def _write_curated(root, *, data, truth, **spec):
    here = root / "a-skill" / "a-case"
    here.mkdir(parents=True)
    (here / "case.json").write_text(
        json.dumps({"data": list(data), "truth": list(truth), **spec}),
        encoding="utf-8",
    )
    np.savez(here / "arrays.npz", **data, **truth)
    return here


def test_a_curated_fixture_reads_back_with_its_truth_kept_apart(tmp_path, monkeypatch):
    """The whole substitution, end to end: a directory becomes a Fixture with
    the same shape a procedural provider builds."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    _write_curated(
        tmp_path,
        data={"movie": np.zeros((2, 3, 3), np.float32)},
        truth={"offsets": np.array([[0.0, 0.0], [1.0, 2.0]])},
        provenance="2026-03-04 timelapse, drift from the bead in the corner",
        about="a real acquisition",
        tolerance={"trajectory_rms_px": 1.5},
    )

    f = CuratedNpz(skill_id="a-skill", case_id="a-case").build()
    assert f.kind == "curated"
    assert set(f.data) == {"movie"} and set(f.truth) == {"offsets"}
    assert f.tolerance["trajectory_rms_px"] == 1.5
    assert "bead" in f.provenance


def test_a_curated_fixture_may_not_declare_a_key_as_both(tmp_path, monkeypatch):
    """A truth the run can see is not a truth. This is the one way a curated
    tree can be wrong that produces a *plausible* score rather than an error,
    so it is rejected at load rather than left to a reviewer."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    _write_curated(
        tmp_path,
        data={"movie": np.zeros((2, 2, 2))},
        truth={},
        provenance="p",
    )
    here = tmp_path / "a-skill" / "a-case"
    spec = json.loads((here / "case.json").read_text())
    spec["truth"] = ["movie"]
    (here / "case.json").write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="both data and truth"):
        CuratedNpz(skill_id="a-skill", case_id="a-case").build()


def test_an_absent_curated_tree_registers_nothing(monkeypatch):
    """The normal state on every machine but one. It has to register nothing
    rather than register-and-skip, or every local run carries a permanent
    yellow line for data that was never going to be there."""
    monkeypatch.delenv(FIXTURE_DIR_ENV, raising=False)
    assert register_curated("a-skill") == []
    usable, why = CuratedNpz(skill_id="a-skill", case_id="a-case").available()
    assert not usable and FIXTURE_DIR_ENV in why


def test_a_curated_case_missing_its_arrays_is_unavailable_not_broken(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    (tmp_path / "a-skill" / "a-case").mkdir(parents=True)
    (tmp_path / "a-skill" / "a-case" / "case.json").write_text("{}", encoding="utf-8")

    usable, why = CuratedNpz(skill_id="a-skill", case_id="a-case").available()
    assert not usable and "arrays.npz" in why


# --- artifacts -------------------------------------------------------------


def test_the_report_records_what_could_not_be_measured(tmp_path):
    """An artifact directory has to answer "was this green because it passed,
    or because nothing ran", so the unavailable metrics are written out too."""
    outcome = Outcome(
        fixture=_fixture(),
        attempt=Attempt(subject="s"),
        metrics=[Metric("a", 0.5, 1.0), Metric("b", None, 1.0, unavailable="no truth")],
        detail={"series": np.array([1.0, 2.0])},
    )
    where = write_report(outcome, tmp_path)
    written = json.loads((where / "summary.json").read_text(encoding="utf-8"))

    assert written["passed"] is True
    by_name = {m["name"]: m for m in written["metrics"]}
    assert by_name["b"]["scored"] is False and by_name["b"]["passed"] is None
    assert written["detail"]["series"] == [1.0, 2.0]


def test_the_artifact_root_is_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv(ARTIFACT_DIR_ENV, str(tmp_path / "elsewhere"))
    assert artifact_root() == tmp_path / "elsewhere"
    monkeypatch.delenv(ARTIFACT_DIR_ENV)
    assert artifact_root().name == ".skill-outcomes"
