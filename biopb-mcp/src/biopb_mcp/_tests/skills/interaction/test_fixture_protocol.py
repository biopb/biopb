"""The fixture protocol itself — including the on-disk path, which no machine
in CI has data for.

Without these, `OnDisk` is a promise rather than a mechanism: the door is only
open if something has walked through it. So these tests build a fixture tree in
a temp dir and read it back, which exercises every line of the curated path
using arrays that are three floats long.

Hermetic and instant, so they run in the ordinary suite — a regression in the
protocol should be a normal test failure rather than something noticed on a
workstation later, halfway through a paid run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ._fixture import (
    ARTIFACT_DIR_ENV,
    FIXTURE_DIR_ENV,
    Attempt,
    FileRef,
    Fixture,
    Metric,
    NpzRef,
    OnDisk,
    Outcome,
    Procedural,
    artifact_root,
    read_array,
    read_scalar,
    relative_error,
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
    """A fixture whose truth supports no metric has not been tested, and neither
    has a run that left nothing behind. Reporting either as a pass is how a
    whole layer turns green while measuring nothing."""
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


# --- reading what a run left ------------------------------------------------


def test_an_array_of_the_wrong_shape_is_unscorable_not_a_crash():
    """An agent binds a name to the wrong thing about as often as to the right
    one, and every verifier would otherwise open with the same four lines."""
    attempt = Attempt(subject="s", arrays={"v": np.zeros((3,))})
    got, why = read_array(attempt, "v", (3,))
    assert got is not None and not why

    got, why = read_array(attempt, "v", (4,))
    assert got is None and "(3,), not (4,)" in why

    got, why = read_array(attempt, "missing", (3,))
    assert got is None and "left no `missing`" in why


def test_a_scalar_is_read_however_the_kernel_boxed_it():
    """`get_array` returns whatever `np.asarray` made of the expression, so a
    float arrives 0-d and a one-element list arrives as (1,)."""
    for value in (np.float64(0.8), np.array(0.8), np.array([0.8])):
        got, why = read_scalar(Attempt(subject="s", arrays={"p": value}), "p")
        assert got == pytest.approx(0.8) and not why

    got, why = read_scalar(Attempt(subject="s", arrays={"p": np.zeros(3)}), "p")
    assert got is None and "single number" in why

    got, why = read_scalar(Attempt(subject="s", arrays={"p": np.array(np.nan)}), "p")
    assert got is None, "a nan score is not a score"


def test_relative_error_is_the_worst_element_not_the_mean():
    """One object measured in voxels among twelve in µm is the failure; an
    average over the twelve would bury it."""
    assert relative_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert relative_error([1.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)


# --- arrays that have not been read yet ------------------------------------


def test_a_ref_answers_shape_and_dtype_from_the_header(tmp_path):
    """The property the in-band manifest check rests on: asking what an array
    *is* must not be a pass over its bytes, or checking a multi-gigabyte volume
    would cost more than the run it guards."""
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    np.save(tmp_path / "v.npy", volume)
    np.savez(tmp_path / "a.npz", stack=volume)

    for ref in (FileRef(tmp_path / "v.npy"), NpzRef(tmp_path / "a.npz", "stack")):
        assert ref.shape == (2, 3, 4)
        assert ref.dtype == np.uint16
        assert np.array_equal(np.asarray(ref), volume)


def test_a_verifier_reads_a_ref_the_same_way_it_reads_an_array(tmp_path):
    """Why refs cost no verifier a line: everything downstream already goes
    through `np.asarray`, so deferral is invisible above this file."""
    np.save(tmp_path / "v.npy", np.zeros((3,), np.float32))
    attempt = Attempt(subject="s", arrays={"v": FileRef(tmp_path / "v.npy")})
    got, why = read_array(attempt, "v", (3,))
    assert got is not None and not why


def test_a_format_this_machine_cannot_read_is_an_availability_answer(tmp_path):
    """Answerable without touching the file, so a tree half of whose formats
    are unreadable here says so before anything is spawned or spent."""
    from ._fixture import reader_missing, reader_suffix

    assert reader_suffix("scan.nii.gz") == ".nii.gz"
    assert "no reader" in reader_missing(tmp_path / "scan.czi")
    assert reader_missing(tmp_path / "v.npy") == ""


# --- the on-disk path ------------------------------------------------------

CITATION = "Bernard et al., IEEE TMI 37(11):2514, 2018"


def _write_tree(root, *, data, truth, case="a-case", entry=None, **spec):
    """A fixture tree: the manifest that records the acquisition, and the case
    directory that says which array is data and which is truth."""
    here = root / "a-skill" / case
    here.mkdir(parents=True)
    arrays = {**data, **truth}
    np.savez(here / "arrays.npz", **arrays)
    (here / "case.json").write_text(
        json.dumps(
            {
                "data": dict.fromkeys(data, "arrays.npz"),
                "truth": dict.fromkeys(truth, "arrays.npz"),
                **spec,
            }
        ),
        encoding="utf-8",
    )
    record = {
        "skill": "a-skill",
        "case_id": case,
        "provenance": "2026-03-04 timelapse, drift from the bead in the corner",
        "citation": CITATION,
        "licence": "CC BY 4.0",
        "files": {
            "arrays.npz": {
                "sha256": "unchecked in-band",
                "arrays": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in arrays.items()
                },
            }
        },
    }
    record.update(entry or {})
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "fixtures": [record]}), encoding="utf-8"
    )
    return here


def test_an_on_disk_fixture_reads_back_with_its_truth_kept_apart(tmp_path, monkeypatch):
    """The whole curated path, end to end: a directory becomes a Fixture with
    the same shape a case's own builder returns — and its values arrive as
    handles, so a truth volume larger than the test process is addressable."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    _write_tree(
        tmp_path,
        data={"movie": np.zeros((2, 3, 3), np.float32)},
        truth={"offsets": np.zeros((2, 2))},
        about="a real acquisition",
    )

    spec = OnDisk(tolerance={"trajectory_rms_px": 1.5})
    assert spec.available("a-skill", "a-case") == (True, "")
    f = spec.build("a-skill", "a-case")

    assert f.kind == "curated" and f.label == "a-skill/a-case"
    assert set(f.data) == {"movie"} and set(f.truth) == {"offsets"}
    assert f.tolerance["trajectory_rms_px"] == 1.5
    assert "bead" in f.provenance and f.citation == CITATION
    assert isinstance(f.data["movie"], NpzRef)
    assert np.asarray(f.data["movie"]).shape == (2, 3, 3)


def test_an_on_disk_fixture_may_not_declare_a_key_as_both(tmp_path, monkeypatch):
    """A truth the run can see is not a truth. This is the one way a curated
    tree can be wrong that produces a *plausible* score rather than an error,
    so it is rejected at load rather than left to a reviewer."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    here = _write_tree(tmp_path, data={"movie": np.zeros((2, 2, 2))}, truth={})
    spec = json.loads((here / "case.json").read_text())
    spec["truth"] = {"movie": "arrays.npz"}
    (here / "case.json").write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="both data and truth"):
        OnDisk().build("a-skill", "a-case")


def test_data_that_is_not_here_is_unavailable_and_never_substituted(
    tmp_path, monkeypatch
):
    """The principle, as a mechanism. A curated case on a machine with no tree
    does not fall back to something else and does not quietly pass; it reports
    a reason and its run is skipped, the same discipline as a missing API key.
    """
    monkeypatch.delenv(FIXTURE_DIR_ENV, raising=False)
    usable, why = OnDisk().available("a-skill", "a-case")
    assert not usable and FIXTURE_DIR_ENV in why

    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    usable, why = OnDisk().available("a-skill", "a-case")
    assert not usable and "case.json" in why


def test_a_file_the_case_names_but_the_tree_lacks_is_unavailable_not_broken(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    here = _write_tree(tmp_path, data={"movie": np.zeros((2, 2))}, truth={})
    (here / "arrays.npz").unlink()

    usable, why = OnDisk().available("a-skill", "a-case")
    assert not usable and "arrays.npz" in why


def test_an_acquisition_the_manifest_does_not_record_does_not_run(
    tmp_path, monkeypatch
):
    """How an unreviewed acquisition would otherwise slip into a run: a
    directory appears in the tree, and nothing anywhere says where it came
    from or who it belongs to."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    _write_tree(tmp_path, data={"movie": np.zeros((2, 2))}, truth={})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 1, "fixtures": []}), encoding="utf-8"
    )

    usable, why = OnDisk().available("a-skill", "a-case")
    assert not usable and "manifest.json" in why


def test_real_data_without_a_citation_is_refused(tmp_path, monkeypatch):
    """ACDC ships a `MANDATORY_CITATION.md`. That obligation belongs to the
    harness rather than to whoever remembers it."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    _write_tree(
        tmp_path, data={"movie": np.zeros((2, 2))}, truth={}, entry={"citation": " "}
    )

    with pytest.raises(ValueError, match="citation"):
        OnDisk().build("a-skill", "a-case")


def test_data_that_is_not_what_the_manifest_records_is_refused(tmp_path, monkeypatch):
    """The last remaining way a case name could quietly denote two experiments:
    the file under this path not being the file the case was written against.
    Checked in-band because it is a header read, not a pass over the bytes."""
    monkeypatch.setenv(FIXTURE_DIR_ENV, str(tmp_path))
    here = _write_tree(tmp_path, data={"movie": np.zeros((2, 3, 3))}, truth={})
    np.savez(here / "arrays.npz", movie=np.zeros((2, 4, 4)))

    with pytest.raises(ValueError, match="not the data the case was written against"):
        OnDisk().build("a-skill", "a-case")


# --- identity --------------------------------------------------------------


def test_the_spec_stamps_the_identity_the_case_declares(tmp_path):
    """Identity is the `Case`'s, not the builder's. A builder that sets its own
    would be a second place for a case to be named, and the two would drift —
    with the report saying one thing and the artifact directory another."""
    built = Procedural(
        lambda: Fixture(
            provenance="p",
            data={},
            truth={},
            tolerance={},
            skill_id="whatever-the-builder-said",
            case_id="likewise",
        )
    ).build("a-skill", "a-case")

    assert built.label == "a-skill/a-case"
    assert built.kind == "synthetic"


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
