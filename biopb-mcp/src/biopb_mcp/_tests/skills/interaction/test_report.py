"""The engine's own tests: classification, flags, and the report.

Hermetic — no session, no model, no fixture build — and they run with the
ordinary suite. The report **is** this layer's deliverable, so the code that
writes it should not be exercised for the first time twenty minutes into a paid
run. Everything here is fed hand-built `Outcome`s, which is also the only way to
reach the corners that a real run reaches rarely and expensively: an arm that
died, one severed by a cap, one that left an unscorable array.

The metric columns are the part most worth pinning. They used to be three
hardcoded names; now they are read off whatever the verifier reported, because a
curated fixture and a synthetic one for the same skill support different
measurements and the table has to follow.

What a *case* has to be true of — its persona, its fixture, its verifier — is
`test_cases.py`. This file knows about one only as something to write a report
for.
"""

from __future__ import annotations

import json
import os

import pytest

from ...agentbench._conversation import (
    AGENT_TRUNCATED,
    FINISHED,
    RESPONDENT_FAILED,
    SILENT,
    TURN_CAP,
)
from ...agentbench._fixture import Attempt, Fixture, Metric, Outcome
from ...agentbench._models import ENV_FILE_ENV, reload_env_file
from . import _benchmark, conftest
from ._benchmark import (
    ARMS,
    ARMS_ENV,
    FLAG_CATALOG_MISMATCH,
    FLAG_CUT_OFF,
    FLAG_NEVER_ASKED,
    FLAG_OVER_BUDGET,
    FLAG_PEEKED,
    FLAG_UNANSWERED,
    GAVE_UP,
    HARNESS_ERROR,
    NO_RESULT,
    NO_SESSION,
    OK,
    OUT_OF_TURNS,
    UNSCORABLE,
    WRONG_ANSWER,
    Result,
    Run,
    catalog_size,
    selected_arms,
)
from .cases import CASES

FIXTURE = Fixture(
    skill_id="demo",
    case_id="demo-case",
    kind="synthetic",
    provenance="this test",
    data={},
    truth={},
    tolerance={},
    about="a stand-in, so nothing here depends on a real skill",
)


class FakeTrace:
    def __init__(self, stopped=FINISHED, turns=5, asked=(), tools=(), answers=None):
        self.stopped = stopped
        self.turns_used = turns
        self.questions = list(asked)
        self.blocking_questions = [q for q in asked if "?" in q]
        self.tool_names = list(tools)
        # Answered unless a test says otherwise: an unanswered question is the
        # exceptional case and should have to be asked for by name.
        self.answers = ["ok"] * len(asked) if answers is None else list(answers)


def outcome(*metrics: Metric, arrays=None) -> Outcome:
    return Outcome(
        fixture=FIXTURE,
        attempt=Attempt(subject="x", arrays=arrays or {}),
        metrics=metrics,
    )


def scored(value: float, limit: float = 1.0) -> Outcome:
    return outcome(Metric("err_px", value, limit), arrays={"corrected": object()})


def result(arm=ARMS[0], **kwargs) -> Result:
    kwargs.setdefault("trace", FakeTrace())
    kwargs.setdefault("catalog_hits", 1 if arm.skills else 0)
    return Result(arm=arm, **kwargs)


# --- classification --------------------------------------------------------


def test_a_run_within_tolerance_is_ok():
    assert result(outcome=scored(0.2)).classify() == (OK, "within every tolerance")


def test_a_run_outside_tolerance_names_the_metric_and_the_limit():
    verdict, reason = result(outcome=scored(3.0)).classify()
    assert verdict == WRONG_ANSWER
    assert "err_px 3 > 1" in reason


def test_a_cap_beats_a_bad_number():
    """A run severed mid-workflow may still have left a plausible array, and
    calling that a wrong answer blames the skill for a budget."""
    verdict, reason = result(
        trace=FakeTrace(stopped=TURN_CAP), outcome=outcome(Metric("err_px", None, 1.0))
    ).classify()
    assert verdict == OUT_OF_TURNS
    assert "turn cap" in reason


def test_an_agent_that_stopped_talking_is_not_a_wrong_answer():
    verdict, _ = result(
        trace=FakeTrace(stopped=SILENT), outcome=outcome(Metric("err_px", None, 1.0))
    ).classify()
    assert verdict == GAVE_UP


def test_leaving_nothing_and_leaving_something_unscorable_are_different():
    """Both score zero metrics and they point at different causes: one agent
    never got there, the other got there and produced the wrong shape."""
    nothing = result(outcome=outcome(Metric("err_px", None, 1.0))).classify()
    assert nothing[0] == NO_RESULT

    junk = result(
        outcome=outcome(
            Metric("err_px", None, 1.0, unavailable="corrected is (3,) not (24,2)"),
            arrays={"corrected": object()},
        )
    ).classify()
    assert junk[0] == UNSCORABLE
    assert "not (24,2)" in junk[1]


def test_an_arm_that_never_ran_carries_its_error():
    verdict, reason = Result(arm=ARMS[0], error="Boom: bang").classify()
    assert verdict == HARNESS_ERROR
    assert reason == "Boom: bang"


# --- flags -----------------------------------------------------------------


def test_asking_too_much_and_never_asking_are_both_flagged():
    over = result(trace=FakeTrace(asked=["a?", "b?", "c?", "d?"]), outcome=scored(0.1))
    assert f"{FLAG_OVER_BUDGET}(4)" in over.flags(budget=3)

    silent = result(trace=FakeTrace(asked=["let me look at this"]), outcome=scored(0.1))
    assert FLAG_NEVER_ASKED in silent.flags()


def test_a_scored_but_severed_run_says_so():
    cut = result(trace=FakeTrace(stopped=TURN_CAP), outcome=scored(0.1))
    assert FLAG_CUT_OFF in cut.flags()


def test_a_run_that_asked_and_was_never_answered_is_flagged():
    """An `asked` arm whose respondent never replied ran the `silent`
    condition under the `asked` label, so its row is not comparable to the
    thing it exists to be compared against."""
    unanswered = result(
        trace=FakeTrace(asked=["which channel is structural?"], answers=()),
        outcome=scored(0.1),
    )
    assert FLAG_UNANSWERED in unanswered.flags()

    answered = result(
        trace=FakeTrace(asked=["which channel is structural?"]), outcome=scored(0.1)
    )
    assert FLAG_UNANSWERED not in answered.flags()


def test_a_provider_failure_is_a_harness_error_not_an_agent_outcome():
    """Both of these look exactly like the agent finishing or giving up, and
    scoring them against the skill is how one broken respondent gets reported
    as four bad rows."""
    for stopped in (RESPONDENT_FAILED, AGENT_TRUNCATED):
        verdict, reason = result(
            trace=FakeTrace(stopped=stopped), outcome=scored(0.1)
        ).classify()
        assert verdict == HARNESS_ERROR, stopped
        assert reason, "a harness error has to say which one"


def test_the_catalog_flag_fires_in_both_directions():
    """The ablation reading wrong either way is the failure that makes a whole
    table meaningless, so it is noticed on the arm as well as asserted."""
    withheld_but_present = result(arm=ARMS[2], outcome=scored(0.1), catalog_hits=3)
    assert FLAG_CATALOG_MISMATCH in withheld_but_present.flags()

    offered_but_absent = result(arm=ARMS[0], outcome=scored(0.1), catalog_hits=0)
    assert FLAG_CATALOG_MISMATCH in offered_but_absent.flags()


def test_a_run_that_read_the_answer_key_says_so_on_its_own_row():
    """`execute_code` is arbitrary Python, so this is a thing that *can* happen
    and the layer's whole defence is that it cannot happen quietly. The number
    still gets computed — suppressing it would hide what was read — but the row
    carries the flag and the count."""
    peeked = result(
        arm=ARMS[0],
        outcome=scored(0.0001),
        peeked=("/x/biopb_mcp/_tests/skills/interaction/cases/drift_correction.py",),
    )
    assert any(f.startswith(FLAG_PEEKED) for f in peeked.flags()), peeked.flags()
    assert "(1)" in next(f for f in peeked.flags() if f.startswith(FLAG_PEEKED))

    clean = result(arm=ARMS[0], outcome=scored(0.0001))
    assert not any(f.startswith(FLAG_PEEKED) for f in clean.flags())


def test_an_arm_that_peeked_and_then_died_still_reports_it():
    """The flag is raised before the `no trace` guard: a run can read the
    fixture and then fall over, and that is exactly when the reason for a
    strange number matters most."""
    dead = Result(arm=ARMS[0], error="Boom", peeked=("/x/biopb_mcp/_tests/a.py",))
    assert any(f.startswith(FLAG_PEEKED) for f in dead.flags())


def test_a_failed_bring_up_leaves_the_process_environment_as_it_found_it(
    monkeypatch, tmp_path
):
    """Hermetic, and it guards a failure that would be invisible where it
    happened.

    `live_session` redirects `XDG_CONFIG_HOME` and the tensor URL for the whole
    process and undoes it in a `finally`. Anything that raises *before* that
    `try` leaves the redirect standing, so every later test in the process reads
    a temp config tree that has already been deleted — and fails somewhere with
    no connection to the cause. Staging the wheel is the step most likely to
    raise (it shells out to `uv build`), which is why it belongs inside.
    """
    from ...agentbench import _session

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "mine"))
    monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://example:1234")
    monkeypatch.setattr(_session, "why_unavailable", lambda: "")
    monkeypatch.setattr(
        _session,
        "staged_package",
        lambda: (_ for _ in ()).throw(_session.SessionUnavailable("no uv")),
    )

    with pytest.raises(_session.SessionUnavailable, match="no uv"):
        with _session.live_session():
            pass

    assert os.environ["XDG_CONFIG_HOME"] == str(tmp_path / "mine")
    assert os.environ["BIOPB_TENSOR_URL"] == "grpc://example:1234"
    assert _session.ENV_GUARD_LOG not in os.environ


def test_the_catalog_is_counted_from_what_the_tool_returned():
    """Whether the ablation took effect rests on this number, so it is parsed
    rather than pattern-counted — and an empty catalog and an unreadable one are
    not the same claim."""
    assert catalog_size(json.dumps([{"id": "a"}, {"id": "b"}])) == 2
    assert catalog_size("[]") == 0
    assert catalog_size("") == 0
    # A list return can reach a client wrapped in structured content.
    assert catalog_size(json.dumps({"result": [{"id": "a"}]})) == 1
    assert catalog_size(json.dumps({"result": []})) == 0
    # Not JSON at all: whatever this is, it is not evidence that the catalog was
    # withheld — and reading it as such would turn a broken ablation into a
    # clean-looking table.
    assert catalog_size("1 skill: drift-correction") == 1


# --- the report ------------------------------------------------------------


@pytest.fixture
def report(tmp_path, monkeypatch):
    """A whole run's report, written somewhere disposable."""
    monkeypatch.setattr(_benchmark, "artifact_root", lambda: tmp_path)
    case = CASES[0]
    run = Run(
        case=case,
        fixture=FIXTURE,
        results=[
            result(ARMS[0], outcome=scored(0.05)),
            result(ARMS[1], trace=FakeTrace(stopped=TURN_CAP), outcome=scored(9.0)),
            result(ARMS[2], outcome=outcome(Metric("err_px", None, 1.0))),
            Result(arm=ARMS[3], error="RuntimeError: boom"),
        ],
    )
    text = run.summary()
    where = _benchmark.where_for(case)
    return text, json.loads((where / "summary.json").read_text()), where


def test_the_columns_come_from_the_metrics_that_were_reported(report):
    text, _, _ = report
    assert "| err_px |" in text
    assert "err_px ≤ 1" in text


def test_every_arm_reaches_the_table_including_the_one_that_died(report):
    text, data, _ = report
    for arm in ARMS:
        assert f"`{arm.name}`" in text
    assert [row["arm"] for row in data["arms"]] == [arm.name for arm in ARMS]
    assert data["arms"][3]["outcome"] == HARNESS_ERROR


def test_a_metric_no_arm_could_produce_reads_as_absent_not_as_zero(report):
    """`—`, not a number: an arm that left nothing must never be readable as
    having scored well, which is the same rule `Outcome.passed` enforces."""
    _, data, _ = report
    assert data["arms"][2]["metrics"]["err_px"] is None


def test_a_row_says_how_long_its_arm_took(report):
    """The only cost signal a reader gets afterwards. An arm is minutes, and
    "was this twenty minutes or ninety" is not recoverable from the transcript."""
    text, data, _ = report
    assert all("seconds" in row for row in data["arms"])
    assert "| min |" in text


def test_the_report_names_both_models_and_the_fixture(report):
    """A row is uninterpretable a week later without them."""
    text, data, _ = report
    assert data["agent"] and data["respondent"]
    assert FIXTURE.case_id in text
    assert FIXTURE.about in text


def test_the_report_lands_under_its_own_case(report):
    """Under `<skill>/<case_id>`, so neither a second skill nor a second case
    for the same skill overwrites the first — the thing that made the old
    single-directory-per-skill layout a problem the moment there were two."""
    _, _, where = report
    assert where.name == CASES[0].case_id
    assert where.parent.name == CASES[0].skill
    assert (where / "summary.md").is_file()


# --- which corners get paid for --------------------------------------------


@pytest.fixture
def arms_env(tmp_path, monkeypatch):
    """`BIOPB_SKILL_ARMS` from the environment and nowhere else.

    `setting()` falls back to the dotenv, so a developer who put the variable in
    their `.env` would otherwise decide the default test's answer.
    """
    monkeypatch.delenv(ARMS_ENV, raising=False)
    monkeypatch.setenv(ENV_FILE_ENV, str(tmp_path / "absent.env"))
    reload_env_file()
    yield monkeypatch
    monkeypatch.undo()
    reload_env_file()


def test_the_default_is_the_whole_square(arms_env):
    assert selected_arms() == ARMS


def test_asked_drops_exactly_the_silent_arms(arms_env):
    """The point of the option: the skill's delta is the two `+asked` corners,
    so the other two are droppable without touching what the layer measures."""
    arms_env.setenv(ARMS_ENV, "asked")
    chosen = selected_arms()
    assert [a.name for a in chosen] == ["skill+asked", "noskill+asked"]
    assert {a.skills for a in chosen} == {True, False}  # the delta survives


def test_an_unknown_selection_raises_rather_than_running_everything(arms_env):
    """A typo that silently spent the full square would be found by looking at
    the clock, twenty minutes per case later."""
    arms_env.setenv(ARMS_ENV, "askd")
    with pytest.raises(ValueError, match="askd"):
        selected_arms()


def test_a_partial_report_names_what_it_did_not_run(tmp_path, monkeypatch):
    """A two-row table is otherwise indistinguishable from a 2x2 whose other
    corners died, and the missing rows are the fixture's, not the skill's."""
    monkeypatch.setattr(_benchmark, "artifact_root", lambda: tmp_path)
    asked = [a for a in ARMS if a.asked]
    text = Run(
        case=CASES[0],
        fixture=FIXTURE,
        results=[result(a, outcome=scored(0.05)) for a in asked],
    ).summary()
    data = json.loads((_benchmark.where_for(CASES[0]) / "summary.json").read_text())

    assert "2 of 4 arms" in text
    assert "`skill+silent`" in text and "`noskill+silent`" in text
    assert data["arms_not_run"] == ["skill+silent", "noskill+silent"]
    # The reading guide must not send anyone to a row that is not there.
    assert "skill+asked vs skill+silent" not in text


def test_a_full_report_says_nothing_about_skipping(report):
    text, data, _ = report
    assert data["arms_not_run"] == []
    assert "arms** — not run" not in text


# --- a run that never started ----------------------------------------------


def test_a_run_that_never_got_a_session_is_distinguishable(tmp_path):
    """It means the machine, not the skill, and it is the one shape that should
    skip rather than be reported as four failed arms."""
    dead = Run(
        case=CASES[0],
        fixture=FIXTURE,
        results=[Result(arm=a, error=f"{NO_SESSION}no display") for a in ARMS],
    )
    assert dead.failed_to_start
    assert not Run(
        case=CASES[0],
        fixture=FIXTURE,
        results=[result(a, outcome=scored(0.1)) for a in ARMS],
    ).failed_to_start


# --- smoke runs first, and gates ------------------------------------------


class FakeItem:
    """Just enough of a pytest item for the collection hook."""

    def __init__(self, filename: str, where=None):
        self.path = (where or conftest.HERE) / filename
        self.nodeid = f"{filename}::a_test"


def test_the_smoke_tests_are_moved_to_the_front():
    """Alphabetically `test_benchmark` sorts first, so four paid conversations
    went out before anything checked the stack could hold a napari layer."""
    items = [FakeItem("test_benchmark.py"), FakeItem(conftest.SMOKE)]
    conftest.pytest_collection_modifyitems(items)
    assert [i.path.name for i in items] == [conftest.SMOKE, "test_benchmark.py"]


def test_only_this_directory_is_reordered(tmp_path):
    """The hook is handed every item in the run, so a directory-level conftest
    that re-sorted all of them would silently rearrange the rest of the suite."""
    outsider = FakeItem("test_zzz_elsewhere.py", where=tmp_path)
    items = [outsider, FakeItem("test_benchmark.py"), FakeItem(conftest.SMOKE)]
    conftest.pytest_collection_modifyitems(items)
    assert items[0] is outsider
    assert [i.path.name for i in items[1:]] == [conftest.SMOKE, "test_benchmark.py"]


def test_a_failed_smoke_test_is_recorded_and_a_skipped_one_is_not(monkeypatch):
    """Ordering alone gates nothing without `-x`; the benchmark reads this list
    and refuses to spend. A skip is not a failure — no display is reported by
    `unavailable()`, with better instructions."""
    monkeypatch.setattr(conftest, "_SMOKE_FAILURES", [])

    class Report:
        def __init__(self, nodeid, failed):
            self.nodeid, self.failed = nodeid, failed

    conftest.pytest_runtest_logreport(Report(f"{conftest.SMOKE}::boom", True))
    conftest.pytest_runtest_logreport(Report(f"{conftest.SMOKE}::skipped", False))
    conftest.pytest_runtest_logreport(Report("test_benchmark.py::other", True))
    assert conftest.smoke_failures() == [f"{conftest.SMOKE}::boom"]


def test_the_arm_sets_partition_the_square(monkeypatch):
    """`asked` and `silent` are complements, and together they are `all` — so
    two half-price runs cover the same ground as one full one."""
    monkeypatch.setenv(ARMS_ENV, "asked")
    _benchmark._dotenv_reset = None
    asked = set(selected_arms())
    monkeypatch.setenv(ARMS_ENV, "silent")
    silent = set(selected_arms())

    assert asked and silent
    assert not (asked & silent)
    assert asked | silent == set(ARMS)
