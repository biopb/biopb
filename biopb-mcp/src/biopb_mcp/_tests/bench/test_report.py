"""The engine's own tests: options, classification, flags, and the report.

Hermetic — no session, no model, no fixture build — and they run with the
ordinary suite. The report **is** this layer's deliverable, so the code that
writes it should not be exercised for the first time twenty minutes into a paid
run. Everything here is fed hand-built `Outcome`s, which is also the only way to
reach the outcomes a real run reaches rarely and expensively: a sample that
died, one severed by a cap, one that left an unscorable array.

The metric columns are the part most worth pinning. They used to be three
hardcoded names; now they are read off whatever the verifier reported, because a
curated fixture and a synthetic one for the same subject support different
measurements and the table has to follow.

The run options get the same treatment for the same reason: `--bench-skills`
and `--bench-cases` decide what an invocation spends *and what its number
means*, and a switch that silently did not take effect is indistinguishable
from one that did until two sessions are compared.

What a *case* has to be true of — its persona, its fixture, its verifier — is
`test_cases.py`. This file knows about one only as something to write a report
for.
"""

from __future__ import annotations

import json
import os

import pytest

from ..agentbench._conversation import (
    AGENT_TRUNCATED,
    FINISHED,
    RESPONDENT_FAILED,
    SILENT,
    TURN_CAP,
)
from ..agentbench._fixture import Attempt, Fixture, Metric, Outcome
from . import _engine, conftest
from ._engine import (
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
    catalog_ids,
    models_in_play,
    respondent_for,
    select,
    session_dir,
    session_id,
    task_for,
    unavailable,
    write_session,
)
from ._options import (
    CASES as CASES_OPTION,
    FIXTURES,
    RESPONDER,
    SAMPLES_DEST,
    SAMPLES_ENV,
    SKILLS,
    BadOption,
    Options,
    Setting,
    resolve,
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
    about="a stand-in, so nothing here depends on a real case",
)

SKILL_CASE = next(c for c in CASES if c.about_a_skill)
TASK_CASE = next(c for c in CASES if not c.about_a_skill)


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


def result(sample: int = 1, skills: bool = True, **kwargs) -> Result:
    kwargs.setdefault("trace", FakeTrace())
    kwargs.setdefault("catalog", ("a-skill",) if skills else ())
    return Result(sample=sample, skills_offered=skills, **kwargs)


# --- the run options -------------------------------------------------------


class FakeConfig:
    """Just enough of pytest's `Config` to answer `getoption`."""

    def __init__(self, **flags):
        self._flags = flags

    def getoption(self, name, default=None):
        return self._flags.get(name, default)


@pytest.fixture
def clean_env(monkeypatch):
    """No `BIOPB_BENCH_*` from the developer's shell. There is deliberately no
    dotenv behind these — a file somebody forgot about should not decide what a
    run spends — so clearing the environment is the whole isolation."""
    for setting in (CASES_OPTION, FIXTURES, SKILLS, RESPONDER):
        monkeypatch.delenv(setting.env, raising=False)
    monkeypatch.delenv(SAMPLES_ENV, raising=False)
    return monkeypatch


def test_the_defaults_run_everything_once_in_the_shipped_configuration(clean_env):
    chosen = resolve(FakeConfig())
    assert (chosen.cases, chosen.fixtures) == ("all", "all")
    assert chosen.skills is True and chosen.responder == "model"
    assert chosen.samples == 1
    assert not chosen.filtered


def test_a_flag_beats_the_environment(clean_env):
    """The direction that makes a surprising bill easy to explain: what is on
    the command line is what ran."""
    clean_env.setenv(RESPONDER.env, "silent")
    assert resolve(FakeConfig(**{RESPONDER.dest: "model"})).responder == "model"
    assert resolve(FakeConfig()).responder == "silent"


def test_an_unknown_value_in_the_environment_raises(clean_env):
    """A typo that silently ran the larger thing would be found by looking at
    the clock, twenty minutes per case later. The flag form cannot reach here —
    argparse rejects it against the same list."""
    clean_env.setenv(RESPONDER.env, "modle")
    with pytest.raises(BadOption, match="modle"):
        resolve(FakeConfig())


def test_a_bad_option_stops_the_run_before_anything_is_collected(clean_env):
    """As pytest's own usage error, from the tests-root conftest. Left to
    collection it would arrive as one traceback per module in this directory —
    five of them, for a typo in one environment variable."""
    from .. import conftest as tests_root

    clean_env.setenv(RESPONDER.env, "modle")
    with pytest.raises(pytest.UsageError, match="modle"):
        tests_root.pytest_configure(FakeConfig())


@pytest.mark.parametrize("raw", ["0", "-2", "nonsense", "1.5"])
def test_a_sample_count_that_is_not_a_count_raises(clean_env, raw):
    """Rather than clamping to 1. Asking for zero samples is not a request for
    one; it is someone expecting a run they will not get."""
    with pytest.raises(BadOption):
        resolve(FakeConfig(**{SAMPLES_DEST: raw}))


def test_a_sample_count_is_read_from_the_environment_too(clean_env):
    clean_env.setenv(SAMPLES_ENV, "3")
    assert resolve(FakeConfig()).samples == 3


def test_every_responder_the_options_offer_is_one_the_engine_can_build(monkeypatch):
    """`_options.py` is stdlib-only and imports nothing from this package — it
    has to be, because it answers `pytest_addoption` from the tests-root
    conftest, which every run in the repo loads. The cost is that the values a
    flag offers live in one file and the dispatch in another, so they are
    pinned rather than remembered: a value the flag accepts and the engine
    silently ignores would run the wrong condition under the right label."""
    built = []
    monkeypatch.setattr(_engine, "model_respondent", lambda persona: ("model", persona))
    for value in RESPONDER.values:
        built.append(respondent_for(SKILL_CASE, Options(responder=value)))
    assert built[RESPONDER.values.index("silent")].name == "silent"
    assert built[RESPONDER.values.index("briefed")].name == "briefed"
    assert built[RESPONDER.values.index("model")][0] == "model"
    assert len(built) == len(RESPONDER.values)


def test_a_filter_says_it_is_a_filter(clean_env):
    """`filtered` is what makes the terminal summary print. A narrowed run that
    does not say so reads afterwards exactly like a complete one."""
    assert not Options().filtered
    assert Options(cases="skills").filtered
    assert Options(fixtures="curated").filtered
    # The switches and the sample count change what a run measures and how
    # deep it goes, not how much of the catalogue it covers, and all three are
    # already named in the report's own header and in `session.json`.
    assert not Options(skills=False, responder="silent", samples=4).filtered


def test_the_option_line_names_every_option(clean_env):
    """Including the ones left alone: a header listing only what was changed
    cannot be read as a record of what was run."""
    line = Options(cases="skills", samples=2).describe()
    for token in (
        "cases=skills",
        "fixtures=all",
        "skills=true",
        "responder=model",
        "samples=2",
    ):
        assert token in line


def test_a_setting_derives_its_own_argparse_destination():
    assert Setting("--bench-cases", "X", (), "", "").dest == "bench_cases"


# --- which cases a run pays for --------------------------------------------


def test_selecting_skills_drops_the_cases_that_have_none():
    chosen = select(CASES, Options(cases="skills"))
    assert chosen and all(c.about_a_skill for c in chosen)
    assert len(chosen) < len(CASES), "this tree has no case without a skill"


def test_selecting_tasks_is_the_complement():
    # By label: a `Case` is frozen but not hashable — `collect` and the
    # persona's fact table are mappings — so the set algebra goes through the
    # thing that identifies a case anyway.
    skills = {c.label for c in select(CASES, Options(cases="skills"))}
    tasks = {c.label for c in select(CASES, Options(cases="tasks"))}
    assert tasks
    assert not (skills & tasks)
    assert skills | tasks == {c.label for c in CASES}


def test_selecting_a_fixture_kind_keeps_only_that_kind():
    for kind in ("synthetic", "curated"):
        chosen = select(CASES, Options(fixtures=kind))
        assert chosen, f"nothing in this tree is {kind}"
        assert all(c.fixture.kind == kind for c in chosen)


def test_the_filters_compose():
    chosen = select(CASES, Options(cases="skills", fixtures="synthetic"))
    assert chosen, "this tree has no synthetic skill case, so this proves nothing"
    assert all(c.about_a_skill and c.fixture.kind == "synthetic" for c in chosen)


def test_composing_filters_may_legitimately_select_nothing():
    """And must say so by being empty rather than by falling back.

    Every curated case in the tree happens to name no skill, so this pair is
    satisfied by nothing — which is a real answer to a real question, and the
    terminal summary is what makes it legible. An `all()` over an empty tuple
    is *also* how the test above would look if the filters had stopped working
    entirely, which is why that one asserts it selected something first.
    """
    assert select(CASES, Options(cases="skills", fixtures="curated")) == ()


def test_selection_keeps_the_order_it_was_given():
    """The report directory is per case, but the *run order* is what somebody
    watching a two-hour invocation is reading against."""
    chosen = select(CASES, Options())
    assert list(chosen) == list(CASES)


# --- the switches ----------------------------------------------------------


def test_the_switches_are_what_a_session_is(clean_env):
    """One invocation, one configuration, and it is two values wide. There is
    no per-case arm set any more: what a run does no longer depends on what
    kind of case it is looking at."""
    default = resolve(FakeConfig())
    assert default.configuration == "skills=on responder=model"

    ablated = resolve(FakeConfig(**{SKILLS.dest: "false", RESPONDER.dest: "silent"}))
    assert ablated.skills is False
    assert ablated.configuration == "skills=off responder=silent"


def test_the_skills_switch_is_a_bool_and_not_the_string_false(clean_env):
    """`--bench-skills=false` arrives from argparse as a string, and a string
    is truthy. It reaches `live_session(skills_enabled=...)`, so getting this
    wrong runs the un-ablated session and reports it as the ablation."""
    assert resolve(FakeConfig(**{SKILLS.dest: "false"})).skills is False
    assert resolve(FakeConfig(**{SKILLS.dest: "true"})).skills is True


def test_the_responder_switch_reaches_the_dispatch(monkeypatch):
    monkeypatch.setattr(_engine, "model_respondent", lambda persona: "a-model")
    assert respondent_for(SKILL_CASE, Options(responder="silent")).name == "silent"
    assert respondent_for(SKILL_CASE, Options(responder="briefed")).name == "briefed"
    assert respondent_for(SKILL_CASE, Options(responder="model")) == "a-model"


def test_the_same_switches_apply_to_a_case_of_either_kind(monkeypatch):
    """The last asymmetry gone. A skill case used to get four configurations
    and every other case one, so a case's kind decided what a run cost."""
    monkeypatch.setattr(_engine, "model_respondent", lambda persona: "a-model")
    for case in (SKILL_CASE, TASK_CASE):
        assert respondent_for(case, Options(responder="silent")).name == "silent"
        assert respondent_for(case, Options(responder="briefed")).name == "briefed"
        assert respondent_for(case, Options(responder="model")) == "a-model"


# --- what a briefed run is handed -------------------------------------------


def test_only_a_briefed_run_is_handed_the_persona_s_facts():
    """The half of `--bench-responder=briefed` the respondent cannot do.

    Every other configuration makes the fact something to ask for or go
    without, so the prompt has to be the case's own text and nothing else — a
    briefing that leaked into a `model` run would hand over the answer to the
    question that run exists to measure being asked.
    """
    spoken = task_for(SKILL_CASE, Options(responder="model"))
    silent = task_for(SKILL_CASE, Options(responder="silent"))
    briefed = task_for(SKILL_CASE, Options(responder="briefed"))

    assert spoken == SKILL_CASE.task
    assert silent == SKILL_CASE.task
    assert briefed.startswith(SKILL_CASE.task.rstrip())
    for value in SKILL_CASE.persona.facts.values():
        assert value not in spoken
        assert value in briefed


def test_a_briefed_prompt_countermands_the_task_s_offer_of_a_person():
    """Every case's task says the microscopist is there and how to reach them,
    because that is the condition every other switch value runs. The brief has
    to cancel it explicitly and *after* it — a prompt holding both would leave
    a run reasonably spending turns waiting on somebody who is not coming."""
    case = SKILL_CASE
    # Whitespace-normalised: where the header happens to wrap is formatting,
    # and a test that reads the prompt line by line fails on a reflow that
    # changed nothing about what it says.
    flowed = " ".join(task_for(case, Options(responder="briefed")).split())
    assert "not available for this session" in flowed
    assert flowed.index("not available") > flowed.index(
        " ".join(case.task.split())[-40:]
    )


def test_a_brief_hands_over_the_facts_and_not_the_persona_s_own_instructions():
    """`background` is written in the second person to whoever plays the part
    — "you acquired these fields, you are happy to answer questions" — so
    including it would tell the agent it ran the experiment and re-offer the
    conversation this switch exists to remove."""
    with_background = next(c for c in CASES if c.persona.background)
    assert with_background.persona.background not in with_background.persona.briefing()


def test_a_briefed_run_and_a_spoken_one_differ_by_nothing_but_the_asking():
    """The pair only measures the cost of eliciting if the *information* is the
    same on both sides. It is rendered from one fact table for that reason, and
    this is the assertion that the rendering did not drop any of it."""
    briefing = SKILL_CASE.persona.briefing()
    for key, value in SKILL_CASE.persona.facts.items():
        assert key in briefing, f"{key!r} is missing from the brief"
        assert value in briefing


# --- what the configuration needs to exist ----------------------------------


class FakeChoice:
    """A model the availability checks can be pointed at."""

    def __init__(self, name, why=""):
        self.name, self.why, self.base_url = name, why, ""

    def why_unavailable(self):
        return self.why


class RunnableAnywhere:
    """A case with nothing of its own to be unavailable about.

    So the only thing left for `unavailable()` to object to is the models —
    which is the whole subject here.
    """

    layers = ()
    about_a_skill = False

    def available(self):
        return True, ""


@pytest.fixture
def only_the_models(monkeypatch):
    """Everything that is not a model check answers yes, and reachability is
    counted rather than performed."""
    from ..agentbench import _session

    probed = []
    monkeypatch.setattr(_session, "why_unavailable", lambda: "")
    monkeypatch.setattr(_engine, "text_backend", lambda choice: choice)
    monkeypatch.setattr(
        _engine, "reachable", lambda choice: probed.append(choice) or ""
    )
    monkeypatch.setattr(_engine, "agent_choice", lambda: FakeChoice("the-agent"))
    return probed


@pytest.mark.parametrize("local", ["silent", "briefed"])
def test_a_session_with_a_local_respondent_needs_no_respondent_model(
    only_the_models, monkeypatch, local
):
    """Both answer from a constant, so neither run reaches for a respondent at
    all.

    Demanding one anyway skipped **every case** on a machine holding one key —
    the ordinary way to run these conditions — and reported it as an
    environment problem with the cases.
    """
    monkeypatch.setattr(
        _engine, "respondent_choice", lambda: FakeChoice("the-respondent", "no key")
    )
    case = RunnableAnywhere()

    assert unavailable(case, Options(responder=local)) == ""
    assert unavailable(case, Options(responder="model")) == "respondent: no key"


@pytest.mark.parametrize("local", ["silent", "briefed"])
def test_a_local_respondent_spends_no_request_proving_a_model_it_will_not_call(
    only_the_models, monkeypatch, local
):
    """The reachability probe is the one check here that costs money, and a
    respondent endpoint is not on the path of either local condition."""
    monkeypatch.setattr(
        _engine, "respondent_choice", lambda: FakeChoice("the-respondent")
    )
    case = RunnableAnywhere()

    assert unavailable(case, Options(responder=local)) == ""
    assert [c.name for c in only_the_models] == ["the-agent"]

    only_the_models.clear()
    assert unavailable(case, Options(responder="model")) == ""
    assert [c.name for c in only_the_models] == ["the-agent", "the-respondent"]


def test_the_models_in_play_are_the_ones_the_switch_selects(monkeypatch):
    """One list, read by both the key check and the reachability probe, so the
    two cannot disagree about who this session talks to."""
    monkeypatch.setattr(_engine, "agent_choice", lambda: FakeChoice("the-agent"))
    monkeypatch.setattr(
        _engine, "respondent_choice", lambda: FakeChoice("the-respondent")
    )
    silent = [side for side, _ in models_in_play(Options(responder="silent"))]
    briefed = [side for side, _ in models_in_play(Options(responder="briefed"))]
    spoken = [side for side, _ in models_in_play(Options(responder="model"))]
    assert silent == ["agent"]
    assert briefed == ["agent"]
    assert spoken == ["agent", "respondent"]


# --- classification --------------------------------------------------------


def test_a_run_within_tolerance_is_ok():
    assert result(outcome=scored(0.2)).classify() == (OK, "within every tolerance")


def test_a_run_outside_tolerance_names_the_metric_and_the_limit():
    verdict, reason = result(outcome=scored(3.0)).classify()
    assert verdict == WRONG_ANSWER
    assert "err_px 3 > 1" in reason


def test_a_cap_beats_a_bad_number():
    """A run severed mid-workflow may still have left a plausible array, and
    calling that a wrong answer blames the case for a budget."""
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


def test_a_run_that_never_ran_carries_its_error():
    verdict, reason = Result(error="Boom: bang").classify()
    assert verdict == HARNESS_ERROR
    assert reason == "Boom: bang"


def test_a_provider_failure_is_a_harness_error_not_an_agent_outcome():
    """Both of these look exactly like the agent finishing or giving up, and
    scoring them against the case is how one broken respondent gets reported
    as four bad rows."""
    for stopped in (RESPONDENT_FAILED, AGENT_TRUNCATED):
        verdict, reason = result(
            trace=FakeTrace(stopped=stopped), outcome=scored(0.1)
        ).classify()
        assert verdict == HARNESS_ERROR, stopped
        assert reason, "a harness error has to say which one"


# --- flags -----------------------------------------------------------------


def test_asking_too_much_and_never_asking_are_both_flagged():
    over = result(trace=FakeTrace(asked=["a?", "b?", "c?", "d?"]), outcome=scored(0.1))
    assert f"{FLAG_OVER_BUDGET}(4)" in over.flags(budget=3)

    silent = result(trace=FakeTrace(asked=["let me look at this"]), outcome=scored(0.1))
    assert FLAG_NEVER_ASKED in silent.flags()


def test_a_briefed_run_is_not_flagged_for_asking_nothing():
    """It was handed everything and there is nobody to ask, so not asking is
    the switch working rather than something to notice about the agent — and a
    flag on every row of a session says nothing about any of them."""
    briefed = result(
        responder="briefed",
        trace=FakeTrace(asked=["here is what I am doing"]),
        outcome=scored(0.1),
    )
    assert FLAG_NEVER_ASKED not in briefed.flags()

    # The other half: it is still worth seeing when a briefed run went asking
    # anyway, which is the agent not using what it was given.
    chatty = result(
        responder="briefed",
        trace=FakeTrace(asked=["a?", "b?", "c?", "d?"]),
        outcome=scored(0.1),
    )
    assert f"{FLAG_OVER_BUDGET}(4)" in chatty.flags(budget=3)


def test_a_scored_but_severed_run_says_so():
    cut = result(trace=FakeTrace(stopped=TURN_CAP), outcome=scored(0.1))
    assert FLAG_CUT_OFF in cut.flags()


def test_a_run_that_asked_and_was_never_answered_is_flagged():
    """A `--bench-responder=model` session whose respondent never replied ran
    the `silent` condition under the other label, so its report is not
    comparable to the thing it exists to be compared against."""
    unanswered = result(
        trace=FakeTrace(asked=["which channel is structural?"], answers=()),
        outcome=scored(0.1),
    )
    assert FLAG_UNANSWERED in unanswered.flags()

    answered = result(
        trace=FakeTrace(asked=["which channel is structural?"]), outcome=scored(0.1)
    )
    assert FLAG_UNANSWERED not in answered.flags()


def test_the_catalog_flag_fires_in_both_directions():
    """The switch reading wrong either way is the failure that makes a whole
    report meaningless, so it is noticed on the row as well as asserted."""
    withheld_but_present = result(
        skills=False, outcome=scored(0.1), catalog=("a", "b", "c")
    )
    assert FLAG_CATALOG_MISMATCH in withheld_but_present.flags()

    offered_but_absent = result(skills=True, outcome=scored(0.1), catalog=())
    assert FLAG_CATALOG_MISMATCH in offered_but_absent.flags()


def test_a_run_that_read_the_answer_key_says_so_on_its_own_row():
    """`execute_code` is arbitrary Python, so this is a thing that *can* happen
    and the layer's whole defence is that it cannot happen quietly. The number
    still gets computed — suppressing it would hide what was read — but the row
    carries the flag and the count."""
    peeked = result(
        outcome=scored(0.0001),
        peeked=("/x/biopb_mcp/_tests/bench/cases/drift_correction.py",),
    )
    assert any(f.startswith(FLAG_PEEKED) for f in peeked.flags()), peeked.flags()
    assert "(1)" in next(f for f in peeked.flags() if f.startswith(FLAG_PEEKED))

    clean = result(outcome=scored(0.0001))
    assert not any(f.startswith(FLAG_PEEKED) for f in clean.flags())


def test_a_sample_that_peeked_and_then_died_still_reports_it():
    """The flag is raised before the `no trace` guard: a run can read the
    fixture and then fall over, and that is exactly when the reason for a
    strange number matters most."""
    dead = Result(error="Boom", peeked=("/x/biopb_mcp/_tests/a.py",))
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
    from ..agentbench import _session

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


def test_the_catalog_is_read_from_what_the_tool_returned():
    """Whether `--bench-skills` took effect rests on this, so it is parsed
    rather than pattern-counted — and an empty catalog and an unreadable one
    are not the same claim. Ids rather than a count, because the list itself is
    the provenance a later release is compared against."""
    assert catalog_ids(json.dumps([{"id": "a"}, {"id": "b"}])) == ("a", "b")
    assert catalog_ids("[]") == ()
    assert catalog_ids("") == ()
    # A list return can reach a client wrapped in structured content.
    assert catalog_ids(json.dumps({"result": [{"id": "a"}]})) == ("a",)
    assert catalog_ids(json.dumps({"result": []})) == ()
    # Not JSON at all: whatever this is, it is not evidence that the catalog was
    # withheld — and reading it as such would turn a switch that never took
    # effect into a clean-looking report.
    assert catalog_ids("1 skill: drift-correction") == ("<unparseable>",)


def test_the_catalog_reads_the_two_shapes_the_tool_actually_returns():
    """The shapes that reached it in practice, both of which it got wrong.

    The tool answers with one content block per skill and the client joins
    them, so the text is a *stream* of JSON values rather than one document —
    `{...}{...}` for two matches, which `json.loads` rejects. Filed as
    unreadable, it still counted as "something", so the switch check stayed
    green while the provenance line said `<unparseable>` for most of the
    catalogue.

    A lone match is worse, because it parses. One skill is a dict whose only
    list is `tags`, and hunting for the entries by type reported *those* as the
    catalog — every "Skills the catalog offered" line ever written was a tag
    list. The identifying keys are checked first for exactly that reason.
    """
    one = json.dumps({"id": "flatfield", "tags": ["illumination", "correction"]})
    two = json.dumps({"id": "stitch-tiles", "tags": ["mosaic"]})

    assert catalog_ids(one) == ("flatfield",)
    assert catalog_ids(one + two) == ("flatfield", "stitch-tiles")
    assert catalog_ids(f"{one}\n{two}") == ("flatfield", "stitch-tiles")


# --- the report ------------------------------------------------------------


@pytest.fixture
def report(tmp_path, monkeypatch):
    """One case's report, four samples of one configuration, somewhere
    disposable."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    run = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        options=Options(samples=4),
        results=[
            result(sample=1, outcome=scored(0.05)),
            result(sample=2, trace=FakeTrace(stopped=TURN_CAP), outcome=scored(9.0)),
            result(sample=3, outcome=outcome(Metric("err_px", None, 1.0))),
            Result(sample=4, error="RuntimeError: boom"),
        ],
    )
    text = run.summary()
    where = _engine.where_for(SKILL_CASE)
    return text, json.loads((where / "summary.json").read_text()), where


def _table(text: str) -> list[list[str]]:
    """The rows of the markdown table, as cells."""
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith(("| ", "|---"))
    ]


def test_the_columns_come_from_the_metrics_that_were_reported(report):
    text, _, _ = report
    assert "| err_px |" in text
    assert "err_px ≤ 1" in text


def test_every_row_of_the_table_is_as_wide_as_its_rule(report):
    """A markdown renderer takes its column count from the `|---|` rule and
    discards whatever a header has past it. So a width the header and the rule
    disagree about is not a cosmetic defect — it silently deletes a column."""
    text = report[0]
    assert len({len(row) for row in _table(text)}) == 1, f"ragged table:\n{text}"


def test_a_run_where_no_sample_scored_still_renders_its_reasons(tmp_path, monkeypatch):
    """The report most worth reading, and the one the table shape broke on.

    With no outcome anywhere there are no metric columns, and the table used to
    be a fixed skeleton with a joined metric string concatenated into it — so an
    empty join left a phantom cell, the header came out one wider than its rule,
    and the column a renderer drops is the last one: `reason`, which on an
    all-dead run is the only column with anything in it.
    """
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    text = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        options=Options(samples=2),
        results=[
            Result(sample=1, error="RuntimeError: boom"),
            Result(sample=2, error="SessionUnavailable: no display"),
        ],
    ).summary()

    rows = _table(text)
    assert {len(row) for row in rows} == {7}, f"ragged table:\n{text}"
    assert rows[0] == [
        "sample",
        "outcome",
        "turns",
        "asked",
        "tools",
        "min",
        "reason",
    ]
    # The point of the whole report in this state.
    assert rows[2][-1] == "RuntimeError: boom"
    assert rows[3][-1] == "SessionUnavailable: no display"
    # And the header line says so rather than trailing off after the colon.
    assert "Tolerances: none — no sample produced a metric" in text


def test_every_sample_reaches_the_table_including_the_one_that_died(report):
    text, data, _ = report
    assert [row["sample"] for row in data["samples"]] == [1, 2, 3, 4]
    assert data["samples"][3]["outcome"] == HARNESS_ERROR
    assert "RuntimeError: boom" in text


def test_a_metric_no_sample_could_produce_reads_as_absent_not_as_zero(report):
    """`—`, not a number: a sample that left nothing must never be readable as
    having scored well, which is the same rule `Outcome.passed` enforces."""
    _, data, _ = report
    assert data["samples"][2]["metrics"]["err_px"] is None


def test_a_row_says_how_long_its_sample_took(report):
    """The only cost signal a reader gets afterwards. A sample is minutes, and
    "was this twenty minutes or ninety" is not recoverable from the transcript."""
    text, data, _ = report
    assert all("seconds" in row for row in data["samples"])
    assert "| min |" in text


def test_the_report_names_both_models_and_the_fixture(report):
    """A row is uninterpretable a week later without them."""
    text, data, _ = report
    assert data["agent"] and data["respondent"]
    assert FIXTURE.case_id in text
    assert FIXTURE.about in text


def test_the_report_states_the_configuration_it_ran_under(report):
    """The header carries what a column used to. With one configuration per
    invocation, a reader who cannot see the switches cannot tell an ablation
    from the shipped thing — and the two reports look identical otherwise."""
    text, data, _ = report
    assert "Configuration: **skills=on responder=model**" in text
    assert data["configuration"] == "skills=on responder=model"
    assert data["options"]["skills"] is True
    assert data["options"]["responder"] == "model"


@pytest.mark.parametrize("local", ["silent", "briefed"])
def test_a_local_respondent_report_names_no_model_in_either_form(
    tmp_path, monkeypatch, local
):
    """`summary.md` and `summary.json` are one report in two forms, and a
    reader who compares two sessions reads whichever is to hand.

    The JSON used to stamp `respondent_choice().name` unconditionally, so a
    silent run's markdown said "silent" while its JSON named a model that had
    answered nothing — the machine-readable half asserting the opposite of the
    configuration, on exactly the arm a delta is measured against.
    """
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(
        _engine, "respondent_choice", lambda: FakeChoice("a-real-model")
    )
    run = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        options=Options(responder=local),
        results=[result(responder=local, outcome=scored(0.2))],
    )
    text = run.summary()
    data = json.loads((_engine.where_for(SKILL_CASE) / "summary.json").read_text())

    assert f"Respondent: **{local}**" in text
    assert data["respondent"] == local
    assert "a-real-model" not in text


def test_an_ablated_report_says_so_in_its_own_header(tmp_path, monkeypatch):
    """The half of a delta that is easiest to mistake for the other half."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    text = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        options=Options(skills=False),
        results=[result(skills=False, outcome=scored(4.0))],
    ).summary()

    assert "Configuration: **skills=off responder=model**" in text
    assert "none offered" in text


def test_the_report_records_which_catalog_the_samples_saw(report):
    """What stops a number being compared across releases that offered
    different catalogs, and what says the switch took effect."""
    text, data, _ = report
    assert "Skills the catalog offered" in text
    assert data["samples"][0]["catalog"] == ["a-skill"]


def test_the_report_points_at_the_other_session_for_a_delta(report):
    """No single report contains a delta any more. It has to say where the
    other half is, or a reader will take one table for the whole finding."""
    text, _, _ = report
    assert "This report is one configuration" in text
    assert "--bench-skills" in text


def test_a_skill_case_names_the_comparison_worth_making(report):
    text, _, _ = report
    assert f"`{SKILL_CASE.skill}`" in text


def test_a_case_with_no_skill_is_not_sent_looking_for_an_ablation(
    tmp_path, monkeypatch
):
    """Withholding the catalog from it measures something, but not a delta of
    its own — there is no entry of its to withhold."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    text = Run(
        case=TASK_CASE,
        fixture=FIXTURE,
        results=[result(outcome=scored(0.05))],
    ).summary()
    assert "claims something about" not in text


def test_the_report_lands_under_its_own_case_inside_the_session(report):
    """`<session>/<namespace>/<case_id>`, so neither a second subject nor a
    second case for the same skill overwrites the first, and neither does the
    same case run again under different switches."""
    _, _, where = report
    assert where.name == SKILL_CASE.case_id
    assert where.parent.name == SKILL_CASE.namespace
    assert where.parent.parent.name == session_id()
    assert (where / "summary.md").is_file()


def test_two_samples_do_not_share_a_directory():
    """`sample-N`, so the second sample cannot overwrite the first's
    transcript — which is what a report of N rows and one transcript would
    look like."""
    assert {Result(sample=n).name for n in (1, 2)} == {"sample-1", "sample-2"}


# --- the session file ------------------------------------------------------


def test_the_session_id_is_stable_for_the_process():
    """Every case in one invocation writes under one session, or the directory
    stops meaning "one configuration" and starts meaning "one case"."""
    assert session_id() == session_id()
    assert session_id().startswith("session-")


def test_the_session_directory_is_under_the_artifact_root(tmp_path, monkeypatch):
    """Only the *name* is cached, so a test that redirects the root still gets
    a session directory inside it."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    assert session_dir() == tmp_path / session_id()


def test_the_session_file_records_the_configuration(tmp_path, monkeypatch):
    """It is the only thing that makes two report directories comparable, so
    it carries the switches rather than leaving them to the path."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    run = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        options=Options(skills=False, responder="silent", samples=2),
        results=[result(sample=n, skills=False, outcome=scored(0.1)) for n in (1, 2)],
    )
    data = json.loads(write_session(run).read_text())

    assert data["session"] == session_id()
    assert data["options"] == {
        "cases": "all",
        "fixtures": "all",
        "skills": False,
        "responder": "silent",
        "samples": 2,
    }
    assert data["configuration"] == "skills=off responder=silent"
    # The respondent is what actually answered, not what the provider table
    # would have supplied: a `silent` session never asked it for anything.
    assert data["respondent"] == "silent"


def test_the_session_file_names_a_briefed_run_by_its_switch(tmp_path, monkeypatch):
    """The same rule as the silent arm, on the corner that is easiest to
    mistake for the shipped one: a briefed session has all the information a
    spoken session ends up with, so the file has to be what says the agent
    never had to ask for any of it."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(
        _engine, "respondent_choice", lambda: FakeChoice("a-real-model")
    )
    data = json.loads(
        write_session(
            Run(
                case=SKILL_CASE,
                fixture=FIXTURE,
                options=Options(responder="briefed"),
                results=[result(responder="briefed", outcome=scored(0.1))],
            )
        ).read_text()
    )

    assert data["respondent"] == "briefed"
    assert data["configuration"] == "skills=on responder=briefed"


def test_the_session_file_says_which_code_produced_it(tmp_path, monkeypatch):
    """A report is read weeks later and compared with another, and "which code
    was this" is the first question that makes the comparison mean anything."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    data = json.loads(
        write_session(
            Run(case=SKILL_CASE, fixture=FIXTURE, results=[result(outcome=scored(0.1))])
        ).read_text()
    )
    assert data["code"]["biopb_mcp"]
    # In a checkout: the commit, and whether the tree was clean. `dirty` is the
    # load-bearing half — a sha identifies the code only if nothing was edited
    # on top of it, and two sessions can carry the same sha and different code.
    assert "commit" in data["code"] or data["code"]["checkout"] == "none"
    if "commit" in data["code"] and data["code"]["commit"] != "unknown":
        assert isinstance(data["code"]["dirty"], bool)


def test_a_missing_git_does_not_fail_a_run(monkeypatch, tmp_path):
    """Provenance is best-effort. A machine with no git still runs the
    benchmark; it just cannot say which commit."""
    monkeypatch.setattr(_engine, "_CODE_VERSION", None)
    monkeypatch.setattr(
        _engine.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    version = _engine.code_version()
    monkeypatch.setattr(_engine, "_CODE_VERSION", None)
    assert version["biopb_mcp"]
    assert version["commit"] == "unknown"


def test_the_session_file_accumulates_a_roster(tmp_path, monkeypatch):
    """Rewritten after every case, so an interrupted session still describes
    itself and lists what finished. A roster written only on the way out is the
    one you do not get on the run you most want it for."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    for case in (SKILL_CASE, TASK_CASE):
        write_session(
            Run(case=case, fixture=FIXTURE, results=[result(outcome=scored(0.1))])
        )
    data = json.loads((session_dir() / "session.json").read_text())

    assert set(data["cases"]) == {SKILL_CASE.label, TASK_CASE.label}
    assert data["cases"][SKILL_CASE.label]["samples"] == [OK]
    assert data["cases"][SKILL_CASE.label]["skill"] == SKILL_CASE.skill
    assert data["cases"][TASK_CASE.label]["skill"] == ""


def test_a_half_written_session_file_does_not_fail_the_next_case(tmp_path, monkeypatch):
    """The file is read back to accumulate. A run killed mid-write must not
    take the rest of the session down with it."""
    monkeypatch.setattr(_engine, "artifact_root", lambda: tmp_path)
    session_dir().mkdir(parents=True, exist_ok=True)
    (session_dir() / "session.json").write_text('{"cases": {"a"', encoding="utf-8")

    data = json.loads(
        write_session(
            Run(case=SKILL_CASE, fixture=FIXTURE, results=[result(outcome=scored(0.1))])
        ).read_text()
    )
    assert list(data["cases"]) == [SKILL_CASE.label]


# --- a run that never started ----------------------------------------------


def test_a_run_that_never_got_a_session_is_distinguishable():
    """It means the machine, not the case, and it is the one shape that should
    skip rather than be reported as a case that failed."""
    dead = Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        results=[Result(sample=n, error=f"{NO_SESSION}no display") for n in (1, 2)],
    )
    assert dead.failed_to_start
    assert not Run(
        case=SKILL_CASE,
        fixture=FIXTURE,
        results=[result(sample=n, outcome=scored(0.1)) for n in (1, 2)],
    ).failed_to_start


# --- smoke runs first, and gates ------------------------------------------


class FakeItem:
    """Just enough of a pytest item for the collection hook."""

    def __init__(self, filename: str, where=None):
        self.path = (where or conftest.HERE) / filename
        self.nodeid = f"{filename}::a_test"


def test_the_smoke_tests_are_moved_to_the_front():
    """Alphabetically `test_bench` sorts first, so four paid conversations went
    out before anything checked the stack could hold a napari layer."""
    items = [FakeItem("test_bench.py"), FakeItem(conftest.SMOKE)]
    conftest.pytest_collection_modifyitems(items)
    assert [i.path.name for i in items] == [conftest.SMOKE, "test_bench.py"]


def test_only_this_directory_is_reordered(tmp_path):
    """The hook is handed every item in the run, so a directory-level conftest
    that re-sorted all of them would silently rearrange the rest of the suite."""
    outsider = FakeItem("test_zzz_elsewhere.py", where=tmp_path)
    items = [outsider, FakeItem("test_bench.py"), FakeItem(conftest.SMOKE)]
    conftest.pytest_collection_modifyitems(items)
    assert items[0] is outsider
    assert [i.path.name for i in items[1:]] == [conftest.SMOKE, "test_bench.py"]


def test_a_failed_smoke_test_is_recorded_and_a_skipped_one_is_not(monkeypatch):
    """Ordering alone gates nothing without `-x`; the run reads this list and
    refuses to spend. A skip is not a failure — no display is reported by
    `unavailable()`, with better instructions."""
    monkeypatch.setattr(conftest, "_SMOKE_FAILURES", [])

    class Report:
        def __init__(self, nodeid, failed):
            self.nodeid, self.failed = nodeid, failed

    conftest.pytest_runtest_logreport(Report(f"{conftest.SMOKE}::boom", True))
    conftest.pytest_runtest_logreport(Report(f"{conftest.SMOKE}::skipped", False))
    conftest.pytest_runtest_logreport(Report("test_bench.py::other", True))
    assert conftest.smoke_failures() == [f"{conftest.SMOKE}::boom"]
