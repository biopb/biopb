"""Does `drift-correction` change what an agent does? A benchmark, not a gate.

`docs/skill-testing.md` §6. A skill's claim is a **behavioural delta**: an agent
following it does better than one without it. Measuring that needs a baseline,
so this runs a 2x2 and reports the corners:

===================  ==================================  ======================
                     respondent answers                  respondent silent
===================  ==================================  ======================
**skill offered**    does the whole thing work           does *asking* matter
**skill withheld**   does the *skill* matter             the floor
===================  ==================================  ======================

The two controls are orthogonal and were not, at first, both present. The
respondent axis shipped alone, which asks whether the withheld fact matters and
says nothing about whether the skill does — and for a benchmark that is the
wrong single control to have, since the skill's entire claim is the delta.

**Withholding is `services.skills_enabled: false`**, a real shipped
configuration: the kernel, napari, dask and every library stay exactly as they
are and only the curated procedure goes. §7's rule — disclose the environment,
withhold only the skill — and the reason a hand-cut hole would have been worse.

**No run's outcome fails this test.** Out of turns, wrong answer, gave up, even
a harness error — each becomes a *row with a reason*. These are one sample per
corner against a non-deterministic agent, so a pass/fail verdict would be false
precision, and stopping at the first bad corner would discard the three rows
that explain it. A poor fixture is still informative; a missing report is not.

Two things are asserted, and neither is a judgement on the skill: that the
summary reached disk with a transcript per arm, and that the **ablation took
effect**. That second one is not a finding — if `skills_enabled: false` stopped
withholding the catalog, the delta would read as zero for a reason that has
nothing to do with `drift-correction`, which is a green table saying the
opposite of the truth.

Costs four conversations. Marked `interaction`, deselected by default, never in
CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

from ..outcomes import _drift, _drift_channels
from ..outcomes._outcome import Attempt, artifact_root, providers_for, write_report
from . import _session
from ._agent import ToolCallingAgent
from ._conversation import SILENT, TOOL_CAP, TURN_CAP, converse, scrape
from ._models import agent_choice, respondent_choice
from ._personas import DRIFT_CHANNELS
from ._respondent import SilentRespondent, model_respondent
from ._session import SessionUnavailable, live_session

pytestmark = pytest.mark.interaction

#: See `test_drift_interaction`'s history for why these are not the loop
#: defaults: this workflow promotes compute to background jobs, and a run
#: severed mid-workflow leaves half-finished arrays that say nothing about the
#: skill. A cap that stops a stuck model is the point; a cap that stops a
#: working one only produces unreadable results.
MAX_TURNS = 90
MAX_TOOL_CALLS = 200

TASK = """\
A two-channel time-lapse is open in the napari viewer as the layer `timelapse`.
Its axes are (T, C, Y, X) = (24, 2, 192, 192), float32. There is no tensor
server in this session, so that layer is the only source of pixels.

The field of view moved during the acquisition, and I need that corrected
before anything is measured off this movie.

The microscopist who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  offsets    a (24, 2) float array of (dy, dx), relative to frame 0.
             SIGN CONVENTION: offsets[t] is the displacement frame t *carries* —
             a feature at (y, x) in frame 0 appears at (y + dy, x + dx) in
             frame t. It is NOT the shift needed to correct frame t, which is
             the negative of this.
  corrected  the drift-corrected movie, the same shape as the input

Those two names are how your result is collected, so bind them exactly.
"""

GATE_SPY = """
_expensive_calls = []


def _install_skill_spy():
    import functools

    try:
        from skimage import registration as _reg

        _orig = _reg.phase_cross_correlation

        @functools.wraps(_orig)
        def _wrapped(*a, **k):
            _expensive_calls.append("phase_cross_correlation")
            return _orig(*a, **k)

        _reg.phase_cross_correlation = _wrapped
    except Exception:
        pass

    try:
        import pystackreg

        _orig2 = pystackreg.StackReg.register_stack

        @functools.wraps(_orig2)
        def _wrapped2(self, *a, **k):
            _expensive_calls.append("register_stack")
            return _orig2(self, *a, **k)

        pystackreg.StackReg.register_stack = _wrapped2
    except Exception:
        pass


_install_skill_spy()
"""

REGISTRATION_MARKERS = ("phase_cross_correlation", "register_stack", "StackReg")

#: `write-a-skill` step 6 budgets at most three blocking checkpoints. Reported,
#: not asserted: one sample per corner cannot support a verdict.
BLOCKING_BUDGET = 3

# How a run ended. **Nothing here fails the test** — a run that ran out of
# turns, gave up, or got the wrong answer is a *result*, and the report is the
# deliverable. A suite that stopped at the first bad arm would throw away the
# three rows that explain it.
OK = "ok"
WRONG_ANSWER = "wrong-answer"
OUT_OF_TURNS = "out-of-turns"
OUT_OF_TOOL_CALLS = "out-of-tool-calls"
GAVE_UP = "gave-up"
NO_RESULT = "no-result"
UNSCORABLE = "unscorable-result"
HARNESS_ERROR = "harness-error"

#: Observations that do not decide the outcome but change how to read it. A run
#: can be `ok` *and* have asked six questions, or `ok` having never registered
#: anything — both are worth seeing next to the number.
FLAG_OVER_BUDGET = "over-ask-budget"
FLAG_NEVER_ASKED = "never-asked"
FLAG_NEVER_REGISTERED = "never-registered"
FLAG_CUT_OFF = "cut-off-but-scored"
FLAG_CATALOG_MISMATCH = "catalog-mismatch"


@dataclass(frozen=True)
class Arm:
    """One corner of the 2x2."""

    name: str
    skills: bool
    respondent: Callable[[], object]
    about: str


ARMS = (
    Arm(
        "skill+asked",
        True,
        lambda: model_respondent(DRIFT_CHANNELS),
        "everything available: the curated procedure and a microscopist who answers",
    ),
    Arm(
        "skill+silent",
        True,
        SilentRespondent,
        "the procedure, but nobody answers — does the withheld fact cost anything",
    ),
    Arm(
        "noskill+asked",
        False,
        lambda: model_respondent(DRIFT_CHANNELS),
        "no curated procedure, but a microscopist — does the skill add anything",
    ),
    Arm(
        "noskill+silent",
        False,
        SilentRespondent,
        "neither: the floor this whole layer is measured against",
    ),
)


@dataclass
class Result:
    """One arm's run, however it went — including not going at all."""

    arm: Arm
    trace: object = None
    outcome: object = None
    registered: list[str] = field(default_factory=list)
    #: What the catalog actually offered this run, checked at bring-up.
    catalog_hits: int = 0
    #: Set when the arm could not be run to completion at all.
    error: str = ""

    @property
    def metrics(self) -> dict[str, float | None]:
        if self.outcome is None:
            return {}
        return {m.name: m.value for m in self.outcome.metrics}

    def classify(self) -> tuple[str, str]:
        """``(outcome, reason)`` — what happened, in words that name a cause.

        Ordered so the *earliest* thing that went wrong wins. A run cut off at
        a cap may still have left a plausible-looking array, and reporting that
        as a wrong answer would blame the skill for a budget.
        """
        if self.error:
            return HARNESS_ERROR, self.error
        if self.trace is None or self.outcome is None:
            return HARNESS_ERROR, "the arm produced neither a trace nor a score"

        stopped = self.trace.stopped
        scored = self.outcome.scored

        if not scored:
            if stopped == TURN_CAP:
                return OUT_OF_TURNS, (
                    f"hit the {MAX_TURNS}-turn cap with nothing scorable bound"
                )
            if stopped == TOOL_CAP:
                return OUT_OF_TOOL_CALLS, (
                    f"hit the {MAX_TOOL_CALLS}-tool-call cap with nothing "
                    "scorable bound"
                )
            if stopped == SILENT:
                return GAVE_UP, "stopped talking without leaving a result"
            unavailable = [m.unavailable for m in self.outcome.metrics if m.unavailable]
            if any("cannot be compared" in u for u in unavailable):
                return UNSCORABLE, unavailable[0]
            return NO_RESULT, (
                f"finished ({stopped}) but left "
                f"{sorted(self.outcome.attempt.arrays) or 'nothing'}"
            )

        if self.outcome.passed:
            return OK, "within every tolerance"

        missed = [
            f"{m.name} {m.value:.4g} > {m.limit:g}" for m in self.outcome.failures
        ]
        return WRONG_ANSWER, "; ".join(missed)

    def flags(self) -> list[str]:
        out = []
        if self.trace is None:
            return out
        asked = len(self.trace.blocking_questions)
        if asked > BLOCKING_BUDGET:
            out.append(f"{FLAG_OVER_BUDGET}({asked})")
        if asked == 0:
            out.append(FLAG_NEVER_ASKED)
        if not self.registered:
            out.append(FLAG_NEVER_REGISTERED)
        scored_but_severed = (
            self.outcome is not None
            and self.outcome.scored
            and self.trace.stopped in (TURN_CAP, TOOL_CAP)
        )
        if scored_but_severed:
            out.append(FLAG_CUT_OFF)
        if bool(self.catalog_hits) != self.arm.skills:
            out.append(FLAG_CATALOG_MISMATCH)
        return out

    def row(self) -> dict:
        m = self.metrics
        outcome, reason = self.classify()
        return {
            "arm": self.arm.name,
            "skill_offered": self.arm.skills,
            "outcome": outcome,
            "reason": reason,
            "flags": self.flags(),
            "trajectory_rms_px": m.get("trajectory_rms_px"),
            "trajectory_max_err_px": m.get("trajectory_max_err_px"),
            "residual_ratio": m.get("residual_ratio"),
            "stopped": getattr(self.trace, "stopped", "—"),
            "turns": getattr(self.trace, "turns_used", 0),
            "blocking_questions": len(getattr(self.trace, "blocking_questions", [])),
            "messages_to_user": len(getattr(self.trace, "questions", [])),
            "tool_calls": len(getattr(self.trace, "tool_names", [])),
            "catalog_entries": self.catalog_hits,
            "registered": ",".join(self.registered) or "none",
        }


def _fixture():
    (provider,) = providers_for(_drift_channels.SKILL, tier="interaction")
    return provider.build()


def _load_fixture(session, fixture):
    session.put_array("_fixture_movie", np.asarray(fixture.data["movie"]))
    session.setup(
        "viewer.add_image(_fixture_movie, name='timelapse')\n"
        "del _fixture_movie\n"
        "print('layers:', [lyr.name for lyr in viewer.layers])"
    )
    session.setup(GATE_SPY)


def _registered(session) -> list[str]:
    out = session.setup("print('SPY', _expensive_calls)")
    return [m for m in REGISTRATION_MARKERS if m in out.text]


def _run_arm(arm: Arm, fixture) -> Result:
    """One corner: its own session, so the ablation is a real configuration."""
    with live_session(skills_enabled=arm.skills) as session:
        # Checked here rather than inferred from behaviour: whether the catalog was
        # actually withheld is the one thing that would silently make the
        # delta meaningless, and the agent may well call `find_skills` in the
        # ablation arm and simply get nothing back.
        catalog = session.call("find_skills", query="drift")
        catalog_hits = catalog.text.count('"id"')
        _load_fixture(session, fixture)
        trace = converse(
            session,
            ToolCallingAgent(),
            arm.respondent(),
            TASK,
            max_turns=MAX_TURNS,
            max_tool_calls=MAX_TOOL_CALLS,
        )
        where = artifact_root() / "interaction" / arm.name
        trace.write(where)

        registered = _registered(session)
        scraped = scrape(
            session, trace, {"offsets": "offsets", "corrected": "corrected"}
        )

    attempt = Attempt(
        subject=arm.name,
        arrays=scraped,
        notes=f"{arm.about} | stopped={trace.stopped} registered={registered}",
    )
    outcome = _drift.verify(fixture, attempt)
    write_report(outcome, artifact_root() / "interaction")
    _drift.save_artifacts(outcome, artifact_root() / "interaction" / arm.name)
    return Result(
        arm=arm,
        trace=trace,
        outcome=outcome,
        registered=registered,
        catalog_hits=catalog_hits,
    )


def _summary(rows: list[dict], where) -> str:
    agent, respondent = agent_choice(), respondent_choice()
    lines = [
        "# drift-correction — interaction benchmark",
        "",
        f"Agent under test: **{agent.name}**  ",
        f"Respondent: **{respondent.name}**  ",
        "Fixture: `two-channels-one-structural` — the structural channel is not "
        "named in the data.  ",
        f"Tolerances (from §5, unchanged): rms ≤ {_drift.TOLERANCE['trajectory_rms_px']}"
        f" px, max ≤ {_drift.TOLERANCE['trajectory_max_err_px']} px,"
        f" residual ≤ {_drift.TOLERANCE['residual_ratio']}",
        "",
        "One sample per corner. These runs are non-deterministic; read the",
        "table as an observation, not a measurement.",
        "",
        "| arm | skill | outcome | rms px | max px | residual | turns | asked | tools | reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.4g}"

    for r in rows:
        lines.append(
            f"| `{r['arm']}` | {'yes' if r['skill_offered'] else 'no'} "
            f"| **{r['outcome']}** "
            f"| {fmt(r['trajectory_rms_px'])} | {fmt(r['trajectory_max_err_px'])} "
            f"| {fmt(r['residual_ratio'])} | {r['turns']} "
            f"| {r['blocking_questions']} | {r['tool_calls']} "
            f"| {r['reason']} |"
        )
    flagged = [r for r in rows if r["flags"]]
    if flagged:
        lines += ["", "### Flags", ""]
        lines += [f"- `{r['arm']}` — {', '.join(r['flags'])}" for r in flagged]

    lines += [
        "",
        "## What each corner is for",
        "",
        *(f"- `{a.name}` — {a.about}" for a in ARMS),
        "",
        "## Reading it",
        "",
        "- **skill+asked vs noskill+asked** is the skill's behavioural delta, and",
        "  it is the number this whole layer exists to produce.",
        "- **skill+asked vs skill+silent** is whether the withheld fact costs",
        "  anything. If it does not, the fixture's asymmetry is decorative and",
        "  the interaction premise does not hold for this case.",
        "- **noskill+silent** is the floor. A corner at or near the tolerances",
        "  means the task is easy enough that nothing here discriminates.",
        f"- `asked` counts blocking questions; `write-a-skill` step 6 budgets"
        f" {BLOCKING_BUDGET}.",
        "",
        "Transcripts are in `<arm>/transcript.md`, with the raw event stream in",
        "`<arm>/trace.jsonl` and the before/after images beside them.",
        "",
    ]
    text = "\n".join(lines)
    where.mkdir(parents=True, exist_ok=True)
    (where / "summary.md").write_text(text, encoding="utf-8")
    (where / "summary.json").write_text(
        json.dumps(
            {
                "agent": agent.name,
                "respondent": respondent.name,
                "tolerance": _drift.TOLERANCE,
                "arms": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return text


@pytest.fixture(scope="module")
def benchmark():
    """Run every corner once, write the summary, hand back the rows."""
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    for side, choice in (
        ("agent", agent_choice()),
        ("respondent", respondent_choice()),
    ):
        if why := choice.why_unavailable():
            pytest.skip(f"{side}: {why}")
    if agent_choice().from_authoring_family:
        pytest.skip(
            f"the agent is {agent_choice().name}, from the family that wrote these "
            "skills — it could pass by recognising its own prose (§6a)."
        )

    fixture = _fixture()
    results = []
    for arm in ARMS:
        # Every arm is attempted, and a failure becomes a row rather than an
        # exception. Stopping at the first bad corner would throw away the
        # three that explain it -- and the report is the deliverable here, not
        # a verdict.
        try:
            results.append(_run_arm(arm, fixture))
        except SessionUnavailable as exc:
            if not results:
                pytest.skip(f"no session available: {exc}")
            results.append(Result(arm=arm, error=f"session unavailable: {exc}"))
        except Exception as exc:  # noqa: BLE001 - the row is the point
            results.append(Result(arm=arm, error=f"{type(exc).__name__}: {exc}"))

    rows = [r.row() for r in results]
    print("\n\n" + _summary(rows, artifact_root() / "interaction") + "\n")
    return results


def test_the_benchmark_ran_and_wrote_its_report(benchmark):
    """The only assertion, and it is about the deliverable rather than the
    result.

    **No arm's outcome fails this test.** Out of turns, wrong answer, gave up,
    even a harness error — each is a row with a reason, because the report is
    what this layer produces and stopping at the first bad corner would discard
    the three rows that explain it. These are one sample per corner against a
    non-deterministic agent; a pass/fail verdict on that would be false
    precision, and a poor fixture is still informative.

    What *is* checked: the summary reached disk, and every arm has a transcript
    to read. An arm that recorded nothing at all cannot be interpreted later,
    and that is a harness failure rather than a finding.
    """
    where = artifact_root() / "interaction"
    assert (where / "summary.md").is_file(), "the benchmark produced no summary"
    assert (where / "summary.json").is_file()

    missing = [
        r.arm.name
        for r in benchmark
        if not r.error and not (where / r.arm.name / "transcript.md").is_file()
    ]
    assert not missing, (
        f"these arms ran but left no transcript: {missing}. Their rows cannot "
        "be interpreted without one."
    )


def test_the_ablation_took_effect(benchmark):
    """Reported as a flag *and* asserted, because this one is not a finding
    about the skill — it is whether the table means anything.

    If `services.skills_enabled: false` stopped withholding the catalog, the
    no-skill arms would be reading the skill and the delta would read as zero
    for a reason that has nothing to do with `drift-correction`. That is a
    green table saying the opposite of the truth, which is worse than a red
    one.

    Checked on what the catalog *returned*, not on whether `find_skills` was
    called: the tool stays registered either way and it is `load_catalog()`
    that gates, so an ablated run can call it and get an empty list back.
    """
    wrong = [
        f"{r.arm.name}: skill_offered={r.arm.skills} but catalog had "
        f"{r.catalog_hits} entries"
        for r in benchmark
        if not r.error and bool(r.catalog_hits) != r.arm.skills
    ]
    assert not wrong, (
        "the ablation did not take effect, so the skill delta in the report is "
        f"not real: {wrong}"
    )
