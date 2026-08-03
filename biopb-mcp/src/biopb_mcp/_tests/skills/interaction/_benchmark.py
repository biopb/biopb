"""The benchmark engine: arms, outcomes, the report. Skill-agnostic.

`biopb-mcp/docs/skill-testing.md` §5. Nothing here knows what drift is. A skill's whole
contribution is one :class:`Case` — a task prompt, a persona, a fixture builder,
a verifier, and the names it wants back out of the kernel — and this module runs
the arms, classifies what happened, and writes the report.

That split is the point, and it is what makes the layer affordable at catalogue
scale: adding a skill writes one module under `cases/` and no test code. The
first version of this layer was a single `test_drift_benchmark.py` in which
about three quarters of the code was the engine and one quarter was drift, so a
second skill meant copying the engine.

The arms are a 2x2, because a skill's claim is a *behavioural delta* and a delta
needs a baseline:

===================  ==================================  ======================
                     respondent answers                  respondent silent
===================  ==================================  ======================
**skill offered**    does the whole thing work           does *asking* matter
**skill withheld**   does the *skill* matter             the floor
===================  ==================================  ======================

**Withholding is `services.skills_enabled: false`**, a real shipped
configuration: the kernel, napari, dask and every library stay exactly as they
are and only the curated procedure goes. §6's rule — disclose the environment,
withhold only the skill — and the reason a hand-cut hole would have been worse.

The right-hand column is about the **fixture**, not the skill, so
`BIOPB_SKILL_ARMS=asked` drops it and halves the cost of a case. See
:data:`ARMS_ENV`.

**No run's outcome fails anything.** Out of turns, wrong answer, gave up, even a
harness error — each becomes a *row with a reason*. These are one sample per
corner against a non-deterministic agent, so a pass/fail verdict would be false
precision, and stopping at the first bad corner would discard the three rows
that explain it. The report is the deliverable; a poor fixture is still
informative, a missing report is not.

This module never imports pytest. Skipping is a decision about a test run, and
it is made in `test_benchmark.py`; here a run that cannot happen is a
:class:`Result` carrying the reason.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ._agent import ToolCallingAgent
from ._conversation import (
    AGENT_TRUNCATED,
    RESPONDENT_FAILED,
    SILENT,
    STALLED,
    TOOL_CAP,
    TURN_CAP,
    converse,
    scrape,
)
from ._fixture import (
    Attempt,
    Fixture,
    Outcome,
    artifact_root,
    curated_for,
    write_report,
)
from ._models import agent_choice, reachable, respondent_choice, setting, text_backend
from ._respondent import Persona, SilentRespondent, model_respondent
from ._session import SessionUnavailable, live_session

#: Bounds on one run. Not the loop's own defaults: these workflows promote
#: compute to background jobs, and a run severed mid-workflow leaves
#: half-finished arrays that say nothing about the skill. A cap that stops a
#: stuck model is the point; a cap that stops a working one only produces
#: unreadable results.
MAX_TURNS = 90
MAX_TOOL_CALLS = 200

#: `write-a-skill` step 4 budgets at most three blocking checkpoints. Reported,
#: not asserted: one sample per corner cannot support a verdict.
BLOCKING_BUDGET = 3

# How a run ended. **Nothing here fails a test** -- a run that ran out of turns,
# gave up, or got the wrong answer is a *result*.
OK = "ok"
WRONG_ANSWER = "wrong-answer"
OUT_OF_TURNS = "out-of-turns"
OUT_OF_TOOL_CALLS = "out-of-tool-calls"
GAVE_UP = "gave-up"
NO_RESULT = "no-result"
UNSCORABLE = "unscorable-result"
HARNESS_ERROR = "harness-error"

#: Observations that do not decide the outcome but change how to read it. A run
#: can be `ok` *and* have asked six questions, or `ok` having been cut off at a
#: cap — both are worth seeing next to the number.
FLAG_OVER_BUDGET = "over-ask-budget"
FLAG_NEVER_ASKED = "never-asked"
FLAG_CUT_OFF = "cut-off-but-scored"
FLAG_CATALOG_MISMATCH = "catalog-mismatch"
#: Asked, and was never once answered. On an `asked` arm that is not a result,
#: it is the respondent being broken — the arm measured the `silent` condition
#: under the `asked` label, and its row is not comparable to anything.
FLAG_UNANSWERED = "asked-but-unanswered"
#: The run ended because it stopped progressing, not because the agent finished.
#: Distinct from `cut-off-but-scored`: a stalled run was not severed mid-workflow,
#: it was talking in circles — usually because the respondent could not end it.
FLAG_STALLED = "stalled"

#: A missing session is worth telling apart from any other harness failure: it
#: means the machine, not the run.
NO_SESSION = "session unavailable: "


@dataclass(frozen=True)
class Layer:
    """One fixture array, as the agent finds it on the viewer.

    ``kind`` picks `add_image` or `add_labels`, which is not cosmetic: a Labels
    layer is what makes a segmentation addressable as objects, and several
    skills' Parameters tables ask for one by name.
    """

    name: str
    key: str
    kind: str = "image"


@dataclass(frozen=True)
class Case:
    """One skill's whole contribution to this layer.

    Everything that is *about the skill* and nothing that is about
    benchmarking. Adding a skill is writing one of these — see
    `cases/drift_correction.py` for the worked example, and `cases/__init__.py`
    for the three-line procedure.

    The fixture is a callable rather than a value so a case module costs
    nothing at import: 30 of these are collected by every ordinary test run,
    and only the one being benchmarked should build megabytes of arrays.
    """

    #: Skill id, as `find_skills` and `skill://<id>` know it.
    skill: str
    #: What the agent is asked to do, including where its results should land.
    task: str
    #: Who it is talking to, and the fact the fixture strips out.
    persona: Persona
    #: ``() -> Fixture``: the data, the truth withheld from it, the tolerances.
    build: Callable[[], Fixture]
    #: Where the fixture's arrays land on the viewer, in order.
    layers: Sequence[Layer]
    #: What the verifier wants -> the kernel expression that yields it.
    collect: Mapping[str, str]
    #: ``(fixture, attempt) -> Outcome``. Numeric, never judged prose: these
    #: skills emit numbers with knowable right answers.
    score: Callable[[Fixture, Attempt], Outcome]
    #: Optional ``(outcome, dir) -> None`` — the before/after images.
    save_artifacts: Callable[[Outcome, Path], None] | None = None
    #: Kernel plugins the skill's `checklist:` names, seeded into the session's
    #: own config tree. Without this a `plugin:` token is unresolvable and the
    #: run is scoring an environment the skill declares it cannot work in.
    plugins: Sequence[str] = ()
    #: What to ask `find_skills`, to check the ablation actually took effect.
    #: Defaults to the skill id.
    catalog_query: str = ""
    #: Case-folded substrings that must appear in the persona's rendered
    #: prompt: the fact the fixture strips, so the run is answerable at all.
    persona_must_know: Sequence[str] = ()
    #: Case-folded substrings that must **not**. A persona that has absorbed the
    #: procedure can answer a question the agent never properly asked, and the
    #: numeric result stops meaning what it appears to. Name the skill's own
    #: vocabulary here — `test_cases` asserts it, hermetically and free.
    persona_must_not_know: Sequence[str] = ()
    blocking_budget: int = BLOCKING_BUDGET
    max_turns: int = MAX_TURNS
    max_tool_calls: int = MAX_TOOL_CALLS

    @property
    def query(self) -> str:
        return self.catalog_query or self.skill

    def build_fixture(self) -> Fixture:
        """This case's fixture — real data if this machine has any, else the
        procedural one the case ships (`_fixture.curated_for`)."""
        return curated_for(self.skill) or self.build()


@dataclass(frozen=True)
class Arm:
    """One corner of the 2x2."""

    name: str
    skills: bool
    asked: bool
    about: str


ARMS = (
    Arm(
        "skill+asked",
        True,
        True,
        "everything available: the curated procedure and a user who answers",
    ),
    Arm(
        "skill+silent",
        True,
        False,
        "the procedure, but nobody answers — does the withheld fact cost anything",
    ),
    Arm(
        "noskill+asked",
        False,
        True,
        "no curated procedure, but a user who answers — does the skill add anything",
    ),
    Arm(
        "noskill+silent",
        False,
        False,
        "neither: the floor this whole layer is measured against",
    ),
)

#: Which corners to spend on. The default is the whole 2x2.
#:
#: **The two `+silent` arms measure the fixture, not the skill.** They ask
#: whether the withheld fact is really unobtainable from the pixels -- a
#: property of the construction in `cases/`, which does not change when a body
#: is edited, and which `test_cases.py` already checks the cheap half of by
#: asserting no truth key appears in `data`. The skill's own delta is
#: `skill+asked` against `noskill+asked`, and it needs neither silent arm.
#:
#: So they are droppable once a fixture's asymmetry has been established, and
#: dropping them halves the wall-clock of a case. Re-run the full set when the
#: fixture changes, or when a report's `skill+asked` row makes the asymmetry look
#: decorative. `drift-correction` is the standing reason to keep re-running it:
#: a capable agent recovered its withheld fact anyway (§5c).
ARMS_ENV = "BIOPB_SKILL_ARMS"
ARM_SETS: dict[str, tuple[Arm, ...]] = {
    "all": ARMS,
    "asked": tuple(a for a in ARMS if a.asked),
}


def selected_arms() -> tuple[Arm, ...]:
    """The corners this run will spend on, per ``BIOPB_SKILL_ARMS``.

    Unknown values raise rather than fall back to the full set: the two are
    twenty minutes apart per case, and a typo that quietly spends the larger
    number is discovered by looking at the clock.
    """
    choice = setting(ARMS_ENV, "all").strip().lower()
    if choice not in ARM_SETS:
        raise ValueError(
            f"{ARMS_ENV}={choice!r} is not one of {sorted(ARM_SETS)} — "
            "`all` is the 2x2, `asked` drops the two silent arms, which measure "
            "the fixture rather than the skill."
        )
    return ARM_SETS[choice]


def _respondent_for(arm: Arm, case: Case):
    return model_respondent(case.persona) if arm.asked else SilentRespondent()


@dataclass
class Result:
    """One arm's run, however it went — including not going at all."""

    arm: Arm
    trace: object = None
    outcome: Outcome | None = None
    #: How many skills the catalog actually offered this run, read at bring-up.
    catalog_hits: int = 0
    #: Wall-clock for this arm, including bring-up and teardown. Reported so the
    #: cost of a case is legible from its own report rather than remembered.
    seconds: float = 0.0
    #: Set when the arm could not be run to completion at all.
    error: str = ""

    @property
    def metrics(self) -> dict[str, float | None]:
        if self.outcome is None:
            return {}
        return {m.name: m.value for m in self.outcome.metrics}

    def classify(self) -> tuple[str, str]:
        """``(outcome, reason)`` — what happened, in words that name a cause.

        Ordered so the *earliest* thing that went wrong wins. A run cut off at a
        cap may still have left a plausible-looking array, and reporting that as
        a wrong answer would blame the skill for a budget.
        """
        if self.error:
            return HARNESS_ERROR, self.error
        if self.trace is None or self.outcome is None:
            return HARNESS_ERROR, "the arm produced neither a trace nor a score"

        stopped = self.trace.stopped
        # Before anything about the agent: these two are the *provider* ending
        # the run, and they are indistinguishable from an agent finishing or
        # giving up unless they are read first. Scoring them against the skill
        # is how a broken respondent gets reported as four bad rows.
        if stopped == RESPONDENT_FAILED:
            return HARNESS_ERROR, "the respondent never answered; see the trace"
        if stopped == AGENT_TRUNCATED:
            return HARNESS_ERROR, "the agent was cut off at its token budget"

        if not self.outcome.scored:
            if stopped == TURN_CAP:
                return OUT_OF_TURNS, "hit the turn cap with nothing scorable bound"
            if stopped == TOOL_CAP:
                return (
                    OUT_OF_TOOL_CALLS,
                    "hit the tool-call cap with nothing scorable bound",
                )
            if stopped == SILENT:
                return GAVE_UP, "stopped talking without leaving a result"
            left = sorted(self.outcome.attempt.arrays)
            if left:
                # It produced *something*, so this is a bad result rather than
                # no result -- a distinction the verifier's `unavailable` text
                # can usually explain, and one that points at different causes.
                why = [m.unavailable for m in self.outcome.metrics if m.unavailable]
                return UNSCORABLE, why[0] if why else f"left {left}, none scorable"
            return NO_RESULT, f"{stopped} without leaving a result"

        if self.outcome.passed:
            return OK, "within every tolerance"
        return WRONG_ANSWER, "; ".join(
            f"{m.name} {m.value:.4g} > {m.limit:g}" for m in self.outcome.failures
        )

    def flags(self, budget: int = BLOCKING_BUDGET) -> list[str]:
        out = []
        if self.trace is None:
            return out
        asked = len(self.trace.blocking_questions)
        if asked > budget:
            out.append(f"{FLAG_OVER_BUDGET}({asked})")
        if asked == 0:
            out.append(FLAG_NEVER_ASKED)
        if asked and not self.trace.answers:
            out.append(FLAG_UNANSWERED)
        severed = self.trace.stopped in (TURN_CAP, TOOL_CAP)
        if self.outcome is not None and self.outcome.scored and severed:
            out.append(FLAG_CUT_OFF)
        if self.trace.stopped == STALLED:
            out.append(FLAG_STALLED)
        if bool(self.catalog_hits) != self.arm.skills:
            out.append(FLAG_CATALOG_MISMATCH)
        return out

    def row(self, budget: int = BLOCKING_BUDGET) -> dict:
        outcome, reason = self.classify()
        return {
            "arm": self.arm.name,
            "skill_offered": self.arm.skills,
            "outcome": outcome,
            "reason": reason,
            "flags": self.flags(budget),
            "metrics": self.metrics,
            "stopped": getattr(self.trace, "stopped", "—"),
            "turns": getattr(self.trace, "turns_used", 0),
            "blocking_questions": len(getattr(self.trace, "blocking_questions", [])),
            "messages_to_user": len(getattr(self.trace, "questions", [])),
            "tool_calls": len(getattr(self.trace, "tool_names", [])),
            "catalog_entries": self.catalog_hits,
            "seconds": round(self.seconds, 1),
        }


def where_for(case: Case) -> Path:
    """Where this case's report and transcripts land."""
    return artifact_root() / "interaction" / case.skill


def catalog_size(text: str) -> int:
    """How many skills `find_skills` returned, from the text an agent sees.

    Parsed rather than pattern-counted: whether the ablation took effect is the
    one thing that would silently make the whole table meaningless, so it must
    not rest on a substring surviving a formatting change. Text this cannot
    parse counts as *something*, never as nothing — an unreadable answer is not
    evidence that the catalog was withheld, and reading it as one would turn a
    broken ablation into a clean-looking table.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return 1 if text.strip() else 0
    if isinstance(parsed, dict):
        # A list return can reach a client wrapped in structured content; the
        # entries are still the only list in it.
        parsed = next((v for v in parsed.values() if isinstance(v, list)), parsed)
    if isinstance(parsed, list):
        return len(parsed)
    return 1 if parsed else 0


def _load_fixture(session, case: Case, fixture: Fixture) -> None:
    """Put the fixture on the viewer as *setup*, not as something the agent did.

    Injecting before handover keeps the fixture out of the agent's context and
    stops it burning turns on a setup the harness can do instantly. It goes
    through `session.setup`, recorded at turn -1, so the trace still answers
    "what did the agent do" honestly.
    """
    for layer in case.layers:
        session.put_array("_fixture_array", np.asarray(fixture.data[layer.key]))
        adder = "add_labels" if layer.kind == "labels" else "add_image"
        session.setup(
            f"viewer.{adder}(_fixture_array, name={layer.name!r})\ndel _fixture_array"
        )
    session.setup("print('layers:', [lyr.name for lyr in viewer.layers])")


def run_arm(case: Case, arm: Arm, fixture: Fixture) -> Result:
    """One corner: its own session, so the ablation is a real configuration."""
    with live_session(skills_enabled=arm.skills, plugins=case.plugins) as session:
        # Read here rather than inferred from behaviour: the agent may well call
        # `find_skills` in the ablation arm and simply get nothing back, and
        # `load_catalog()` is what gates, not whether the tool was registered.
        catalog_hits = catalog_size(session.call("find_skills", query=case.query).text)
        _load_fixture(session, case, fixture)
        trace = converse(
            session,
            ToolCallingAgent(),
            _respondent_for(arm, case),
            case.task,
            max_turns=case.max_turns,
            max_tool_calls=case.max_tool_calls,
        )
        trace.write(where_for(case) / arm.name)
        scraped = scrape(session, trace, dict(case.collect))

    attempt = Attempt(
        subject=arm.name,
        arrays=scraped,
        notes=f"{arm.about} | stopped={trace.stopped}",
    )
    outcome = case.score(fixture, attempt)
    write_report(outcome, where_for(case))
    if case.save_artifacts is not None:
        case.save_artifacts(outcome, where_for(case) / arm.name)
    return Result(arm=arm, trace=trace, outcome=outcome, catalog_hits=catalog_hits)


@dataclass
class Run:
    """One case's four arms, and the fixture they were scored against."""

    case: Case
    fixture: Fixture
    results: list[Result]

    @property
    def failed_to_start(self) -> bool:
        """No arm got a session. The machine, not the skill — and not a row
        worth reading, since nothing was ever asked of the agent."""
        return bool(self.results) and all(
            r.error.startswith(NO_SESSION) for r in self.results
        )

    def summary(self) -> str:
        return write_summary(self.case, self.results, self.fixture)


def progress(message: str) -> None:
    """One line of progress, straight to stdout.

    A case is four conversations and the better part of half an hour, and
    without this the only sign of life is the artifact directory filling up.
    This module never imports pytest, so it emits and the *test* decides
    visibility — pytest discards a passing test's captured output, which is why
    the documented command passes ``-s``.
    """
    print(message, flush=True)


def run_case(case: Case) -> Run:
    """Every selected corner, once. A failing arm becomes a row, never an
    exception."""
    arms = selected_arms()
    fixture = case.build_fixture()
    progress(
        f"\n[{case.skill}] {len(arms)} of {len(ARMS)} arms against "
        f"`{fixture.case_id}` -> {where_for(case)}"
    )
    results = []
    for n, arm in enumerate(arms, start=1):
        progress(f"[{case.skill}] {n}/{len(arms)} {arm.name}: running")
        started = time.monotonic()
        try:
            result = run_arm(case, arm, fixture)
        except SessionUnavailable as exc:
            result = Result(arm=arm, error=f"{NO_SESSION}{exc}")
        except Exception as exc:  # noqa: BLE001 - the row is the point
            result = Result(arm=arm, error=f"{type(exc).__name__}: {exc}")
        result.seconds = time.monotonic() - started
        outcome, reason = result.classify()
        progress(
            f"[{case.skill}] {n}/{len(arms)} {arm.name}: {outcome} "
            f"in {result.seconds / 60:.1f} min — {reason}"
        )
        results.append(result)
    return Run(case=case, fixture=fixture, results=results)


def _metric_columns(results: Sequence[Result]) -> list[tuple[str, float]]:
    """``(name, limit)`` for every metric any arm produced, first seen first.

    Read off the metrics rather than declared on the :class:`Case`: a verifier
    reports what a fixture's truth supports, which differs between a synthetic
    and a curated case for the same skill, so the table has to follow it.
    """
    seen: dict[str, float] = {}
    for result in results:
        if result.outcome is None:
            continue
        for metric in result.outcome.metrics:
            seen.setdefault(metric.name, metric.limit)
    return list(seen.items())


def write_summary(case: Case, results: Sequence[Result], fixture: Fixture) -> str:
    """The report. Written to `where_for(case)` and returned for printing."""
    columns = _metric_columns(results)
    rows = [r.row(case.blocking_budget) for r in results]
    agent, respondent = agent_choice(), respondent_choice()
    ran = [r.arm for r in results]
    skipped = [a for a in ARMS if a not in ran]

    def fmt(value):
        return "—" if value is None else f"{value:.4g}"

    lines = [
        f"# {case.skill} — interaction benchmark",
        "",
        f"Agent under test: **{agent.name}**  ",
        f"Respondent: **{respondent.name}**  ",
        f"Fixture: `{fixture.case_id}` — {fixture.about or 'no description'}  ",
        "Tolerances: " + ", ".join(f"{name} ≤ {limit:g}" for name, limit in columns),
        "",
    ]
    if skipped:
        # Named, not merely absent: a short table is otherwise indistinguishable
        # from a 2x2 whose other corners died, and the missing corners are the
        # ones that say whether the fixture's asymmetry is real.
        lines += [
            f"**{len(ran)} of {len(ARMS)} arms** — not run: "
            + ", ".join(f"`{a.name}`" for a in skipped)
            + f" (`{ARMS_ENV}`). Those measure the fixture rather than the "
            "skill; the delta below is unaffected.",
            "",
        ]
    lines += [
        "One sample per corner. These runs are non-deterministic; read the",
        "table as an observation, not a measurement.",
        "",
        "| arm | skill | outcome | "
        + " | ".join(name for name, _ in columns)
        + " | turns | asked | tools | min | reason |",
        "|---|---|---|" + "---|" * (len(columns) + 5),
    ]
    for row in rows:
        cells = " | ".join(fmt(row["metrics"].get(name)) for name, _ in columns)
        lines.append(
            f"| `{row['arm']}` | {'yes' if row['skill_offered'] else 'no'} "
            f"| **{row['outcome']}** | {cells} | {row['turns']} "
            f"| {row['blocking_questions']} | {row['tool_calls']} "
            f"| {row['seconds'] / 60:.1f} | {row['reason']} |"
        )

    if flagged := [r for r in rows if r["flags"]]:
        lines += ["", "### Flags", ""]
        lines += [f"- `{r['arm']}` — {', '.join(r['flags'])}" for r in flagged]

    lines += [
        "",
        "## What each corner is for",
        "",
        *(f"- `{a.name}` — {a.about}" for a in ran),
        "",
        "## Reading it",
        "",
        "- **skill+asked vs noskill+asked** is the skill's behavioural delta, and",
        "  it is the number this whole layer exists to produce.",
    ]
    if not skipped:
        lines += [
            "- **skill+asked vs skill+silent** is whether the withheld fact costs",
            "  anything. If it does not, the fixture's asymmetry is decorative and",
            "  the interaction premise does not hold for this case.",
            "- **noskill+silent** is the floor. A corner at or near the tolerances",
            "  means the task is easy enough that nothing here discriminates.",
        ]
    lines += [
        f"- `asked` counts blocking questions; `write-a-skill` step 4 budgets"
        f" {case.blocking_budget}.",
        "",
        "Transcripts are in `<arm>/transcript.md`, with the raw event stream in",
        "`<arm>/trace.jsonl` and any images beside them.",
        "",
    ]

    text = "\n".join(lines)
    where = where_for(case)
    where.mkdir(parents=True, exist_ok=True)
    (where / "summary.md").write_text(text, encoding="utf-8")
    (where / "summary.json").write_text(
        json.dumps(
            {
                "skill": case.skill,
                "fixture": fixture.case_id,
                "agent": agent.name,
                "respondent": respondent.name,
                "tolerance": dict(columns),
                # So a two-arm report is machine-distinguishable from a 2x2
                # whose other corners died, which the rows alone cannot say.
                "arms_not_run": [a.name for a in skipped],
                "arms": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return text


def unavailable(case: Case) -> str:
    """Why this case cannot be benchmarked here, or ``""``.

    The environment checks that are cheap and answerable before anything is
    spawned or spent. §5a is one of them: an agent from the family that wrote
    these skills could pass by recognising its own prose.
    """
    from . import _session

    if reason := _session.why_unavailable():
        return reason
    for side, choice in (
        ("agent", agent_choice()),
        ("respondent", respondent_choice()),
    ):
        if why := choice.why_unavailable():
            return f"{side}: {why}"
    if agent_choice().from_authoring_family:
        return (
            f"the agent is {agent_choice().name}, from the family that wrote these "
            "skills — it could pass by recognising its own prose (§5a)."
        )
    # Last, because it is the only one that costs a request: a model the
    # endpoint does not serve fails every arm identically, and a shell export
    # beating the dotenv is the ordinary way to arrive there.
    for side, choice in (
        ("agent", agent_choice()),
        ("respondent", respondent_choice()),
    ):
        if why := reachable(text_backend(choice)):
            return (
                f"{side} {choice.name} at {choice.base_url or 'the provider default'} "
                f"is not usable: {why}"
            )
    return ""
