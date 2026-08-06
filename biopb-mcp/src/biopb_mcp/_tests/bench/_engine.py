"""The engine: one configuration, N samples, outcome classification, the report.

`biopb-mcp/docs/skills.md` §10. Nothing here knows what drift is, or what a
landmark is. A case's whole contribution is one :class:`~._case.Case` — a task
prompt, a persona, a fixture spec, a verifier, and the names it wants back out
of the kernel — and this module runs the corners, classifies what happened, and
writes the report.

That split is what makes the layer affordable at catalogue scale: adding a case
writes one module under `cases/` and no test code. The first version was a
single `test_drift_benchmark.py` in which about three quarters of the code was
the engine and one quarter was drift, so a second skill meant copying the
engine — which is then what happened, twice.

**One invocation runs one configuration.** Whether the catalog is offered and
who answers the agent are switches (`_options.py`), fixed for the whole session,
so this module varies exactly one thing: how many times to run each case.

That was a 2x2 the engine iterated per case, and it is four commands now:

===================  ==================================  ======================
                     `--bench-responder=model`           `=silent`
===================  ==================================  ======================
`--bench-skills=true`   does the whole thing work        does *asking* matter
`=false`                does the *skill* matter          the floor
===================  ==================================  ======================

What that buys is that a run costs what you asked for, every row in a report was
configured identically, and the four corners are four directories a reader can
diff. What it costs is that no single report contains a delta: the delta is two
sessions, which is why `session.json` records the configuration and the report
header repeats it.

Samples are the axis a single session still varies, and the only source of
information a case with no ablation ever had: run the same thing again, and the
spread between runs is the finding a single number cannot carry.

**No run's outcome fails anything.** Out of turns, wrong answer, gave up, even
a harness error — each becomes a *row with a reason*. These are a handful of
samples against a non-deterministic agent, so a pass/fail verdict would be
false precision, and stopping at the first bad corner would discard the rows
that explain it. The report is the deliverable; a poor fixture is still
informative, a missing report is not.

This module never imports pytest. Skipping is a decision about a test run, and
it is made in `test_bench.py`; here a run that cannot happen is a
:class:`Result` carrying the reason.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..agentbench import _plane
from ..agentbench._agent import ToolCallingAgent
from ..agentbench._conversation import (
    AGENT_TRUNCATED,
    RESPONDENT_FAILED,
    SILENT,
    STALLED,
    TOOL_CAP,
    TURN_CAP,
    converse,
    scrape,
    with_protocol,
)
from ..agentbench._fixture import (
    Attempt,
    Fixture,
    Outcome,
    artifact_root,
    checkout_root,
    write_report,
)
from ..agentbench._models import (
    agent_choice,
    reachable,
    respondent_choice,
    text_backend,
)
from ..agentbench._respondent import SilentRespondent, model_respondent
from ..agentbench._session import SessionUnavailable, live_session
from ._case import BLOCKING_BUDGET, LAYER_KINDS, TENSOR_HANDLE, Case
from ._options import Options

# How a run ended. **Nothing here fails a test** -- a run that ran out of turns,
# gave up, or got the wrong answer is a *result*. One vocabulary for both kinds
# of case, so a reader who knows one report can read the other; these were two
# copies that agreed by hand until this file merged them.
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
#: Asked, and was never once answered. Under `--bench-responder=model` that is
#: not a result, it is the respondent being broken — the session measured the
#: `silent` condition under the other label, and is comparable to nothing.
FLAG_UNANSWERED = "asked-but-unanswered"
#: The run ended because it stopped progressing, not because the agent finished.
#: Distinct from `cut-off-but-scored`: a stalled run was not severed mid-workflow,
#: it was talking in circles — usually because the respondent could not end it.
FLAG_STALLED = "stalled"
#: The kernel read something the harness owns — a case's `truth`, or the skill
#: markdown a `--bench-skills=false` run is meant to lack. Unlike every other
#: flag here, this says the *number* is void rather than qualified: a run that
#: read its own answer key measured nothing, and no amount of context makes its
#: row comparable. `execute_code` is arbitrary Python and always will be, so the
#: layer's defence is that this cannot happen quietly (`_session` tripwire).
FLAG_PEEKED = "read-harness-internals"
#: A fixture array on the run's plane is no longer the bytes the harness put
#: there. Like `read-harness-internals` this voids rather than qualifies the
#: number — and it voids every *later* sample too, since the plane outlives the
#: session. A fixture id is a one-way hash of a name the agent never sees, so
#: reaching this needs deliberate effort; the flag exists so that effort cannot
#: pass unnoticed.
FLAG_CONTAMINATED = "fixture-overwritten"

#: A missing session is worth telling apart from any other harness failure: it
#: means the machine, not the run.
NO_SESSION = "session unavailable: "


# --- selecting what to run --------------------------------------------------


def select(cases: Sequence[Case], options: Options) -> tuple[Case, ...]:
    """The cases *options* asks for, in the order they were given.

    Filtered rather than skipped, because these are the paid runs: a
    `--bench-cases=skills` invocation should collect the skill cases and nothing
    else, not print a screen of skips. What was dropped is said out loud
    instead (`conftest.pytest_terminal_summary`), since a shorter list is
    otherwise indistinguishable from a shorter catalogue.
    """
    chosen = []
    for case in cases:
        if options.cases == "skills" and not case.about_a_skill:
            continue
        if options.cases == "tasks" and case.about_a_skill:
            continue
        if options.fixtures != "all" and case.fixture.kind != options.fixtures:
            continue
        chosen.append(case)
    return tuple(chosen)


def respondent_for(case: Case, options: Options):
    """Who answers this run, per `--bench-responder`.

    The dispatch is here and the vocabulary is in `_options`, which is
    stdlib-only and cannot import this module. `test_report.py` pins the two
    together, so a value the flag offers and the engine cannot build is a red
    test rather than a `KeyError` twenty minutes into a paid run.
    """
    if options.responder == "silent":
        return SilentRespondent()
    return model_respondent(case.persona)


# --- one run's result -------------------------------------------------------


@dataclass
class Result:
    """One sample, however it went — including not going at all.

    There is no arm here any more. The configuration a sample ran under is the
    *session's*, identical for every sample and every case in the invocation,
    and it is recorded once in `session.json` rather than on each row.
    """

    #: 1-based. Part of the artifact path, so two samples never overwrite each
    #: other's transcript.
    sample: int = 1
    #: Whether this run's session offered the catalog — `--bench-skills`, copied
    #: onto the result so a row can be read without its session file.
    skills_offered: bool = True
    trace: object = None
    outcome: Outcome | None = None
    #: Which skills the catalog offered this run, read at bring-up. Ids rather
    #: than a count, because for a case with no ablation this is *provenance*:
    #: a task that later gains a covering skill silently re-bases its own
    #: number, and a run that does not say which catalog it saw cannot be
    #: compared with one from another release.
    catalog: tuple[str, ...] = ()
    #: Wall-clock for this sample, including bring-up and teardown. Reported so
    #: the cost of a case is legible from its own report rather than remembered.
    seconds: float = 0.0
    #: Set when the sample could not be run to completion at all.
    error: str = ""
    #: Harness-owned paths the kernel opened, from `LiveSession.peeked()`.
    peeked: tuple[str, ...] = ()
    #: Fixture arrays on the plane whose bytes changed under this sample.
    contaminated: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """This sample's directory under the case's artifact root."""
        return f"sample-{self.sample}"

    @property
    def metrics(self) -> dict[str, float | None]:
        if self.outcome is None:
            return {}
        return {m.name: m.value for m in self.outcome.metrics}

    def classify(self) -> tuple[str, str]:
        """``(outcome, reason)`` — what happened, in words that name a cause.

        Ordered so the *earliest* thing that went wrong wins. A run cut off at a
        cap may still have left a plausible-looking array, and reporting that as
        a wrong answer would blame the case for a budget.
        """
        if self.error:
            return HARNESS_ERROR, self.error
        if self.trace is None or self.outcome is None:
            return HARNESS_ERROR, "the run produced neither a trace nor a score"

        stopped = self.trace.stopped
        # Before anything about the agent: these two are the *provider* ending
        # the run, and they are indistinguishable from an agent finishing or
        # giving up unless they are read first. Scoring them against the case
        # is how one broken respondent gets reported as a case that failed.
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
        # Before the `trace is None` guard: a run that read the answer key and
        # then died still has to say so.
        if self.peeked:
            out.append(f"{FLAG_PEEKED}({len(self.peeked)})")
        if self.contaminated:
            out.append(f"{FLAG_CONTAMINATED}({','.join(self.contaminated)})")
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
        if bool(self.catalog) != self.skills_offered:
            out.append(FLAG_CATALOG_MISMATCH)
        return out

    def row(self, budget: int = BLOCKING_BUDGET) -> dict:
        outcome, reason = self.classify()
        return {
            "sample": self.sample,
            "skills_offered": self.skills_offered,
            "outcome": outcome,
            "reason": reason,
            "flags": self.flags(budget),
            "metrics": self.metrics,
            "stopped": getattr(self.trace, "stopped", "—"),
            "turns": getattr(self.trace, "turns_used", 0),
            "blocking_questions": len(getattr(self.trace, "blocking_questions", [])),
            "messages_to_user": len(getattr(self.trace, "questions", [])),
            "tool_calls": len(getattr(self.trace, "tool_names", [])),
            "catalog": list(self.catalog),
            "seconds": round(self.seconds, 1),
        }


# --- the session ------------------------------------------------------------
#
# One invocation is one configuration, so one invocation gets one directory and
# writes what that configuration was into it. Comparing two configurations is
# then comparing two session directories, and `session.json` is what says which
# was which.
#
# The alternative was a directory named after the configuration, under each
# case. It encodes the same fact in a path — and encodes only the two switches,
# so two sessions that differed by model, by sample count, or by a fixture tree
# would land on top of each other with nothing to say they had.

#: `session-<when>`. A timestamp because sessions are compared with each other
#: and the useful order is the order they were run in. Resolved once per
#: process, so every case in one pytest invocation writes under one session —
#: and only the *name* is cached, since `artifact_root()` is monkeypatched by
#: the tests and must stay live.
_SESSION_ID: str | None = None


def session_id() -> str:
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return _SESSION_ID


def session_dir() -> Path:
    return artifact_root() / session_id()


#: What produced a session, cached for the process. A report is read weeks
#: later and compared with another, and "which code was this" is the first
#: question that makes the comparison mean anything — a delta between two
#: sessions built from different working trees is not a delta.
_CODE_VERSION: dict | None = None


def _git(root: Path, *args: str) -> str:
    """A git query that never fails a run. No git, no repo, no answer -> ``""``."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def code_version() -> dict:
    """Which code this session is running, as far as it can be established.

    Both halves are recorded because neither is sufficient. The package version
    is what the session child actually installed — `_session.staged_package()`
    builds a wheel from this checkout, so it is the thing the agent ran against
    — and setuptools_scm already folds a commit into it. But that version is
    resolved from *installed* metadata, which a checkout that has moved since
    the last install will silently under-report.

    `dirty` is the load-bearing one. A commit sha identifies the code only if
    the tree was clean; with uncommitted edits it names the nearest ancestor and
    nothing more, and two sessions can carry the same sha and have run different
    engines. Recording it is the difference between "reproduce this" and "you
    cannot".
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return _CODE_VERSION

    import biopb_mcp

    found = {"biopb_mcp": getattr(biopb_mcp, "__version__", "unknown")}
    root = checkout_root()
    if root is None:
        # An installed copy with no checkout around it. The suite is excluded
        # from the wheel, so this is near-unreachable in practice, and saying
        # "no checkout" is better than implying one.
        found["checkout"] = "none"
    elif commit := _git(root, "rev-parse", "HEAD"):
        found["commit"] = commit
        found["branch"] = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        # Untracked files count: a case module that is not committed yet still
        # changes what ran.
        found["dirty"] = bool(_git(root, "status", "--porcelain"))
    else:
        found["checkout"] = str(root)
        found["commit"] = "unknown"
    _CODE_VERSION = found
    return found


def write_session(run: Run) -> Path:
    """Record what this session is, and which cases it has run so far.

    Rewritten after every case rather than once at the end: a session that was
    interrupted — and these are long — still leaves a file that describes
    itself and lists what completed. A roster written only on the way out is a
    roster you do not get on the run you most want it for.
    """
    where = session_dir()
    where.mkdir(parents=True, exist_ok=True)
    path = where / "session.json"
    known = {}
    if path.is_file():
        try:
            known = json.loads(path.read_text(encoding="utf-8")).get("cases") or {}
        except ValueError:  # a half-written file is not worth a failed run
            known = {}
    known[run.case.label] = {
        "namespace": run.case.namespace,
        "case_id": run.case.case_id,
        "skill": run.case.skill,
        "fixture": run.fixture.kind,
        "samples": [r.classify()[0] for r in run.results],
    }
    path.write_text(
        json.dumps(
            {
                "session": session_id(),
                "code": code_version(),
                "options": run.options.as_json(),
                "configuration": run.options.configuration,
                "agent": agent_choice().name,
                "respondent": respondent_choice().name
                if run.options.responder == "model"
                else "silent",
                "cases": known,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def where_for(case: Case) -> Path:
    """Where this case's report and transcripts land, inside the session.

    Keyed on `(namespace, case_id)`, never on the skill alone: a skill covered
    two ways is two cases, and one directory for both would have the second
    silently overwrite the first's report.
    """
    return session_dir() / case.namespace / case.case_id


def catalog_ids(text: str) -> tuple[str, ...]:
    """Which skills `find_skills` returned, from the text an agent sees.

    Parsed rather than pattern-counted: whether the ablation took effect is the
    one thing that would silently make a whole table meaningless, so it must not
    rest on a substring surviving a formatting change. Text this cannot parse
    counts as *something*, never as nothing — an unreadable answer is not
    evidence that the catalog was withheld, and reading it as one would turn a
    broken ablation into a clean-looking table.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return ("<unparseable>",) if text.strip() else ()
    if isinstance(parsed, dict):
        # A list return can reach a client wrapped in structured content; the
        # entries are still the only list in it.
        parsed = next((v for v in parsed.values() if isinstance(v, list)), parsed)
    if not isinstance(parsed, list):
        return ("<unparseable>",) if parsed else ()
    return tuple(
        str(e.get("id") or e.get("name") or "?") if isinstance(e, dict) else str(e)
        for e in parsed
    )


def read_catalog(session, case: Case) -> tuple[str, ...]:
    """What the catalog offered this run. Best-effort; never fails a run."""
    try:
        return catalog_ids(session.call("find_skills", query=case.query).text)
    except Exception:  # noqa: BLE001 -- provenance is best-effort, the run is not
        return ("<unavailable>",)


# --- the fixture on the plane -----------------------------------------------

#: Per case, and per run: `label -> {layer key: array_id}`.
_UPLOADED: dict[str, dict[str, str]] = {}
#: What each uploaded fixture looked like when the harness put it there.
_FINGERPRINTS: dict[str, str] = {}


def uploaded_ids(case: Case, fixture: Fixture) -> dict[str, str]:
    """``layer key -> array_id`` for this case's `tensor` layers, uploaded once.

    Paid once per case rather than per run: a case is several runs, and these
    are the large fixtures by construction. The ids are memoised on the plane,
    so a second call over the same run returns what the first uploaded.
    """
    lazy = [layer for layer in case.layers if layer.lazy]
    if not lazy:
        return {}
    plane = _plane.ensure_plane()
    ids = _UPLOADED.setdefault(case.label, {})
    for layer in lazy:
        if layer.key not in ids:
            array_id = plane.upload(
                f"{case.namespace}-{case.case_id}-{layer.key}",
                np.asarray(fixture.data[layer.key]),
                chunks=layer.chunks,
                dim_labels=layer.dim_labels,
            )
            ids[layer.key] = array_id
            _FINGERPRINTS[array_id] = plane.fingerprint(array_id)
    return ids


def contaminated(ids: Mapping[str, str]) -> tuple[str, ...]:
    """Fixture arrays whose served bytes are no longer what was uploaded.

    A fixture id is `sha256(a per-run secret name)[:12]` and the name is never
    sent anywhere, so this should be unreachable. It is checked anyway, because
    the alternative is trusting an argument about a surface that runs arbitrary
    Python — and a run that ran against different data than it reports is not a
    weak row, it is a wrong one.
    """
    plane = _plane.running_plane()
    if plane is None:
        return ()
    changed = []
    for key, array_id in ids.items():
        try:
            now = plane.fingerprint(array_id)
        except Exception as exc:  # noqa: BLE001 - a flag, never a failed run
            changed.append(f"{key}: unreadable ({type(exc).__name__})")
            continue
        if now != _FINGERPRINTS.get(array_id):
            changed.append(key)
    return tuple(changed)


def load_fixture(session, case: Case, fixture: Fixture, ids: Mapping[str, str]) -> None:
    """Put the fixture on the viewer as *setup*, not as something the agent did.

    Injecting before handover keeps the fixture out of the agent's context and
    stops it burning turns on a setup the harness can do instantly. It goes
    through `session.setup`, recorded at turn -1, so the trace still answers
    "what did the agent do" honestly.

    A `tensor` layer is added by id through `viewer.add_tensor`, which is the
    same call the agent would make. Note the array is addressable but **not
    discoverable**: an uploaded source is deliberately not synced to the
    catalog, so `query_sources()` will not find it. The ids therefore arrive in
    the namespace under :data:`TENSOR_HANDLE` — a harness convention, exactly
    like the `collect` names, and `test_cases.py` asserts the task says so.
    """
    handles = {}
    for layer in case.layers:
        if layer.lazy:
            handles[layer.name] = ids[layer.key]
            session.setup(f"viewer.add_tensor({ids[layer.key]!r}, name={layer.name!r})")
            continue
        session.put_array("_fixture_array", np.asarray(fixture.data[layer.key]))
        session.setup(
            f"viewer.{LAYER_KINDS[layer.kind]}(_fixture_array, name={layer.name!r})"
            "\ndel _fixture_array"
        )
    if handles:
        session.setup(f"{TENSOR_HANDLE} = {handles!r}")
    session.setup("print('layers:', [lyr.name for lyr in viewer.layers])")


# --- running ----------------------------------------------------------------


def run_one(
    case: Case,
    fixture: Fixture,
    options: Options,
    sample: int = 1,
    ids: Mapping[str, str] | None = None,
) -> Result:
    """One sample, in a session of its own configured by *options*."""
    ids = {} if ids is None else ids
    plane = _plane.running_plane() if ids else None
    result = Result(sample=sample, skills_offered=options.skills)
    where = where_for(case) / result.name
    with live_session(
        skills_enabled=options.skills,
        plugins=case.plugins,
        tensor_url=plane.url if plane is not None else "",
    ) as session:
        # Read here rather than inferred from behaviour: the agent may well call
        # `find_skills` under `--bench-skills=false` and simply get nothing back,
        # and `load_catalog()` is what gates, not whether the tool was registered.
        result.catalog = read_catalog(session, case)
        load_fixture(session, case, fixture, ids)
        trace = converse(
            session,
            ToolCallingAgent(),
            respondent_for(case, options),
            with_protocol(case.task),
            max_turns=case.max_turns,
            max_tool_calls=case.max_tool_calls,
        )
        trace.write(where)
        scraped = scrape(session, trace, dict(case.collect))
        # After the scrape, so the harness's own reads of the kernel are not
        # what gets reported, and inside the session — the log dies with it.
        result.peeked = tuple(dict.fromkeys(e["path"] for e in session.peeked()))

    attempt = Attempt(
        subject=result.name,
        arrays=scraped,
        notes=f"{options.configuration} | stopped={trace.stopped}",
    )
    result.trace = trace
    result.outcome = case.score(fixture, attempt)
    write_report(result.outcome, where_for(case))
    if case.save_artifacts is not None:
        case.save_artifacts(result.outcome, where)
    # After the session is gone, so the run cannot still be writing.
    result.contaminated = contaminated(ids)
    return result


@dataclass
class Run:
    """One case's samples, and the fixture every one of them was scored against."""

    case: Case
    fixture: Fixture
    results: list[Result] = field(default_factory=list)
    options: Options = field(default_factory=Options)
    #: The report, as `run_case` left it on disk. Held so the caller that wants
    #: to print it does not re-run the writer — which would work, since it is
    #: idempotent, and would still be a second pass over every row for nothing.
    report: str = ""

    @property
    def failed_to_start(self) -> bool:
        """No sample got a session. The machine, not the case — and not a row
        worth reading, since nothing was ever asked of the agent."""
        return bool(self.results) and all(
            r.error.startswith(NO_SESSION) for r in self.results
        )

    def summary(self) -> str:
        return write_summary(self)


def progress(message: str) -> None:
    """One line of progress, straight to stdout.

    A case is one conversation per sample and the better part of half an hour,
    and without this the only sign of life is the artifact directory filling up.
    This module never imports pytest, so it emits and the *test* decides
    visibility — pytest discards a passing test's captured output, which is why
    the documented command passes ``-s``.
    """
    print(message, flush=True)


def run_case(case: Case, options: Options | None = None) -> Run:
    """Every sample of one case, in this invocation's configuration.

    A failure becomes a row, never an exception."""
    options = Options() if options is None else options
    fixture = case.build_fixture()
    # Before the first sample, and once: bringing the plane up and uploading are
    # both setup, and paying for them inside sample 1 would put a minute of
    # bring-up into that row's wall-clock and nowhere else's.
    ids = uploaded_ids(case, fixture)
    total = options.samples
    progress(
        f"\n[{case.label}] {total} sample(s), {options.configuration}, against "
        f"`{fixture.case_id}` -> {where_for(case)}"
    )
    if ids:
        progress(f"[{case.label}] on the run's plane as {sorted(ids.values())}")
    run = Run(case=case, fixture=fixture, options=options)
    for sample in range(1, total + 1):
        progress(f"[{case.label}] sample {sample}/{total}: running")
        started = time.monotonic()
        try:
            result = run_one(case, fixture, options, sample, ids)
        except SessionUnavailable as exc:
            result = Result(
                sample=sample,
                skills_offered=options.skills,
                error=f"{NO_SESSION}{exc}",
            )
        except Exception as exc:  # noqa: BLE001 - the row is the point
            result = Result(
                sample=sample,
                skills_offered=options.skills,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.seconds = time.monotonic() - started
        outcome, reason = result.classify()
        progress(
            f"[{case.label}] sample {sample}/{total}: {outcome} "
            f"in {result.seconds / 60:.1f} min — {reason}"
        )
        run.results.append(result)
    run.report = run.summary()
    write_session(run)
    return run


# --- the report -------------------------------------------------------------


def _metric_columns(results: Sequence[Result]) -> list[tuple[str, float]]:
    """``(name, limit)`` for every metric any run produced, first seen first.

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


def catalog_line(results: Sequence[Result]) -> str:
    """Which catalogs these runs saw, deduplicated.

    Recorded whether or not it was manipulated. For an ablated case the two
    answers *are* the manipulation and the table has a column for it; for a
    case with no ablation this line is what keeps a number comparable across
    releases, since a task that later gains a covering skill silently re-bases
    itself.
    """
    seen = {r.catalog for r in results}
    if not seen or seen == {()}:
        return "none offered"
    return " | ".join(sorted(", ".join(c) or "(none)" for c in seen))


def write_summary(run: Run) -> str:
    """The report. Written to `where_for(case)` and returned for printing."""
    case, fixture, results = run.case, run.fixture, run.results
    columns = _metric_columns(results)
    rows = [r.row(case.blocking_budget) for r in results]
    agent = agent_choice()
    # Resolved only under `--bench-responder=model`. Two reasons it is not a
    # plain `respondent_choice().name`: a silent run has no respondent model to
    # name, and stamping one anyway made this file disagree with its own
    # `summary.md`; and `respondent_choice()` raises on a malformed
    # BIOPB_RESPONDENT, which would take out the report of a run already paid
    # for that never had a respondent model in it.
    respondent = (
        respondent_choice().name if run.options.responder == "model" else "silent"
    )
    many = run.options.samples > 1

    def fmt(value):
        return "—" if value is None else f"{value:.4g}"

    lines = [
        f"# {case.label} — agent benchmark",
        "",
        f"Session: `{session_id()}`  ",
        f"Configuration: **{run.options.configuration}**  ",
        f"Agent under test: **{agent.name}**  ",
        f"Respondent: **{respondent}**  ",
        f"Fixture: `{fixture.case_id}` [{fixture.kind}] — "
        f"{fixture.about or 'no description'}  ",
        f"Provenance: {fixture.provenance}  ",
        f"Options: `{run.options.describe()}`  ",
        f"Skills the catalog offered: {catalog_line(results)}  ",
        "Tolerances: " + ", ".join(f"{name} ≤ {limit:g}" for name, limit in columns),
        "",
    ]
    if fixture.citation:
        # Carried into the report rather than left to whoever remembers: real
        # data comes from someone, and a result quoted without them is the
        # obligation quietly dropped.
        lines += [f"Data: {fixture.citation}", ""]
    lines += [
        "These runs are non-deterministic; read the table as an observation,",
        "not a measurement.",
        "",
        "| sample | outcome | "
        + " | ".join(name for name, _ in columns)
        + " | turns | asked | tools | min | reason |",
        "|---|---|" + "---|" * (len(columns) + 5),
    ]
    for row in rows:
        cells = " | ".join(fmt(row["metrics"].get(name)) for name, _ in columns)
        lines.append(
            f"| {row['sample']} | **{row['outcome']}** | {cells} | {row['turns']} "
            f"| {row['blocking_questions']} | {row['tool_calls']} "
            f"| {row['seconds'] / 60:.1f} | {row['reason']} |"
        )

    if flagged := [r for r in rows if r["flags"]]:
        lines += ["", "### Flags", ""]
        lines += [f"- sample-{r['sample']} — {', '.join(r['flags'])}" for r in flagged]

    lines += ["", "## Reading it", ""]
    if many:
        lines += [
            f"- {run.options.samples} samples, one configuration. The spread",
            "  between them is the finding a single row cannot carry.",
        ]
    else:
        lines += [
            "- One sample, which is not a measurement. `--bench-samples` is the",
            "  knob, and the spread between samples is routinely larger than the",
            "  difference anyone is trying to read off a single one.",
        ]
    lines += [
        # Where the delta went. It used to be two rows of this table; it is two
        # sessions now, which is why the configuration is in the header and in
        # `session.json` rather than in a column.
        "- **This report is one configuration.** A delta — a skill's, or the",
        "  cost of the withheld fact — is this session against another one run",
        f"  with a different `--bench-skills` or `--bench-responder`. `{session_id()}`",
        "  and its `session.json` are what make the pair comparable.",
    ]
    if case.about_a_skill:
        lines += [
            f"- This case claims something about `{case.skill}`, so the delta",
            "  worth having is `--bench-skills=true` against `--bench-skills=false`,",
            "  everything else held.",
        ]
    lines += [
        f"- `asked` counts blocking questions; `write-a-skill` step 4 budgets"
        f" {case.blocking_budget}.",
        "",
        "Transcripts are in `sample-N/transcript.md`, with the raw event stream",
        "in `trace.jsonl` and any images beside them.",
        "",
    ]

    text = "\n".join(lines)
    where = where_for(case)
    where.mkdir(parents=True, exist_ok=True)
    (where / "summary.md").write_text(text, encoding="utf-8")
    (where / "summary.json").write_text(
        json.dumps(
            {
                "session": session_id(),
                "case": case.label,
                "skill": case.skill,
                "namespace": case.namespace,
                "case_id": case.case_id,
                "kind": fixture.kind,
                "citation": fixture.citation,
                "provenance": fixture.provenance,
                "fixture": fixture.case_id,
                "agent": agent.name,
                "respondent": respondent,
                "options": run.options.as_json(),
                "configuration": run.options.configuration,
                "tolerance": dict(columns),
                "samples": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return text


# --- can this machine run it at all -----------------------------------------


def models_in_play(options: Options) -> tuple[tuple[str, object], ...]:
    """The `(side, choice)` pairs this configuration will actually call.

    The respondent is only one of them under `--bench-responder=model`.
    `SilentRespondent` is local and answers from a constant, so a silent session
    needs no respondent key, no respondent endpoint, and no request spent
    proving either — see `respondent_for`.
    """
    sides = [("agent", agent_choice())]
    if options.responder == "model":
        sides.append(("respondent", respondent_choice()))
    return tuple(sides)


def unavailable(case: Case, options: Options) -> str:
    """Why this case cannot be benchmarked *in this configuration*, or ``""``.

    The environment checks that are cheap and answerable before anything is
    spawned or spent. It takes the options because half of what it checks is
    about the models the configuration will reach for, and those differ by
    switch: demanding a respondent key from a `--bench-responder=silent` session
    would skip every case on a machine that can run all of them.
    """
    from ..agentbench import _session

    # First, because it is free and it is about the *case* rather than the
    # machine's model access: a case written against an acquisition this tree
    # does not have cannot run, and must never quietly run against something
    # else.
    usable, why = case.available()
    if not usable:
        return f"fixture: {why}"
    wants_plane = any(layer.lazy for layer in case.layers)
    if wants_plane and (why := _plane.plane_unavailable()):
        return f"this case is presented on a data plane, and {why}"
    if reason := _session.why_unavailable():
        return reason
    sides = models_in_play(options)
    for side, choice in sides:
        if why := choice.why_unavailable():
            return f"{side}: {why}"
    # §5a, and it constrains a *skill* case only: an agent from the family that
    # wrote these bodies could pass by recognising its own prose. A case with no
    # skill has no authored prose to recognise — only an acquisition and a
    # question — so every model is a legitimate subject for it.
    if case.about_a_skill and agent_choice().from_authoring_family:
        return (
            f"the agent is {agent_choice().name}, from the family that wrote these "
            "skills — it could pass by recognising its own prose (§5a)."
        )
    # Last, because it is the only one that costs a request: a model the
    # endpoint does not serve fails every case identically, and a shell export
    # beating the dotenv is the ordinary way to arrive there.
    for side, choice in sides:
        if why := reachable(text_backend(choice)):
            return (
                f"{side} {choice.name} at {choice.base_url or 'the provider default'} "
                f"is not usable: {why}"
            )
    return ""
