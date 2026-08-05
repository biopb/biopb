"""The engine: one task, one arm, N samples, and a report.

What this is *not* is the thing next door. `skills/interaction` runs a 2x2
because a skill's claim is a behavioural delta, and a delta needs a control.
A task has no claim to isolate — the question is whether an agent can do a
named piece of work against real data — so there is one arm and nothing is
withheld. What replaces the ablation as the source of information is repetition:
`samples` runs the same task again, and the spread between runs is the finding
a single number cannot carry.

**The session is the shipped one.** `skills_enabled` stays on, because this
measures the product a user actually has rather than a model in a stripped
environment. That choice has a cost and the report pays it explicitly: every
summary records which skills the catalog offered at run time
(:func:`catalog_state`), so a task that later gains a skill cannot silently
re-base its own number. A score compared across releases without reading that
line is a score compared against a different system.

Nothing here fails a test. A run that timed out, gave up or got the answer
wrong is a *result*, and the report is the deliverable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..agentbench import _plane
from ..agentbench._agent import ToolCallingAgent
from ..agentbench._conversation import (
    AGENT_TRUNCATED,
    FINISHED,
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
    FixtureSpec,
    Outcome,
    artifact_root,
    write_report,
)
from ..agentbench._models import (
    agent_choice,
    reachable,
    respondent_choice,
    text_backend,
)
from ..agentbench._respondent import Persona, model_respondent
from ..agentbench._session import SessionUnavailable, live_session

MAX_TURNS = 90
MAX_TOOL_CALLS = 200

#: How a sample ended. Deliberately the same vocabulary the interaction layer
#: reports, so a reader who knows one report can read the other.
OK = "ok"
WRONG_ANSWER = "wrong-answer"
OUT_OF_TURNS = "out-of-turns"
OUT_OF_TOOL_CALLS = "out-of-tool-calls"
GAVE_UP = "gave-up"
NO_RESULT = "no-result"
UNSCORABLE = "unscorable-result"
HARNESS_ERROR = "harness-error"

FLAG_CUT_OFF = "cut-off-but-scored"
FLAG_STALLED = "stalled"
#: The kernel read something the harness owns. Voids rather than qualifies: a
#: run that read its own answer key measured nothing. `execute_code` is
#: arbitrary Python by design, so the defence is that it cannot happen quietly.
FLAG_PEEKED = "read-harness-internals"
#: A fixture array on the plane is no longer the bytes the harness put there.
#: Voids this sample *and every later one*, since the plane outlives a session.
FLAG_CONTAMINATED = "fixture-overwritten"

NO_SESSION = "session unavailable: "

PRESENTATIONS = ("array", "tensor")
#: Where a `tensor` case's ids arrive in the kernel, as ``{layer name: id}``.
#: An id minted at run time cannot be written into a prompt in advance, so a
#: case presenting `tensor` has to name this handle in its task text.
TENSOR_HANDLE = "fixture_tensors"

SAMPLES_ENV = "BIOPB_TASK_SAMPLES"


@dataclass(frozen=True)
class Layer:
    """One fixture array, as the agent finds it on the viewer."""

    name: str
    key: str
    #: `image`, `labels` or `points`. Not cosmetic: a Points layer is how a
    #: person's clicked correspondences actually reach napari, and a task about
    #: landmarks handed a raw (N, 2) array would be testing a different route.
    kind: str = "image"
    presentation: str = "array"
    chunks: tuple[int, ...] | None = None
    dim_labels: tuple[str, ...] | None = None

    @property
    def lazy(self) -> bool:
        return self.presentation == "tensor"


@dataclass(frozen=True)
class TaskCase:
    """One task: what is asked, what it is given, and how the result is scored.

    Everything that is *about this task* and nothing that is about running one.
    Adding a task is writing one of these under `cases/` — there is no
    registration step, the module is found by being there.
    """

    case_id: str
    #: The prompt, including where results are to be left.
    task: str
    #: Who answers when the agent asks. Holds the experimental context in full
    #: and volunteers none of it — the task text is self-sufficient, so asking
    #: neither rescues nor penalises a run, it only makes it more like a session.
    persona: Persona
    fixture: FixtureSpec
    layers: tuple[Layer, ...]
    #: ``{report key: kernel name}`` the harness scrapes when the run ends.
    collect: Mapping[str, str]
    score: Callable[[Fixture, Attempt], Outcome]
    save_artifacts: Callable[[Outcome, Path], None] | None = None
    plugins: tuple[str, ...] = ()
    max_turns: int = MAX_TURNS
    max_tool_calls: int = MAX_TOOL_CALLS

    #: The namespace this case's curated data sits under in the fixture tree,
    #: and the first half of its artifact path.
    namespace: str = "tasks"

    @property
    def label(self) -> str:
        return f"{self.namespace}/{self.case_id}"

    def build_fixture(self) -> Fixture:
        return self.fixture.build(self.namespace, self.case_id)

    def available(self) -> tuple[bool, str]:
        return self.fixture.available(self.namespace, self.case_id)


@dataclass
class Sample:
    """One run of one task."""

    index: int
    outcome: Outcome | None
    stopped: str = ""
    seconds: float = 0.0
    error: str = ""
    peeked: tuple[str, ...] = ()
    contaminated: tuple[str, ...] = ()
    catalog: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.error:
            return HARNESS_ERROR
        # A provider ending the run looks exactly like a model deciding to, and
        # is not the agent's doing. Classified before anything else, so a
        # truncated completion never gets reported as a capability.
        if self.stopped in (AGENT_TRUNCATED, RESPONDENT_FAILED):
            return HARNESS_ERROR
        if self.outcome is None:
            return NO_RESULT
        if not self.outcome.metrics:
            return UNSCORABLE
        if not self.outcome.scored:
            # Nothing scorable came back. Whether that is the agent stopping
            # early or never producing anything is worth keeping apart.
            return GAVE_UP if self.stopped in (FINISHED, SILENT) else NO_RESULT
        if self.outcome.passed:
            return OK
        if self.stopped == TURN_CAP:
            return OUT_OF_TURNS
        if self.stopped == TOOL_CAP:
            return OUT_OF_TOOL_CALLS
        return WRONG_ANSWER

    @property
    def flags(self) -> tuple[str, ...]:
        out = []
        if self.peeked:
            out.append(FLAG_PEEKED)
        if self.contaminated:
            out.append(FLAG_CONTAMINATED)
        if self.stopped == STALLED:
            out.append(FLAG_STALLED)
        if self.stopped in (TURN_CAP, TOOL_CAP) and self.status == OK:
            out.append(FLAG_CUT_OFF)
        return tuple(out)


def where_for(case: TaskCase) -> Path:
    """Where this task's report and transcripts land.

    A root of its own rather than a subdirectory of the skills one: these
    answer a different question and are read on different occasions, and one
    tree holding both invites a reader to compare rows that are not comparable.
    """
    return artifact_root().with_name(".task-outcomes") / case.case_id


def samples_wanted() -> int:
    """How many times to run each task. One unless asked for more.

    A single sample is the right default for iterating on a case; it is not a
    measurement. Raise it when a number is going to be quoted, because the
    spread between runs of the same task is routinely larger than the
    difference anyone is trying to read off it.
    """
    import os

    raw = os.environ.get(SAMPLES_ENV, "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def catalog_state(session) -> tuple[str, ...]:
    """Which skills the catalog offered, recorded rather than controlled.

    This suite runs with the shipped configuration, so a skill covering the
    task would help the agent and change what the number means. Nothing stops
    that -- it is the configuration a user has -- but a run that does not say
    which catalog it saw cannot be compared with one from another release.
    """
    try:
        text = session.call("find_skills", query="").text
    except Exception:  # noqa: BLE001 -- provenance is best-effort, the run is not
        return ("<unavailable>",)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # Unreadable is not the same as empty. Recording "" here would claim the
        # catalog was bare, which is exactly the false reassurance this exists
        # to prevent.
        return ("<unparseable>",) if text.strip() else ()
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), parsed)
    if not isinstance(parsed, list):
        return ("<unparseable>",) if parsed else ()
    return tuple(
        str(e.get("id") or e.get("name") or "?") if isinstance(e, dict) else str(e)
        for e in parsed
    )


def uploaded_ids(case: TaskCase, fixture: Fixture) -> dict[str, str]:
    """``layer key -> array_id`` for this case's `tensor` layers, uploaded once.

    Paid once per case rather than per sample: these are the large fixtures by
    construction, and N samples would otherwise upload the same bytes N times.
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


_UPLOADED: dict[str, dict[str, str]] = {}
_FINGERPRINTS: dict[str, str] = {}


def contaminated(ids: Mapping[str, str]) -> tuple[str, ...]:
    """Fixture arrays whose served bytes are no longer what was uploaded."""
    plane = _plane.running_plane()
    if plane is None:
        return ()
    changed = []
    for key, array_id in ids.items():
        try:
            now = plane.fingerprint(array_id)
        except Exception as exc:  # noqa: BLE001 -- a flag, never a failed run
            changed.append(f"{key}: unreadable ({type(exc).__name__})")
            continue
        if now != _FINGERPRINTS.get(array_id):
            changed.append(key)
    return tuple(changed)


def load_fixture(session, case: TaskCase, fixture: Fixture, ids: Mapping[str, str]):
    """Put the fixture on the viewer as *setup*, not as something the agent did.

    Injecting before handover keeps the fixture out of the agent's context and
    stops it spending turns on a setup the harness can do instantly. It goes
    through `session.setup`, recorded at turn -1, so the trace still answers
    "what did the agent do" honestly.
    """
    handles = {}
    for layer in case.layers:
        if layer.lazy:
            handles[layer.name] = ids[layer.key]
            session.setup(f"viewer.add_tensor({ids[layer.key]!r}, name={layer.name!r})")
            continue
        session.put_array("_fixture_array", np.asarray(fixture.data[layer.key]))
        adder = {"labels": "add_labels", "points": "add_points"}.get(
            layer.kind, "add_image"
        )
        session.setup(
            f"viewer.{adder}(_fixture_array, name={layer.name!r})\ndel _fixture_array"
        )
    if handles:
        session.setup(f"{TENSOR_HANDLE} = {handles!r}")
    session.setup("print('layers:', [lyr.name for lyr in viewer.layers])")


def run_sample(
    case: TaskCase, fixture: Fixture, index: int, ids: Mapping[str, str] | None = None
) -> Sample:
    """One run, in its own session. Every failure becomes a row, not a raise."""
    ids = {} if ids is None else ids
    started = time.time()
    plane = _plane.running_plane() if ids else None
    try:
        with live_session(
            skills_enabled=True,
            plugins=case.plugins,
            tensor_url=plane.url if plane is not None else "",
        ) as session:
            catalog = catalog_state(session)
            load_fixture(session, case, fixture, ids)
            trace = converse(
                session,
                ToolCallingAgent(),
                model_respondent(case.persona),
                with_protocol(case.task),
                max_turns=case.max_turns,
                max_tool_calls=case.max_tool_calls,
            )
            trace.write(where_for(case) / f"sample-{index}")
            scraped = scrape(session, trace, dict(case.collect))
            peeked = tuple(dict.fromkeys(e["path"] for e in session.peeked()))
    except SessionUnavailable as exc:
        return Sample(index=index, outcome=None, error=f"{NO_SESSION}{exc}")
    except Exception as exc:  # noqa: BLE001 -- a dead sample is a row, not a crash
        return Sample(
            index=index,
            outcome=None,
            error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.time() - started, 1),
        )

    attempt = Attempt(
        subject=f"sample-{index}",
        arrays=scraped,
        notes=f"stopped={trace.stopped}",
    )
    outcome = case.score(fixture, attempt)
    write_report(outcome, where_for(case))
    if case.save_artifacts is not None:
        case.save_artifacts(outcome, where_for(case) / f"sample-{index}")
    return Sample(
        index=index,
        outcome=outcome,
        stopped=trace.stopped,
        seconds=round(time.time() - started, 1),
        peeked=peeked,
        contaminated=contaminated(ids),
        catalog=catalog,
    )


@dataclass
class Run:
    """One task's samples, and the fixture they were scored against."""

    case: TaskCase
    fixture: Fixture
    samples: list[Sample] = field(default_factory=list)
    summary: str = ""


def progress(message: str) -> None:
    print(message, flush=True)


def run_case(case: TaskCase) -> Run:
    """Every sample of one task, then the report. Needs `-s` to be watched."""
    fixture = case.build_fixture()
    ids = uploaded_ids(case, fixture)
    n = samples_wanted()
    progress(f"[{case.case_id}] {n} sample(s) -> {where_for(case)}")
    run = Run(case=case, fixture=fixture)
    for index in range(1, n + 1):
        progress(f"[{case.case_id}] {index}/{n}: running")
        sample = run_sample(case, fixture, index, ids)
        run.samples.append(sample)
        note = sample.error or ", ".join(
            str(m) for m in (sample.outcome.metrics if sample.outcome else [])
        )
        progress(
            f"[{case.case_id}] {index}/{n}: {sample.status} in {sample.seconds / 60:.1f} min — {note}"
        )
    run.summary = write_summary(run)
    return run


def _metric_columns(samples: Sequence[Sample]) -> list[tuple[str, float]]:
    """``(name, limit)`` for every metric any sample produced, first seen first.

    Read off the metrics rather than declared on the case: a verifier reports
    what the fixture's truth supports, so the table follows it.
    """
    seen: dict[str, float] = {}
    for sample in samples:
        if sample.outcome is None:
            continue
        for metric in sample.outcome.metrics:
            seen.setdefault(metric.name, metric.limit)
    return list(seen.items())


def write_summary(run: Run) -> str:
    """The report. Written to `where_for(case)` and returned for printing."""
    case, fixture, samples = run.case, run.fixture, run.samples
    columns = _metric_columns(samples)
    head = ["sample", "status", "min"] + [f"{n} (<={lim:g})" for n, lim in columns]
    rows = []
    for sample in samples:
        cells = [f"sample-{sample.index}", sample.status, f"{sample.seconds / 60:.1f}"]
        for name, _ in columns:
            metric = next(
                (
                    m
                    for m in (sample.outcome.metrics if sample.outcome else [])
                    if m.name == name
                ),
                None,
            )
            cells.append(
                "-"
                if metric is None
                else ("n/a" if not metric.scored else f"{metric.value:.4g}")
            )
        if flags := sample.flags:
            cells[1] = f"{cells[1]} [{', '.join(flags)}]"
        rows.append(cells)

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(head)
    ]
    table = ["  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True))]
    table += [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows
    ]

    catalogs = {s.catalog for s in samples if s.catalog}
    catalog_line = (
        "no skills offered"
        if not catalogs
        else " | ".join(", ".join(c) or "(none)" for c in catalogs)
    )
    text = "\n".join(
        [
            f"# {case.label}",
            "",
            fixture.about or "",
            "",
            f"- fixture: {fixture.kind}",
            f"- citation: {fixture.citation}",
            f"- provenance: {fixture.provenance}",
            f"- agent: {agent_choice()}   respondent: {respondent_choice()}",
            f"- skills the catalog offered: {catalog_line}",
            "",
            "```",
            *table,
            "```",
            "",
            "No row here fails a test. A task that timed out, gave up or got the",
            "answer wrong is a result; the report is the deliverable.",
            "",
        ]
    )
    root = where_for(case)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "case": case.label,
                "kind": fixture.kind,
                "citation": fixture.citation,
                "agent": agent_choice(),
                "respondent": respondent_choice(),
                "samples": [
                    {
                        "index": s.index,
                        "status": s.status,
                        "flags": list(s.flags),
                        "seconds": s.seconds,
                        "error": s.error,
                        "catalog": list(s.catalog),
                        "metrics": [
                            {"name": m.name, "value": m.value, "limit": m.limit}
                            for m in (s.outcome.metrics if s.outcome else [])
                        ],
                    }
                    for s in samples
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return text


def unavailable(case: TaskCase) -> str:
    """Why this machine cannot run *case*, or ``""``.

    The cheap, answerable-before-spending checks, in that order. Note what is
    *not* here: the interaction layer refuses an agent from the family that
    wrote the skills, because it could pass by recognising its own prose. No
    such rule applies to a task — there is no authored prose to recognise, only
    an acquisition and a question — so every model is a legitimate subject.
    """
    from ..agentbench import _session

    # First, because it is free and it is about the *case* rather than the
    # machine: a task written against an acquisition this tree does not have
    # cannot run, and must never quietly run against something else.
    ok, why = case.available()
    if not ok:
        return f"fixture: {why}"
    if any(layer.lazy for layer in case.layers) and (why := _plane.plane_unavailable()):
        return f"this case is presented on a data plane, and {why}"
    if reason := _session.why_unavailable():
        return reason
    for side, choice in (
        ("agent", agent_choice()),
        ("respondent", respondent_choice()),
    ):
        if why := choice.why_unavailable():
            return f"{side}: {why}"
    # Last, because it is the only one that costs a request.
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
