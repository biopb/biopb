"""What a case is, and what a run may vary about it. No engine, no pytest.

One :class:`Case` covers both questions this layer asks, because they turned out
to be the same object with one field set differently:

* **does *this skill* change what an agent does** — ``skill`` names it, and two
  runs either side of ``--bench-skills`` are its delta;
* **can an agent do *this work*** — ``skill`` is empty, and repetition rather
  than a control is where the information comes from.

They were two dataclasses in two packages for a while (`Case` and `TaskCase`),
with two engines behind them that had drifted in the small: two outcome
vocabularies that agreed by hand, two classifiers that ordered `turn-cap` and
`wrong-answer` differently, two report writers, two copies of the upload and
contamination checks. The second engine was written by copying the first, which
is exactly the cost this file exists to stop paying a third time.

**Nothing here decides how a run is configured.** Whether the catalog is offered
and who answers the agent are switches on the invocation (`_options.py`), not
properties of the case — so ``skill`` no longer changes what a run *does*, only
what the case *is*. It labels, and labels are what filters and ledgers read.

That was the last thing a case took from the skills tree. There was an arm set
here that a skill case got four of and every other case got one of; it meant a
case's kind decided what a run cost, and that a report had to explain a table
whose rows were configured differently from one another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..agentbench._fixture import Attempt, Fixture, FixtureSpec, Outcome
from ..agentbench._respondent import Persona

#: Bounds on one run. Not the loop's own defaults: these workflows promote
#: compute to background jobs, and a run severed mid-workflow leaves
#: half-finished arrays that say nothing about the skill. A cap that stops a
#: stuck model is the point; a cap that stops a working one only produces
#: unreadable results.
MAX_TURNS = 90
MAX_TOOL_CALLS = 200

#: `write-a-skill` step 4 budgets at most three blocking checkpoints. Reported,
#: not asserted: a handful of samples cannot support a verdict.
BLOCKING_BUDGET = 3

#: The namespace a case falls back to when it names neither a skill nor one of
#: its own. A curated case is found at ``$BIOPB_FIXTURES/<namespace>/<case_id>/``
#: and every artifact path starts the same way (`docs/fixtures.md`), so this is
#: a directory name on somebody's disk and not merely a label.
TASK_NAMESPACE = "tasks"

#: The two environments a fixture array can arrive in. Peers, not a ladder:
#: neither is a default and neither substitutes for the other. The right one is
#: whichever the skill was written against — a skill about in-memory numpy is
#: correctly tested with `array`, and a viewer layer holding a plain array is a
#: real thing an agent meets, not a concession. A skill written for lazy data
#: gets `tensor`, which is the real path (`client.get_tensor`) with a real
#: server behind it; there is deliberately no faked-lazy option in between.
PRESENTATIONS = ("array", "tensor")

#: Where a `tensor` case's array ids arrive in the kernel namespace, as
#: ``{layer name: array_id}``. A convention rather than something a skill
#: claims, so a case that presents `tensor` has to say so in its task prompt —
#: an id minted at run time cannot be written into a prompt in advance, and an
#: agent told nothing would have to guess at a source the catalog does not list.
TENSOR_HANDLE = "fixture_tensors"

#: Layer kinds the harness knows how to put on a viewer. `points` is not
#: cosmetic: a Points layer is how a person's clicked correspondences actually
#: reach napari, and a task about landmarks handed a raw (N, 2) array would be
#: testing a different route.
LAYER_KINDS = {"image": "add_image", "labels": "add_labels", "points": "add_points"}


@dataclass(frozen=True)
class Layer:
    """One fixture array, as the agent finds it on the viewer.

    ``kind`` picks the `add_*` call, which is not cosmetic: a Labels layer is
    what makes a segmentation addressable as objects, and several skills'
    Parameters tables ask for one by name.

    ``presentation`` is part of the case for the same reason the fixture is:
    changing it changes what is being measured.
    """

    name: str
    key: str
    kind: str = "image"
    presentation: str = "array"
    #: `tensor` only, and explicit rather than automatic — where laziness is the
    #: point, the chunking *is* the thing under test. `None` uploads the array
    #: as one chunk, which is the right choice only when the case is not about
    #: chunk boundaries.
    chunks: tuple[int, ...] | None = None
    #: `tensor` only: the axis semantics the plane will echo back. The server
    #: rejects a non-canonical order, so this is the case's declaration of what
    #: its array's axes mean.
    dim_labels: tuple[str, ...] | None = None

    @property
    def lazy(self) -> bool:
        return self.presentation == "tensor"


@dataclass(frozen=True)
class Case:
    """One case's whole contribution to this layer.

    Everything that is *about the subject* and nothing that is about running a
    benchmark. Adding one is writing a single module under `cases/` — see
    `cases/drift_correction.py` for a skill and
    `cases/align_channels_from_landmarks.py` for a task.

    **A case is non-decomposable.** Task, persona, fixture, verifier and
    tolerances are one artifact, and where the pixels come from — a procedure
    or a file on disk — is decided here, when the case is written, never
    resolved at run time. Covering one subject both ways is *two cases*, each
    with its own `case_id`, and `(namespace, case_id)` is what names a run's
    artifacts (`docs/fixtures.md`).

    The fixture is a spec rather than a built value so a case module costs
    nothing at import: 30 of these are collected by every ordinary test run,
    and only the one being benchmarked should build megabytes of arrays.
    """

    #: What this case is: names the run and its artifacts, and — for a curated
    #: fixture — locates its data on disk.
    case_id: str
    #: What the agent is asked to do, including where its results should land.
    task: str
    #: Who it is talking to. For a skill case, the holder of the fact the
    #: fixture strips out; for a task, the experimental context and no answer.
    persona: Persona
    #: Where this case's data comes from. Exactly one, and no fallback:
    #: substituting it would make a different experiment with the same name.
    fixture: FixtureSpec
    #: Where the fixture's arrays land on the viewer, in order.
    layers: Sequence[Layer]
    #: What the verifier wants -> the kernel expression that yields it.
    collect: Mapping[str, str]
    #: ``(fixture, attempt) -> Outcome``. Numeric, never judged prose: these
    #: cases emit numbers with knowable right answers.
    score: Callable[[Fixture, Attempt], Outcome]

    #: The **served** skill this case is a claim about, as `find_skills` and
    #: `skill://<id>` know it.
    #:
    #: **It changes nothing about a run.** It says what the case *is*, and three
    #: things read that: `--bench-cases=skills|tasks`, the coverage ledger that
    #: keeps a growing catalogue honest, and §5a's refusal to score an agent
    #: from the family that wrote these bodies. A run's configuration comes from
    #: the switches, and is the same whatever this says.
    #:
    #: **Empty is meaningful**: there is no served entry to withhold, so
    #: `--bench-skills=false` against such a case ablates nothing and its two
    #: runs are the same run. A case written alongside a *banked* skill — one
    #: behind the `_` marker, which the runtime does not serve — leaves this
    #: empty and puts the skill's name in :attr:`namespace` instead.
    skill: str = ""
    #: Where this case's data and artifacts live, when it is not a skill's:
    #: ``$BIOPB_FIXTURES/<namespace>/<case_id>/``. Defaults to the skill id, and
    #: to `tasks` for a case that names neither. It is a directory somebody has
    #: on disk, so a banked skill's case keeps the skill's own name here and
    #: nothing moves when the skill is later promoted.
    namespace: str = ""

    #: Optional ``(outcome, dir) -> None`` — the before/after images.
    save_artifacts: Callable[[Outcome, Path], None] | None = None
    #: Kernel plugins the skill's `checklist:` names, seeded into the session's
    #: own config tree. Without this a `plugin:` token is unresolvable and the
    #: run is scoring an environment the skill declares it cannot work in.
    plugins: Sequence[str] = ()
    #: What to ask `find_skills`, to check `--bench-skills` actually took effect.
    #: Defaults to the skill id, and is empty for a case with no skill — where
    #: the catalog is recorded as provenance rather than manipulated, so the
    #: honest query is the one that lists everything.
    catalog_query: str = ""
    #: Case-folded substrings that must appear in the persona's rendered
    #: prompt: the fact the fixture strips, so the run is answerable at all.
    #: Required of a skill case, and checked wherever it is declared — a case
    #: with no skill may still withhold something, and several do.
    persona_must_know: Sequence[str] = ()
    #: Case-folded substrings that must **not**. A persona that has absorbed the
    #: procedure can answer a question the agent never properly asked, and the
    #: numeric result stops meaning what it appears to. Name the skill's own
    #: vocabulary here — `test_cases` asserts it, hermetically and free.
    persona_must_not_know: Sequence[str] = ()
    blocking_budget: int = BLOCKING_BUDGET
    max_turns: int = MAX_TURNS
    max_tool_calls: int = MAX_TOOL_CALLS

    def __post_init__(self):
        # Resolved once, into the field itself, so `case.namespace` is a plain
        # attribute everywhere and there is no second spelling of the fallback.
        if not self.namespace:
            object.__setattr__(self, "namespace", self.skill or TASK_NAMESPACE)

    @property
    def about_a_skill(self) -> bool:
        """Whether there is a claim here to isolate, and so an ablation to run."""
        return bool(self.skill)

    @property
    def query(self) -> str:
        return self.catalog_query or self.skill

    @property
    def label(self) -> str:
        return f"{self.namespace}/{self.case_id}"

    def build_fixture(self) -> Fixture:
        """This case's one fixture, stamped with the identity declared above."""
        return self.fixture.build(self.namespace, self.case_id)

    def available(self) -> tuple[bool, str]:
        """Whether this machine has this case's data, and why not."""
        return self.fixture.available(self.namespace, self.case_id)
