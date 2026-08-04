"""Everything asserted about a case's *data*, before anyone pays for a run.

Hermetic and instant, and they run with the ordinary suite. A case is a task
prompt, a persona, a fixture and a verifier, and each of those has a way of
being quietly wrong that no amount of running would reveal:

- a persona that volunteers what it was not asked rescues a bad agent, and a
  green run looks identical;
- a fixture whose truth is visible in its data scores runs on a question they
  could read the answer to;
- a verifier that passes an empty attempt makes every arm look fine;
- a task that asks for one name while the harness scrapes another scores a run
  on something it was never told to bind.

Every check runs over `cases.CASES`, so a skill added to the benchmark is
checked by arriving rather than by someone remembering to write a test for it.
That is the property this layer needs to survive a catalogue of thirty.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from .._validate import NOT_SKILLS, validate
from ..conftest import SKILLS_DIR
from ._benchmark import PRESENTATIONS, TENSOR_HANDLE
from ._fixture import Attempt, Fixture
from .cases import CASES, NOT_BENCHMARKED


def _ids(case):
    return case.label


#: Built once each: a fixture is megabytes of numpy and several cases share
#: these tests. Keyed on `(skill, case_id)` — a skill covered two ways is two
#: cases, and keying on the skill alone would hand the second the first's data.
_FIXTURES: dict[tuple[str, str], Fixture] = {}


@pytest.fixture(params=CASES, ids=_ids)
def case(request):
    return request.param


def _case(label: str):
    return next(c for c in CASES if c.label == label)


def _built(label: str) -> Fixture:
    case = _case(label)
    key = (case.skill, case.case_id)
    if key not in _FIXTURES:
        _FIXTURES[key] = case.build_fixture()
    return _FIXTURES[key]


@pytest.fixture
def built(case) -> Fixture:
    """This case's own fixture — skipped, never substituted, when the machine
    cannot produce it."""
    usable, why = case.fixture.available(case.skill, case.case_id)
    if not usable:
        pytest.skip(f"{case.label}: {why}")
    return _built(case.label)


def test_there_is_at_least_one_case():
    """The guard against this file going vacuously green, the same shape as
    `test_the_extractor_finds_pkg_tokens` in the contract layer."""
    assert CASES


# --- the catalogue is covered ----------------------------------------------


def _shipped() -> set[str]:
    # `_`-prefixed files are deferred: written and banked, but not served by the
    # runtime, so this layer owes them nothing. Their case module carries the
    # same prefix and lands in `cases.DEFERRED_CASES` for the same reason.
    return {
        p.stem
        for p in SKILLS_DIR.glob("*.md")
        if p.stem not in NOT_SKILLS and not p.name.startswith("_")
    }


def test_every_shipped_skill_is_benchmarked_or_declared_unbenchmarkable():
    """A skill outside this layer is a decision, and it has to be written down.

    Without this the honest answer to "what does the benchmark cover" is
    "whatever anyone got round to", which is indistinguishable from full
    coverage when read off a green suite.
    """
    uncovered = _shipped() - {c.skill for c in CASES} - set(NOT_BENCHMARKED)
    assert not uncovered, (
        f"these shipped skills are neither benchmarked nor listed in "
        f"cases.NOT_BENCHMARKED with a reason: {sorted(uncovered)}"
    )


def test_nothing_claims_to_cover_a_skill_that_does_not_ship():
    """A case or an exemption left behind by a deleted skill. Cheap to check,
    and it is how the contract layer's module came to be written entirely for a
    skill that no longer existed."""
    shipped = _shipped()
    stale = ({c.skill for c in CASES} | set(NOT_BENCHMARKED)) - shipped
    assert not stale, (
        f"these name skills that are not in the catalogue: {sorted(stale)}"
    )


def test_an_exemption_carries_a_reason():
    for skill, why in NOT_BENCHMARKED.items():
        assert len(why.split()) >= 5, f"{skill}: {why!r} does not say why"


#: `checklist:` tokens that say a skill expects a data plane — to read lazily
#: from one, to upload a result to one, or both. They map to the same
#: presentation, because the tensor path is the only place either happens:
#: `client.get_tensor` is what hands a session a dask array in production, and
#: `client is None` is what an `array` case gives instead.
LAZY_TOKENS = ("dask", "tensor")


def test_a_skill_written_for_lazy_data_reports_whether_a_case_presents_it():
    """A coverage ledger, and it **warns rather than fails**.

    The unit is the skill, not the case: a case presenting `array` for a skill
    that declares `dask` is not wrong, it tests the in-memory branch, which is
    a real branch. What it is, is *incomplete* — and since a skill may have
    several cases, the fix is another case rather than a correction to this
    one. A gate here would demand cases nobody has written yet and would punish
    an honest partial benchmark exactly as hard as a wrong one.

    It belongs beside `NOT_BENCHMARKED`, which records "this skill is outside
    the layer, and why". This records "this skill is *partly* inside it, and
    which part".
    """
    entries, _ = validate(SKILLS_DIR)
    lazy_cases = {c.skill for c in CASES if any(layer.lazy for layer in c.layers)}
    for entry in entries:
        declared = [t for t in LAZY_TOKENS if t in entry.checklist]
        if (
            not declared
            or entry.id in lazy_cases
            or entry.id not in {c.skill for c in CASES}
        ):
            continue
        warnings.warn(
            f"{entry.id} declares {declared}, but every case presents `array`, "
            "so every arm runs with `client is None` — neither the lazy read "
            "path nor any step that uploads a result has been benchmarked",
            stacklevel=1,
        )


def test_no_two_cases_share_an_identity():
    """`(skill, case_id)` names a run's artifacts, its cached fixture and its
    report. Two cases colliding on it would overwrite one report with the
    other's and hand the second run the first's data — silently, since nothing
    downstream can tell the two apart."""
    seen = [c.label for c in CASES]
    duplicated = sorted({label for label in seen if seen.count(label) > 1})
    assert not duplicated, f"these cases collide on (skill, case_id): {duplicated}"


def test_every_case_names_itself():
    for case in CASES:
        assert case.case_id.strip(), f"{case.skill}: a case with no case_id"


def test_a_fixture_tree_does_not_change_what_a_procedural_case_runs(
    tmp_path, monkeypatch
):
    """The regression this whole design exists to prevent.

    `$BIOPB_SKILL_FIXTURES` used to be a policy switch: whatever sat under it
    replaced a case's own fixture, silently and per machine. **Substituting the
    data makes it a different experiment with the same name** — the truth
    changes, the achievable accuracy changes, and the conclusion can invert,
    which was measured rather than supposed (`docs/skill-fixtures.md`). It is a
    root path now, and a procedural case must not so much as look at it.
    """
    case = CASES[0]
    decoy = tmp_path / case.skill / case.case_id
    decoy.mkdir(parents=True)
    (decoy / "case.json").write_text('{"data": {}, "truth": {}}', encoding="utf-8")
    monkeypatch.setenv("BIOPB_SKILL_FIXTURES", str(tmp_path))

    built = case.build_fixture()
    assert built.kind == "synthetic"
    assert built.data, "the case built nothing, so the decoy was consulted"


# --- the case is runnable --------------------------------------------------


def test_every_case_is_complete_enough_to_run(case):
    """The fields a run cannot proceed without, checked without running one."""
    assert case.task.strip(), f"{case.skill}: no task prompt"
    assert case.layers, f"{case.skill}: no fixture layer to load"
    assert case.collect, f"{case.skill}: nothing would be collected"
    assert callable(case.score)
    assert callable(getattr(case.fixture, "build", None)), (
        f"{case.label}: `fixture` is not a FixtureSpec"
    )
    assert case.query


def test_the_task_asks_for_exactly_what_is_collected(case):
    """The scrape names are a **harness convention**, not a claim the skill
    makes — so the prompt has to state them, or the run is scored on names the
    agent was never told to bind."""
    for expression in case.collect.values():
        assert expression in case.task, (
            f"{case.skill}: the task never mentions {expression!r}, "
            "which is where its result is read from"
        )


def test_every_layer_kind_is_one_the_harness_can_add(case):
    for layer in case.layers:
        assert layer.kind in ("image", "labels"), (
            f"{case.skill}: layer {layer.name!r} wants a {layer.kind!r} layer"
        )


def test_every_layer_is_presented_in_a_way_the_harness_can_produce(case):
    for layer in case.layers:
        assert layer.presentation in PRESENTATIONS, (
            f"{case.label}: layer {layer.name!r} asks for "
            f"{layer.presentation!r}, not one of {PRESENTATIONS}"
        )


def test_a_case_on_a_plane_tells_the_agent_where_its_data_is(case):
    """A `tensor` fixture is addressable but **not discoverable** — an uploaded
    source is deliberately not synced to the catalog, so `query_sources()` will
    not find it. The ids arrive in the namespace instead, and an agent nobody
    told would be scored on failing to guess at a source it could not list."""
    if not any(layer.lazy for layer in case.layers):
        return
    assert TENSOR_HANDLE in case.task, (
        f"{case.label}: presents a layer on the data plane, but its task never "
        f"mentions `{TENSOR_HANDLE}`, which is where the array ids arrive"
    )


def test_nothing_asks_for_chunking_it_will_not_get(case):
    """`chunks` is uploaded to the plane, so on an `array` layer it is a silent
    no-op — and a silent no-op on the parameter that decides whether the
    out-of-core route is exercised at all is the worst kind."""
    for layer in case.layers:
        if layer.lazy:
            continue
        assert layer.chunks is None and layer.dim_labels is None, (
            f"{case.label}: layer {layer.name!r} sets chunks/dim_labels but is "
            f"presented as {layer.presentation!r}, where neither reaches anything"
        )


# --- the fixture -----------------------------------------------------------


def test_the_fixture_keeps_its_truth_out_of_the_data(built):
    """The leak the whole layer depends on not happening. A truth key visible in
    `data` means every run scores perfectly on a question it could read the
    answer to."""
    shared = set(built.data) & set(built.truth)
    assert not shared, f"{built.label}: {sorted(shared)} is both given and withheld"


#: Widest run of identical rows or columns tolerated at a frame border. A
#: synthetic field is sparse, so some edge really is flat background: measured
#: over six seeds, every frame of every channel stays under 6 px. Rendering
#: frame-sized and shifting in place instead of cropping a padded canvas reached
#: 25 px, and the width tracked the offset.
MAX_FLAT_BORDER_PX = 10


def _flat_border_px(frame, tol=1e-3) -> int:
    """Rows at the top edge of `frame` that are copies of their neighbour."""
    varies = np.abs(np.diff(frame, axis=0)).mean(axis=1) > tol
    return int(np.argmax(varies)) if varies.any() else frame.shape[0]


def test_the_drifted_movie_invents_no_pixels():
    """The same leak as `test_the_fixture_keeps_its_truth_out_of_the_data`, by
    the other route: not a truth *key* left in `data`, but the truth painted
    into the pixels.

    A stage that moves reveals sample that was outside the field of view; it
    does not create pixels. Shift a frame-sized image and the interpolator has
    to invent the vacated border, and the width of what it invents *is* the
    shift — the withheld trajectory, readable off the edges with no registration
    at all, and a band of flat correlated structure sitting inside the very data
    the run registers on.
    """
    movie = np.asarray(
        _built("drift-correction/two-channels-one-structural").data["movie"]
    )
    worst = {"px": 0, "frame": -1, "channel": -1, "edge": -1}
    for t, frame in enumerate(movie):
        for c, plane in enumerate(frame):
            # All four edges: flip to bring each one to the top in turn.
            for edge, view in enumerate((plane, plane[::-1], plane.T, plane.T[::-1])):
                width = _flat_border_px(view)
                if width > worst["px"]:
                    worst = {"px": width, "frame": t, "channel": c, "edge": edge}
    assert worst["px"] <= MAX_FLAT_BORDER_PX, (
        f"drift-correction: {worst['px']} px of flat border at frame "
        f"{worst['frame']}, channel {worst['channel']}, edge {worst['edge']} — "
        "the field of view is showing pixels no acquisition produced"
    )


def test_the_fixture_provides_every_layer_the_case_loads(case, built):
    for layer in case.layers:
        assert layer.key in built.data, (
            f"{case.skill}: layer {layer.name!r} loads data[{layer.key!r}], "
            f"which the fixture does not build ({sorted(built.data)})"
        )


def test_the_fixture_says_where_it_came_from(built):
    """Free text and required. A synthetic seed needs no review; an annotation
    is someone's claim about their own data and is only as good as the review it
    got, and this is where that is recorded."""
    assert built.provenance.strip()
    assert built.about.strip()


def test_a_run_that_left_nothing_scores_nothing(case, built):
    """The anti-vacuous check on the verifier itself.

    A verifier that passes an empty attempt would make every arm look fine,
    including the ones where the agent gave up — and `Outcome.passed` refusing
    to call "nothing scored" a pass only helps if the verifier reports
    unavailable rather than inventing a zero.
    """
    outcome = case.score(built, Attempt(subject="left-nothing"))
    assert not outcome.passed
    assert outcome.metrics, f"{case.skill}: the verifier reported no metrics at all"
    assert all(not m.scored for m in outcome.metrics)
    assert all(m.unavailable for m in outcome.metrics), (
        f"{case.skill}: a metric went unscored without saying why"
    )


def test_every_metric_the_verifier_reports_has_a_tolerance(case, built):
    """Read off the verifier rather than declared twice: the report's columns
    come from the metrics, and a limit of zero would be a silent always-fail."""
    outcome = case.score(built, Attempt(subject="left-nothing"))
    for metric in outcome.metrics:
        assert metric.limit > 0, f"{case.skill}: {metric.name} has no usable limit"


# --- the persona -----------------------------------------------------------


def test_every_fact_reaches_the_prompt(case):
    """The facts are data *and* prose, and the two must not drift. A fact the
    respondent holds but was never told about cannot be asked for, so the
    fixture would be withholding something nobody can obtain."""
    prompt = case.persona.system_prompt()
    for key, value in case.persona.facts.items():
        assert value in prompt, f"{case.skill}: {key!r} never reaches the prompt"


def test_the_persona_is_told_not_to_volunteer(case):
    """The one instruction the whole tier depends on. Asserted on the rendered
    prompt rather than trusted to the template, because the template is exactly
    what a well-meaning edit would loosen."""
    from ._respondent import DONE

    prompt = case.persona.system_prompt()
    assert "never volunteer" in prompt.casefold()
    assert DONE in prompt, "no way to end the conversation"


def test_the_background_gives_nothing_away(case):
    """`background` is what the respondent may share freely, so a private fact
    that leaked into it is available without asking — the fixture would look
    like it tests interaction while handing the answer over."""
    background = case.persona.background.casefold()
    for key, value in case.persona.facts.items():
        assert value.casefold() not in background, (
            f"{case.skill}: {key!r} is in the freely-shared background"
        )


def test_the_persona_knows_the_sample_and_not_the_procedure(case):
    """The two halves of a usable respondent, declared per case.

    It has to be able to answer — a fixture that strips a fact nobody holds is
    unanswerable rather than hard — and it must not have absorbed the skill's
    own vocabulary, or it could answer a question the agent never properly
    asked and the numeric result would stop meaning what it appears to.
    """
    prompt = case.persona.system_prompt().casefold()
    for known in case.persona_must_know:
        assert known.casefold() in prompt, (
            f"{case.skill}: the respondent cannot answer about {known!r}"
        )
    for procedural in case.persona_must_not_know:
        assert procedural.casefold() not in prompt, (
            f"{case.skill}: the respondent knows {procedural!r}, "
            "which is the skill's job"
        )


def test_the_case_declares_what_its_persona_must_hold(case):
    """Both lists non-empty, because either one empty makes the check above
    vacuous — and a vacuous version of it is indistinguishable from a passing
    one from the outside, which is the failure mode this whole file is about."""
    assert case.persona_must_know, f"{case.skill}: nothing declared as askable"
    assert case.persona_must_not_know, f"{case.skill}: no procedural terms fenced off"
