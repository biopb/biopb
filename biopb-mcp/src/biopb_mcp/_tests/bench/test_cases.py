"""Everything asserted about a case's *data*, before anyone pays for a run.

Hermetic and instant, and they run with the ordinary suite. A case is a task
prompt, a persona, a fixture and a verifier, and each of those has a way of
being quietly wrong that no amount of running would reveal:

- a persona that volunteers what it was not asked rescues a bad agent, and a
  green run looks identical;
- a fixture whose truth is visible in its data scores runs on a question they
  could read the answer to;
- a verifier that passes an empty attempt makes every run look fine;
- a task that asks for one name while the harness scrapes another scores a run
  on something it was never told to bind.

Every check runs over every case in `cases/`, so a case is checked by arriving
rather than by someone remembering to write a test for it. That is the property
this layer needs to survive a catalogue of thirty, and it is why there is no
longer a tuple of cases that are checked here and run nowhere: a case worth
checking is a case worth running.

**The run options do not reach this file.** `--bench-fixtures` narrows what an
invocation pays for; it does not narrow what has to be true. A case excluded
from tonight's run is checked tonight anyway, because these cost nothing and
the alternative is a filter that quietly turns off the tests as well.

What is *specific to one case* — a verifier's own unit tests, a fixture with a
property only it can have — is `test_verifiers.py`. This file only knows things
that are true of every case.
"""

from __future__ import annotations

import pytest

from ..agentbench._fixture import Attempt, Fixture
from ._case import LAYER_KINDS, PRESENTATIONS, TENSOR_HANDLE
from .cases import CASES

#: Everything with data to check, which is every case the layer runs. There used
#: to be tuples splitting these by whether they named a served skill; a case
#: names none now, so there is one population and every check runs on all of it.
ALL_CASES = CASES

#: Cases whose prompt is **self-sufficient**: they name no fact the run has to
#: elicit, so asking neither rescues nor penalises and the persona is there for
#: realism alone.
#:
#: `persona_must_know` is the declaration. It used to be tempting to read this
#: off a `skill` field instead, and the two coincided only while every case
#: without a skill was written against real data — the banked skills' cases had
#: no skill *and* withheld a fact the verifier depended on, so `skill` would
#: have applied the wrong half of this file to four of them. That field is gone
#: now and this is the only declaration there ever should have been.
SELF_SUFFICIENT = tuple(c for c in ALL_CASES if not c.persona_must_know)

#: The complement: every case that strips a fact out of its pixels and puts it
#: in the persona. What used to be called a "skill case" — but the property that
#: matters was never that a skill was served for it, only that the run has
#: something to elicit.
WITHHOLDING = tuple(c for c in ALL_CASES if c.persona_must_know)


def _ids(case):
    return case.label


#: Built once each: a fixture is megabytes of numpy and several tests share
#: them. Keyed on the full label — a subject covered two ways is two cases, and
#: keying on the namespace alone would hand the second the first's data.
_FIXTURES: dict[str, Fixture] = {}


@pytest.fixture(params=ALL_CASES, ids=_ids)
def case(request):
    return request.param


@pytest.fixture(params=SELF_SUFFICIENT or [None], ids=_ids)
def self_sufficient_case(request):
    if request.param is None:
        pytest.skip("every case in this tree withholds something")
    return request.param


@pytest.fixture(params=WITHHOLDING or [None], ids=_ids)
def withholding_case(request):
    if request.param is None:
        pytest.skip("no case in this tree withholds anything")
    return request.param


def built_fixture(case) -> Fixture:
    if case.label not in _FIXTURES:
        _FIXTURES[case.label] = case.build_fixture()
    return _FIXTURES[case.label]


@pytest.fixture
def built(case) -> Fixture:
    """This case's own fixture — skipped, never substituted, when the machine
    cannot produce it."""
    usable, why = case.available()
    if not usable:
        pytest.skip(f"{case.label}: {why}")
    return built_fixture(case)


def test_there_is_at_least_one_case():
    """The guard against this file going vacuously green, the same shape as
    `test_the_extractor_finds_pkg_tokens` in the contract layer."""
    assert CASES


def test_no_two_cases_share_an_identity():
    """`(namespace, case_id)` names a run's artifacts, its cached fixture and
    its report. Two cases colliding on it would overwrite one report with the
    other's and hand the second run the first's data — silently, since nothing
    downstream can tell the two apart."""
    seen = [c.label for c in ALL_CASES]
    duplicated = sorted({label for label in seen if seen.count(label) > 1})
    assert not duplicated, f"these cases collide on (namespace, case_id): {duplicated}"


def test_every_case_names_itself():
    for case in ALL_CASES:
        assert case.case_id.strip(), f"{case.namespace}: a case with no case_id"


def test_a_fixture_tree_does_not_change_what_a_procedural_case_runs(
    tmp_path, monkeypatch
):
    """The regression this whole design exists to prevent.

    `$BIOPB_FIXTURES` used to be a policy switch: whatever sat under it
    replaced a case's own fixture, silently and per machine. **Substituting the
    data makes it a different experiment with the same name** — the truth
    changes, the achievable accuracy changes, and the conclusion can invert,
    which was measured rather than supposed (`docs/fixtures.md`). It is a
    root path now, and a procedural case must not so much as look at it.
    """
    case = next(c for c in CASES if c.fixture.kind == "synthetic")
    decoy = tmp_path / case.namespace / case.case_id
    decoy.mkdir(parents=True)
    (decoy / "case.json").write_text('{"data": {}, "truth": {}}', encoding="utf-8")
    monkeypatch.setenv("BIOPB_FIXTURES", str(tmp_path))

    built = case.build_fixture()
    assert built.kind == "synthetic"
    assert built.data, "the case built nothing, so the decoy was consulted"


# --- the case is runnable --------------------------------------------------


def test_every_case_is_complete_enough_to_run(case):
    """The fields a run cannot proceed without, checked without running one."""
    assert case.task.strip(), f"{case.label}: no task prompt"
    assert case.layers, f"{case.label}: no fixture layer to load"
    assert case.collect, f"{case.label}: nothing would be collected"
    assert callable(case.score)
    assert callable(getattr(case.fixture, "build", None)), (
        f"{case.label}: `fixture` is not a FixtureSpec"
    )


def test_the_task_asks_for_exactly_what_is_collected(case):
    """The scrape names are a **harness convention**, not a claim the subject
    makes — so the prompt has to state them, or the run is scored on names the
    agent was never told to bind."""
    for expression in case.collect.values():
        assert expression in case.task, (
            f"{case.label}: the task never mentions {expression!r}, "
            "which is where its result is read from"
        )


def test_the_task_names_every_layer_it_is_given(case):
    """A layer the prompt never mentions is one the agent has to discover by
    listing the viewer, which is a different task from the one written down."""
    for layer in case.layers:
        assert layer.name in case.task, (
            f"{case.label}: puts a layer `{layer.name}` on the viewer that the "
            "task text never mentions"
        )


def test_every_layer_kind_is_one_the_harness_can_add(case):
    for layer in case.layers:
        assert layer.kind in LAYER_KINDS, (
            f"{case.label}: layer {layer.name!r} wants a {layer.kind!r} layer, "
            f"and the harness can add {sorted(LAYER_KINDS)}"
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
    not find it. What the agent gets instead is the layer the harness already
    added and the ids under :data:`TENSOR_HANDLE`, and the prompt has to point
    at one of them: an agent nobody told would be scored on failing to guess at
    a source it could not list."""
    lazy = [layer for layer in case.layers if layer.lazy]
    if not lazy:
        return
    assert TENSOR_HANDLE in case.task or all(
        layer.name in case.task for layer in lazy
    ), (
        f"{case.label}: presents on the data plane, but its task names neither "
        f"`{TENSOR_HANDLE}` nor the layers it creates"
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


def test_the_fixture_provides_every_layer_the_case_loads(case, built):
    for layer in case.layers:
        assert layer.key in built.data, (
            f"{case.label}: layer {layer.name!r} loads data[{layer.key!r}], "
            f"which the fixture does not build ({sorted(built.data)})"
        )


def test_every_array_the_fixture_builds_reaches_the_agent(case, built):
    """The converse of the check above, and the one that catches a `Layer` whose
    key is a typo for another layer's.

    `Layer` is `(name, key)`, so `Layer("beads", "image")` is well-formed, passes
    the check above, and presents `data["image"]` twice under two names while the
    bead stack is never shown. What the agent sees is two layers, both plausible,
    and a case that silently measures something else — `deconvolve-widefield`
    shipped that way and deconvolved its image with a PSF measured from itself.
    An array a fixture pays to build and nobody is shown is the symptom, so that
    is what is asserted: every key presented, each exactly once.
    """
    keys = [layer.key for layer in case.layers]
    duplicated = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicated, (
        f"{case.label}: {duplicated} is presented as more than one layer, so "
        f"some other array of this fixture is presented as none"
    )
    unseen = sorted(set(built.data) - set(keys))
    assert not unseen, (
        f"{case.label}: the fixture builds {unseen}, which no layer presents — "
        f"the agent is never shown it"
    )


def test_the_fixture_says_where_it_came_from(built):
    """Free text and required. A synthetic seed needs no review; an annotation
    is someone's claim about their own data and is only as good as the review it
    got, and this is where that is recorded."""
    assert built.provenance.strip()
    assert built.about.strip()


def test_a_curated_fixture_names_whose_data_it_is(built):
    """Real data comes from someone, and saying so is not optional. A synthetic
    seed owes nobody a citation, which is why this is not asked of one."""
    if built.kind == "curated":
        assert built.citation.strip(), f"{built.label}: curated, and uncited"


def test_a_run_that_left_nothing_scores_nothing(case, built):
    """The anti-vacuous check on the verifier itself.

    A verifier that passes an empty attempt would make every run look fine,
    including the ones where the agent gave up — and `Outcome.passed` refusing
    to call "nothing scored" a pass only helps if the verifier reports
    unavailable rather than inventing a zero.
    """
    outcome = case.score(built, Attempt(subject="left-nothing"))
    assert not outcome.passed
    assert outcome.metrics, f"{case.label}: the verifier reported no metrics at all"
    assert all(not m.scored for m in outcome.metrics)
    assert all(m.unavailable for m in outcome.metrics), (
        f"{case.label}: a metric went unscored without saying why"
    )


def test_every_metric_the_verifier_reports_has_a_tolerance(case, built):
    """Read off the verifier rather than declared twice: the report's columns
    come from the metrics, and a limit of zero would be a silent always-fail."""
    outcome = case.score(built, Attempt(subject="left-nothing"))
    for metric in outcome.metrics:
        assert metric.limit > 0, f"{case.label}: {metric.name} has no usable limit"


# --- the persona -----------------------------------------------------------


def test_every_fact_reaches_the_prompt(case):
    """The facts are data *and* prose, and the two must not drift. A fact the
    respondent holds but was never told about cannot be asked for, so the
    fixture would be withholding something nobody can obtain."""
    prompt = case.persona.system_prompt()
    for key, value in case.persona.facts.items():
        assert value in prompt, f"{case.label}: {key!r} never reaches the prompt"


def test_the_persona_is_told_not_to_volunteer(case):
    """The one instruction the whole tier depends on. Asserted on the rendered
    prompt rather than trusted to the template, because the template is exactly
    what a well-meaning edit would loosen."""
    from ..agentbench._respondent import DONE

    prompt = case.persona.system_prompt()
    assert "never volunteer" in prompt.casefold()
    assert DONE in prompt, "no way to end the conversation"


def test_the_persona_is_told_to_have_no_view_on_the_analysis(case):
    """The sibling of the rule above, and the one it does not cover.

    "Answer only what you were asked" and "never volunteer" both permit a method
    answer when the question *is* about method — which is how a respondent that
    correctly said it had no background value went on to recommend a named
    technique, and the agent, obeying its user, landed on the exact route its
    case is built to penalise. Refusing to advise is a separate instruction and
    is asserted separately.
    """
    prompt = case.persona.system_prompt().casefold()
    assert "the analysis is not yours" in prompt
    assert "theirs to decide" in prompt


def test_the_background_gives_nothing_away(case):
    """`background` is what the respondent may share freely, so a private fact
    that leaked into it is available without asking — the fixture would look
    like it tests interaction while handing the answer over."""
    background = case.persona.background.casefold()
    for key, value in case.persona.facts.items():
        assert value.casefold() not in background, (
            f"{case.label}: {key!r} is in the freely-shared background"
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
            f"{case.label}: the respondent cannot answer about {known!r}"
        )
    for procedural in case.persona_must_not_know:
        assert procedural.casefold() not in prompt, (
            f"{case.label}: the respondent knows {procedural!r}, "
            "which is the skill's job"
        )


def test_the_briefing_carries_the_facts_and_keeps_the_same_fence(case):
    """`--bench-responder=briefed` hands the persona's facts to the agent, and
    both halves of that have to hold per case.

    It must carry **every** fact, or a briefed session is a different fixture
    from the spoken one and the pair measures nothing it claims to. And it must
    fence off the same vocabulary `persona_must_not_know` does: the switch
    exists to remove the *asking*, and a brief that also handed over the skill's
    procedure would remove the subject as well.
    """
    briefing = case.persona.briefing()
    for key, value in case.persona.facts.items():
        assert value in briefing, f"{case.label}: {key!r} never reaches the brief"
    folded = briefing.casefold()
    for procedural in case.persona_must_not_know:
        assert procedural.casefold() not in folded, (
            f"{case.label}: the brief hands over {procedural!r}, "
            "which is the run's job to work out"
        )


def test_a_withholding_case_fences_off_its_own_procedure(withholding_case):
    """A case that withholds a fact must also say which vocabulary the persona
    may not use, because either list empty makes the check above vacuous — and a
    vacuous version of it is indistinguishable from a passing one from the
    outside, which is the failure mode this whole file is about."""
    assert withholding_case.persona_must_not_know, (
        f"{withholding_case.label}: withholds a fact but fences off no "
        "procedural terms, so the persona may hand over the method"
    )


def test_a_self_sufficient_case_holds_no_deliverable_in_its_persona(
    self_sufficient_case,
):
    """A persona that can be asked for the answer measures nothing.

    This is the check for the shape that declares nothing to elicit: the prompt
    is complete, the persona is there for realism, and so a fact naming one of
    the scraped deliverables is a fact about the answer. A case that *does*
    withhold something is checked by `persona_must_not_know` instead, which is
    written in that case's own vocabulary and does not have to guess.

    The names are matched as substrings, which only carries because a
    deliverable here is a kernel identifier. `count-foci-per-cell` collects
    `counts`, and its persona says "what counts as a focus" — a word this check
    cannot tell from the variable, and the reason it is not applied to a case
    whose persona is supposed to be answering questions.
    """
    case = self_sufficient_case
    facts = " ".join(case.persona.facts).casefold()
    facts += " " + " ".join(case.persona.facts.values()).casefold()
    for name in case.collect.values():
        assert name.casefold() not in facts, (
            f"{case.label}: the persona knows `{name}`, which is what the "
            "run is supposed to work out"
        )
