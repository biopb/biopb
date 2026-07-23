"""The shared config-validation scheme (``biopb._config_validate``).

One checker, one policy, three consumers (biopb-tensor-server's load path,
biopb-mcp's load path, the control's admin endpoints). These tests pin the parts
those consumers rely on: it reads dicts and dataclasses alike, it reports *every*
problem rather than stopping at the first, cross-field rules can implicate more
than one field, and the clamping policy names what it substituted.
See biopb/biopb#34.
"""

import logging
from dataclasses import dataclass

import pytest
from biopb._config_constraints import Enum, Range
from biopb._config_validate import (
    MISSING,
    Problem,
    check_sections,
    describe_violation,
    warn_and_clamp,
)


@dataclass
class _Sect:
    port: int = 8815
    mode: str = "fast"


_CONSTRAINTS = {
    "_Sect": {"port": Range(min=1, max=65535), "mode": Enum({"fast", "slow"})}
}


def _dict_sections(**values):
    return [("sect", values)]


# --- the two section shapes -------------------------------------------------


def test_checks_a_mapping_section():
    problems = check_sections(
        _dict_sections(port=0), _CONSTRAINTS, class_names={"sect": "_Sect"}
    )
    assert [p.path for p in problems] == [("sect", "port")]


def test_checks_a_dataclass_section_without_class_names():
    # A dataclass names its own class, which is how the tensor server passes its
    # constructed config straight in.
    problems = check_sections([("sect", _Sect(port=0))], _CONSTRAINTS)
    assert [p.path for p in problems] == [("sect", "port")]


def test_absent_key_is_not_a_problem():
    # An omitted key is not a wrong value; every caller resolves absence to a
    # default before this runs.
    assert (
        check_sections(_dict_sections(), _CONSTRAINTS, class_names={"sect": "_Sect"})
        == []
    )


def test_unconstrained_section_is_skipped():
    assert check_sections([("other", {"port": 0})], _CONSTRAINTS) == []


# --- completeness -----------------------------------------------------------


def test_reports_every_problem_not_just_the_first():
    # The admin form needs them all at once; fixing one and resubmitting to
    # discover the next is the failure mode this rules out.
    problems = check_sections(
        _dict_sections(port=0, mode="sideways"),
        _CONSTRAINTS,
        class_names={"sect": "_Sect"},
    )
    assert {p.path for p in problems} == {("sect", "port"), ("sect", "mode")}


def test_message_names_value_and_accepted_rule():
    (problem,) = check_sections(
        _dict_sections(mode="sideways"), _CONSTRAINTS, class_names={"sect": "_Sect"}
    )
    assert "mode='sideways'" in problem.message
    assert "fast" in problem.message and "slow" in problem.message


def test_problem_as_dict_is_the_wire_shape():
    assert Problem(("a", "b"), "boom").as_dict() == {
        "path": ["a", "b"],
        "message": "boom",
    }


def test_describe_violation_is_the_single_phrasing():
    assert (
        describe_violation("port", 0, Range(min=1)) == "port=0 (expected a number >= 1)"
    )


# --- cross-field rules ------------------------------------------------------


def _lo_le_hi(get):
    lo, hi = get("sect", "lo"), get("sect", "hi")
    if lo is MISSING or hi is MISSING or lo <= hi:
        return []
    return [Problem(("sect", "lo"), "lo > hi"), Problem(("sect", "hi"), "lo > hi")]


def test_cross_field_rule_can_implicate_several_fields():
    # Both ends must be reportable: clamping only one can leave the pair
    # inverted the other way.
    problems = check_sections(
        _dict_sections(lo=9, hi=2),
        _CONSTRAINTS,
        class_names={"sect": "_Sect"},
        cross_field=[_lo_le_hi],
    )
    assert [p.path for p in problems] == [("sect", "lo"), ("sect", "hi")]


def test_cross_field_rule_sees_absent_keys_as_missing():
    problems = check_sections(
        _dict_sections(lo=9),
        _CONSTRAINTS,
        class_names={"sect": "_Sect"},
        cross_field=[_lo_le_hi],
    )
    assert problems == []


# --- the clamping policy ----------------------------------------------------


def test_warn_and_clamp_applies_the_default_and_says_so(caplog):
    values = {"port": 0}
    with caplog.at_level(logging.WARNING):
        warn_and_clamp(
            [Problem(("sect", "port"), describe_violation("port", 0, Range(min=1)))],
            lambda path: 8815,
            lambda path, value: values.__setitem__(path[1], value),
            logging.getLogger(__name__),
        )
    assert values["port"] == 8815
    message = caplog.messages[0]
    assert "sect.port" in message  # the dotted path, findable in the file
    assert "8815" in message  # what actually ran


def test_warn_and_clamp_warns_without_applying_when_there_is_no_default(caplog):
    applied = []
    with caplog.at_level(logging.WARNING):
        warn_and_clamp(
            [Problem(("sect", "port"), "port=0 (expected ...)")],
            lambda path: MISSING,
            lambda path, value: applied.append(path),
            logging.getLogger(__name__),
        )
    assert applied == []
    assert "sect.port" in caplog.messages[0]


@pytest.mark.parametrize("bad", [True, "8815", None, [1]])
def test_wrong_typed_values_are_problems_not_crashes(bad):
    # Untrusted input reaches this from an HTTP body; ok() is total, so a
    # wrong-typed leaf must report rather than raise (the endpoint would 500).
    problems = check_sections(
        _dict_sections(port=bad), _CONSTRAINTS, class_names={"sect": "_Sect"}
    )
    assert [p.path for p in problems] == [("sect", "port")]
