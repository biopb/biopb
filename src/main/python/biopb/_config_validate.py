"""The one config-validation scheme every biopb package uses, in core ``biopb``.

Both config-bearing packages describe their config as dataclasses plus a
class-keyed ``_CONSTRAINTS`` table (see :mod:`biopb._config_constraints` for the
``Range``/``Enum`` primitives and :mod:`biopb._config_schema` for the JSON-Schema
projection). This module is the third piece: **checking values against that
table**, once, for everyone -- biopb-tensor-server's load path, biopb-mcp's load
path, and the control plane's two admin config endpoints, which previously each
walked the table themselves.

The scheme, in one line: **check at the read step; warn and fall back to the
default; reject only where a human is explicitly submitting or asking.**

*Where* it runs: at the read step (file -> config), not in ``__post_init__``.
Every value that can be wrong comes from a file or an HTTP body, and both funnel
through the read step; direct dataclass construction in code is a programming
error caught by the type checker and the tests, not a runtime input to police. One
check point is also the only way the two packages can share this at all --
biopb-mcp never constructs its dataclasses from user data (its runtime form is a
merged dict).

*What* it does on a violation: log a warning naming the key, the offending value,
the accepted range, and the default it is using -- then **use the default**. It
does not raise. The bad value never reaches the runtime either way (that is the
whole requirement -- ``downscale_factor=0`` must not reach ``ceil_div``), and the
difference is what happens to the process. Both consumers are supervised,
long-lived, and user-facing: the tensor server is a control-plane child that gets
restarted on crash with capped backoff (a raise at load = a permanent restart
loop reported as "the data plane keeps dying", with the actual cause buried in a
log), and biopb-mcp is an interactive session an agent spawns over stdio (a raise
= a dead MCP client and no viewer). Refusing to run converts one wrong number
into total unavailability; clamping converts it into one wrong number and a
warning.

*Where it is still strict*: the surfaces where a human is in the loop and can act
on a rejection -- ``PUT /api/config`` / ``PUT /api/mcp_config`` (422 listing every
problem, per field, so the form can highlight them) and
``biopb-tensor-server validate`` (exit 1). Those call :func:`check_sections`
directly and skip the clamping.

Deliberately stdlib-only, like its siblings: it duck-types the constraint objects
(``ok`` / ``describe``) and imports none of them.
"""

from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# Sentinel for "this key is not present", distinct from a present ``None``.
MISSING = object()


@dataclasses.dataclass(frozen=True)
class Problem:
    """One constraint violation, located by its ``(section, key)`` path.

    The path is what lets a caller merge these with JSON-Schema errors (dedupe by
    path) and what lets a form highlight the offending field, so it travels with
    the message rather than being baked into it.
    """

    path: Tuple[str, ...]
    message: str

    def as_dict(self) -> Dict[str, Any]:
        """Wire form: ``{"path": [section, key], "message": str}``."""
        return {"path": list(self.path), "message": self.message}


# A cross-field rule reads the whole config through a ``get(section, key)``
# accessor (returning MISSING for an absent key) and returns a Problem per field
# it implicates -- empty when satisfied. Per-field constraints cannot express
# "min <= max"; declaring these as data beside the _CONSTRAINTS table means every
# surface that validates a config gets them, instead of the one that remembered
# to open-code the comparison.
#
# *Every* implicated field, not just the first, because the clamping policy
# resets exactly the paths it is handed: fixing only the `min` of an inverted
# pair can leave it inverted the other way (a default `min` above the user's
# `max`), while resetting both lands on two defaults that are consistent by
# construction. The strict surfaces benefit equally -- the form highlights both
# ends of the range the user has to reconcile.
CrossFieldRule = Callable[[Callable[[str, str], Any]], Sequence[Problem]]


def describe_violation(key: str, value: Any, constraint: Any) -> str:
    """The one phrasing of a violation: ``key=value (expected <rule>)``."""
    return f"{key}={value!r} (expected {constraint.describe()})"


def _field(values: Any, key: str) -> Any:
    """Read *key* off a section, whether it is a mapping or a dataclass instance.

    The two packages hold a section differently -- biopb-mcp a plain dict, the
    tensor server a constructed dataclass -- and that is the only difference
    between their checks, so it is absorbed here rather than duplicated as two
    walks.
    """
    if isinstance(values, Mapping):
        return values.get(key, MISSING)
    return getattr(values, key, MISSING)


def check_sections(
    sections: Iterable[Tuple[str, Any]],
    constraints: Mapping[str, Mapping[str, Any]],
    *,
    class_names: Optional[Mapping[str, str]] = None,
    cross_field: Sequence[CrossFieldRule] = (),
) -> List[Problem]:
    """Every constraint violation in *sections*, in declaration order.

    Args:
        sections: ``(section_name, values)`` pairs -- ``values`` is a dict or a
            dataclass instance (see :func:`_field`).
        constraints: the ``_CONSTRAINTS`` table, keyed by dataclass name.
        class_names: ``section_name -> dataclass name``, needed only when
            *values* is a mapping (a dataclass instance names its own class).
        cross_field: rules spanning two fields, applied after the per-field pass.

    An absent key is skipped -- it is not a wrong value, and every caller has
    already resolved absence to a default (merged dict / dataclass default).
    """
    problems: List[Problem] = []
    by_section = dict(sections)

    for section, values in by_section.items():
        if isinstance(values, Mapping):
            class_name = (class_names or {}).get(section)
        else:
            class_name = type(values).__name__
        for key, constraint in (constraints.get(class_name) or {}).items():
            value = _field(values, key)
            if value is MISSING or constraint.ok(value):
                continue
            problems.append(
                Problem((section, key), describe_violation(key, value, constraint))
            )

    def get(section: str, key: str) -> Any:
        values = by_section.get(section, MISSING)
        return MISSING if values is MISSING else _field(values, key)

    for rule in cross_field:
        problems.extend(rule(get))

    return problems


def warn_and_clamp(
    problems: Iterable[Problem],
    default_for: Callable[[Tuple[str, ...]], Any],
    apply: Callable[[Tuple[str, ...], Any], None],
    logger: Any,
) -> None:
    """Log each problem and reset its field to the default -- the load-path policy.

    *default_for* returns the default for a path (or :data:`MISSING` if there is
    none, e.g. a cross-field rule whose path is only one of the two culprits) and
    *apply* writes it back -- a setattr on a dataclass, an item assignment in a
    dict. The message names the default in use, because "ignored your value" is
    only actionable if the reader can see what ran instead.
    """
    for problem in problems:
        # The message names the leaf ("port=70000 (expected ...)"); prefixing the
        # section makes the log line carry the full dotted path a reader needs to
        # find the key in the file.
        located = f"{problem.path[0]}.{problem.message}"
        default = default_for(problem.path)
        if default is MISSING:
            logger.warning("Invalid config value %s. Ignoring it.", located)
            continue
        logger.warning(
            "Invalid config value %s; using the default %r instead.", located, default
        )
        apply(problem.path, default)
