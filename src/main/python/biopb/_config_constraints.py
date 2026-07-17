"""Shared config-value constraint primitives, in the core ``biopb`` package.

Both config-bearing packages validate leaf values against ranges/enums, but
neither can import the other's config module: ``biopb-tensor-server`` is not on
PyPI (so ``biopb-mcp`` cannot depend on it at runtime), and the two config
*shapes* differ (the server validates dataclass instances in ``__post_init__``;
biopb-mcp validates a merged dict by dotted path). What they *can* share is the
tiny, format-agnostic core: the ``Range`` / ``Enum`` primitives and the handful
of constraint rows that describe the **same knobs** in both places.

The pyramid rows are the load-bearing case. ``pyramid.{threshold,
downscale_factor, pixel_budget_cubic_root}`` exist in both packages (the tensor
server's ``PyramidConfig`` copied biopb-mcp's historical defaults); an
out-of-range value there silently breaks pyramid construction on either side
(``downscale_factor=1`` -> a single full-res level; ``pixel_budget_cubic_root<=0``
-> an infinite loop). Declaring the bounds here once means the two packages
cannot drift (biopb/biopb#34, #182).

Deliberately stdlib-only, like the sibling :mod:`biopb._locations`, so
importing it stays cheap and pulls in none of the heavy adapter/discovery
machinery.
"""

from __future__ import annotations


class Range:
    """A numeric bound. ``min``/``max`` are inclusive; ``exclusive_min``/
    ``exclusive_max`` are strict. Any subset may be omitted.

    The exclusive forms express the common "must be strictly positive" leaf
    (``exclusive_min=0``) -- e.g. a timeout, where ``0`` is nonsensical -- that
    an inclusive floor cannot capture without an arbitrary epsilon.
    """

    def __init__(self, *, min=None, max=None, exclusive_min=None, exclusive_max=None):
        self.min = min
        self.max = max
        self.exclusive_min = exclusive_min
        self.exclusive_max = exclusive_max

    def ok(self, value) -> bool:
        # bool is an int subclass; a bool where a number is expected is wrong.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return (
            (self.min is None or value >= self.min)
            and (self.max is None or value <= self.max)
            and (self.exclusive_min is None or value > self.exclusive_min)
            and (self.exclusive_max is None or value < self.exclusive_max)
        )

    def describe(self) -> str:
        parts = []
        if self.min is not None:
            parts.append(f">= {self.min}")
        if self.exclusive_min is not None:
            parts.append(f"> {self.exclusive_min}")
        if self.max is not None:
            parts.append(f"<= {self.max}")
        if self.exclusive_max is not None:
            parts.append(f"< {self.exclusive_max}")
        if not parts:
            return "a number"
        return "a number " + " and ".join(parts)

    def to_json_schema(self) -> dict:
        """Bounds as JSON Schema keywords (for a schema emitter)."""
        out: dict = {}
        if self.min is not None:
            out["minimum"] = self.min
        if self.max is not None:
            out["maximum"] = self.max
        if self.exclusive_min is not None:
            out["exclusiveMinimum"] = self.exclusive_min
        if self.exclusive_max is not None:
            out["exclusiveMaximum"] = self.exclusive_max
        return out


class Enum:
    """Membership in a fixed set, optionally case-insensitive for strings."""

    def __init__(self, allowed, *, case_insensitive=False):
        self.case_insensitive = case_insensitive
        self._display = set(allowed)
        # When case-insensitive, fold the allowed set too so the comparison is
        # symmetric (else a lowercased value never matches an upper-case member).
        self.allowed = {
            a.lower() if (case_insensitive and isinstance(a, str)) else a
            for a in allowed
        }

    def ok(self, value) -> bool:
        if self.case_insensitive and isinstance(value, str):
            value = value.strip().lower()
        try:
            return value in self.allowed
        except TypeError:
            # An unhashable value (list / dict) can never be an enum member. Return
            # False rather than letting the membership test raise, so ok() is total
            # like Range.ok -- a bad-typed config leaf is rejected, not a crash
            # (e.g. the control's PUT validator would otherwise 500 on it).
            return False

    def describe(self) -> str:
        return "one of: " + ", ".join(sorted(map(str, self._display)))

    def to_json_schema(self) -> dict:
        """The allowed set as a JSON Schema ``enum`` -- but only for
        case-sensitive enums. A case-insensitive set accepts any casing the
        consumer folds, so a hard ``enum`` of the canonical members would reject
        values that are actually honored; there we stay lenient and surface the
        set via :meth:`describe` in the property description instead."""
        if self.case_insensitive:
            return {}
        return {"enum": sorted(self._display, key=str)}


# The pyramid-knob bounds shared by biopb-tensor-server (PyramidConfig) and
# biopb-mcp (the `pyramid` config section). Keyed by the leaf field name, which
# is identical in both. `reduction_method` is intentionally absent: on-the-fly
# reduction is a tensor-server compute concern, not a biopb-mcp knob, so its
# enum stays local to the server.
#
#   downscale_factor >= 2         : must actually shrink; ==1 yields no pyramid.
#   threshold >= 1                : max x/y extent of the coarsest level.
#   pixel_budget_cubic_root >= 1  : cubed voxel budget; <=1 loops/OOMs.
PYRAMID_CONSTRAINTS = {
    "threshold": Range(min=1),
    "downscale_factor": Range(min=2),
    "pixel_budget_cubic_root": Range(min=1),
}
