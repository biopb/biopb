"""Unit tests for the shared config-constraint primitives.

These primitives (:class:`biopb._config_constraints.Range` / ``Enum``) and the
shared ``PYRAMID_CONSTRAINTS`` rows are consumed by both biopb-tensor-server and
biopb-mcp, so their behavior is pinned here in the core package (biopb#34, #182).
"""

from biopb._config_constraints import PYRAMID_CONSTRAINTS, Enum, Range


class TestRange:
    def test_inclusive_bounds(self):
        r = Range(min=1, max=10)
        assert r.ok(1) and r.ok(10) and r.ok(5)
        assert not r.ok(0) and not r.ok(11)

    def test_open_ended(self):
        assert Range(min=2).ok(1_000_000)
        assert not Range(min=2).ok(1)
        assert Range(max=5).ok(-3)
        assert not Range(max=5).ok(6)

    def test_exclusive_min_rejects_the_bound(self):
        r = Range(exclusive_min=0)
        assert not r.ok(0)
        assert not r.ok(-1)
        assert r.ok(0.001) and r.ok(60.0)

    def test_exclusive_max_rejects_the_bound(self):
        r = Range(exclusive_max=1.0)
        assert r.ok(0.99)
        assert not r.ok(1.0) and not r.ok(2.0)

    def test_bool_is_not_a_number(self):
        # bool is an int subclass; True must not slip through a numeric range.
        assert not Range(min=0, max=10).ok(True)
        assert not Range(min=0, max=10).ok(False)

    def test_non_numeric_rejected(self):
        r = Range(min=1)
        assert not r.ok("4")
        assert not r.ok(None)
        assert not r.ok([1])

    def test_describe(self):
        assert Range(min=2).describe() == "a number >= 2"
        assert Range(exclusive_min=0).describe() == "a number > 0"
        assert Range(min=1, max=10).describe() == "a number >= 1 and <= 10"

    def test_to_json_schema(self):
        assert Range(min=1, max=10).to_json_schema() == {
            "minimum": 1,
            "maximum": 10,
        }
        assert Range(exclusive_min=0).to_json_schema() == {"exclusiveMinimum": 0}


class TestEnum:
    def test_membership(self):
        e = Enum({"a", "b"})
        assert e.ok("a")
        assert not e.ok("c")

    def test_case_insensitive(self):
        e = Enum({"INFO", "DEBUG"}, case_insensitive=True)
        assert e.ok("info") and e.ok("  Debug ")
        assert not e.ok("trace")

    def test_case_sensitive_by_default(self):
        assert not Enum({"INFO"}).ok("info")

    def test_ok_is_total_on_bad_types(self):
        # ok() must never raise -- a bad-typed config leaf is rejected, not a
        # crash (the control's PUT validator calls ok() on arbitrary JSON). This
        # includes unhashable values (list/dict), which a naive `in set` raises on.
        e = Enum({"http", "stdio"})
        for bad in (None, 5, True, 1.5, [1, 2], {"a": 1}, (1, 2)):
            assert e.ok(bad) is False
        assert Enum({"INFO"}, case_insensitive=True).ok(["x"]) is False

    def test_json_schema_only_for_case_sensitive(self):
        assert Enum({"http", "stdio"}).to_json_schema() == {"enum": ["http", "stdio"]}
        # A case-insensitive enum emits no hard `enum` (it would reject casings
        # the consumer actually folds and honors).
        assert Enum({"INFO"}, case_insensitive=True).to_json_schema() == {}


class TestPyramidConstraints:
    def test_the_three_shared_rows(self):
        assert set(PYRAMID_CONSTRAINTS) == {
            "threshold",
            "downscale_factor",
            "pixel_budget_cubic_root",
        }

    def test_downscale_factor_must_shrink(self):
        # ==1 produces no pyramid; the shared rule forbids it in both packages.
        assert not PYRAMID_CONSTRAINTS["downscale_factor"].ok(1)
        assert PYRAMID_CONSTRAINTS["downscale_factor"].ok(2)

    def test_budget_and_threshold_positive(self):
        assert not PYRAMID_CONSTRAINTS["pixel_budget_cubic_root"].ok(0)
        assert not PYRAMID_CONSTRAINTS["threshold"].ok(0)
        assert PYRAMID_CONSTRAINTS["pixel_budget_cubic_root"].ok(1)
        assert PYRAMID_CONSTRAINTS["threshold"].ok(1)
