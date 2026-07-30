"""Tests for mcp/_requires.py — resolving a skill's ``requires:`` in the kernel.

The one property worth defending above all others: **never report a requirement as
met when it isn't.** Everything else is reporting quality. So the tests pin what
each verdict is derived from (the loader's record, not a directory listing; the live
connection, not the per-job copy), and pin the three-way met/unmet/unknown split so
a token this version can't check reaches the agent as a question rather than a pass.
"""

import pytest

from biopb_mcp.mcp import _bootstrap, _requires


@pytest.fixture
def ns():
    """A namespace shaped like a healthy post-bootstrap kernel."""

    class Conn:
        client = object()
        last_message = ""

    return {
        "viewer": object(),
        "client": object(),
        "np": object(),
        "da": object(),
        "ops": {"segmentation": lambda: None, "restoration": lambda: None},
        "_conn": Conn(),
    }


def check(ns, *tokens):
    return _requires.check(list(tokens), ns)


class TestViewer:
    def test_a_real_viewer_is_met(self, ns):
        assert check(ns, "viewer")["met"] == ["viewer"]

    def test_headless_stand_in_is_unmet_with_the_fallback_advice(self, ns):
        # Detected by type, not truthiness: the stand-in is falsy on purpose, but
        # a real napari viewer's truthiness is not ours to depend on.
        ns["viewer"] = _bootstrap._HeadlessViewer()
        (gap,) = check(ns, "viewer")["unmet"]
        assert gap.startswith("viewer — ")
        assert "headless" in gap
        assert "numeric fallback" in gap

    def test_absent_viewer_is_unmet(self, ns):
        ns.pop("viewer")
        assert check(ns, "viewer")["unmet"]


class TestTensor:
    def test_connected_is_met(self, ns):
        assert check(ns, "tensor")["ok"]

    def test_the_live_connection_wins_over_the_per_job_copy(self, ns):
        # `client` is a copy the job runner refreshes; `_conn.client` is the truth,
        # so a check outside a job (or before the first refresh) is still right.
        ns["_conn"].client = None
        ns["client"] = object()
        assert not check(ns, "tensor")["ok"]

    def test_the_reason_carries_the_connection_message(self, ns):
        ns["_conn"].client = None
        ns["_conn"].last_message = "server unreachable at grpc://localhost:8815"
        (gap,) = check(ns, "tensor")["unmet"]
        assert "grpc://localhost:8815" in gap
        assert "biopb control status" in gap  # what the user can actually run

    def test_without_a_connection_object_the_namespace_copy_is_used(self, ns):
        ns.pop("_conn")
        ns["client"] = None
        assert not check(ns, "tensor")["ok"]


class TestDask:
    def test_met_when_dask_array_is_bound(self, ns):
        assert check(ns, "dask")["ok"]

    def test_in_process_scheduler_still_meets_it(self, ns):
        # `dask` in a skill means "this works on lazy arrays", which stays true
        # under the threads scheduler; whether a *distributed* cluster is attached
        # is a performance property, and server_status is where that is reported.
        ns["_dask_client"] = None
        ns["_dask_attach_done"] = True
        assert check(ns, "dask")["ok"]

    def test_absent_dask_is_unmet(self, ns):
        ns.pop("da")
        assert check(ns, "dask")["unmet"]


class TestOps:
    def test_configured_op_is_met(self, ns):
        assert check(ns, "ops:segmentation")["ok"]

    def test_missing_op_names_what_the_servers_do_offer(self, ns):
        # The alternative is usually right there, and the agent can't see `ops`
        # without another round trip.
        (gap,) = check(ns, "ops:tracking")["unmet"]
        assert "restoration" in gap and "segmentation" in gap

    def test_no_ops_at_all_names_the_config_key(self, ns):
        ns["ops"] = {}
        (gap,) = check(ns, "ops:segmentation")["unmet"]
        assert "services.process_image_servers" in gap


class TestPlugin:
    def test_a_loaded_plugin_is_met(self, ns):
        _requires.record_loaded_plugins(["segmentation_qc", "rolling_ball"])
        assert check(ns, "plugin:segmentation_qc")["ok"]

    def test_a_plugin_that_did_not_load_names_the_ones_that_did(self, ns):
        _requires.record_loaded_plugins(["rolling_ball"])
        (gap,) = check(ns, "plugin:segmentation_qc")["unmet"]
        assert "rolling_ball" in gap
        assert "biopb-mcp-seed-plugins" in gap
        assert "session log" in gap  # where a load failure is actually explained

    def test_no_plugins_loaded_points_at_the_seed_command(self, ns):
        _requires.record_loaded_plugins([])
        (gap,) = check(ns, "plugin:segmentation_qc")["unmet"]
        assert "biopb-mcp-seed-plugins" in gap

    def test_disabled_namespace_names_the_switch_not_the_seed_command(self, ns):
        # With plugins switched off, seeding a file would not help, so suggesting
        # it would send the user down the wrong path.
        _requires.record_loaded_plugins([], enabled=False)
        (gap,) = check(ns, "plugin:segmentation_qc")["unmet"]
        assert "services.namespace_enabled" in gap
        assert "biopb-mcp-seed-plugins" not in gap

    def test_the_seed_hint_asks_before_acting(self, ns):
        _requires.record_loaded_plugins([])
        (gap,) = check(ns, "plugin:x")["unmet"]
        assert "user's OK" in gap  # seeding + restarting the kernel need consent


class TestPackage:
    def test_an_installed_distribution_is_met(self, ns):
        assert check(ns, "pkg:biopb-mcp")["ok"]

    def test_an_absent_package_is_unmet_without_offering_to_install_it(self, ns):
        (gap,) = check(ns, "pkg:definitely-not-installed-xyz")["unmet"]
        assert "not installed in this kernel" in gap
        assert "degraded path" in gap
        assert "user's OK" in gap
        assert "pip install" not in gap  # never hand the agent the command

    def test_import_name_without_distribution_metadata_is_met(self, ns, monkeypatch):
        # A `pkg:` token may name the import (skimage) rather than the
        # distribution (scikit-image); absent metadata is not absent code.
        monkeypatch.setattr(_requires, "_installed_version", lambda name: None)
        monkeypatch.setattr(_requires, "_importable", lambda name: True)
        assert check(ns, "pkg:some_module")["ok"]

    def test_unreadable_version_is_unknown_not_a_pass(self, ns, monkeypatch):
        monkeypatch.setattr(_requires, "_installed_version", lambda name: None)
        monkeypatch.setattr(_requires, "_importable", lambda name: True)
        res = check(ns, "pkg:some_module>=2.0")
        assert res["met"] == [] and res["unmet"] == []
        assert "cannot be checked" in res["unknown"][0]

    @pytest.mark.parametrize(
        "installed,spec,ok",
        [
            ("0.11.0", ">=0.11.0", True),
            ("0.11.0", ">=0.12.0", False),
            ("0.12.1", ">=0.12", True),
            ("0.12", ">=0.12.0", True),  # zero-padded, not lexicographic
            ("0.9.0", ">=0.10.0", False),  # numeric, not string, compare
            ("0.12.0", "==0.12", True),
            ("0.13.0", "==0.12", False),
            # A pre-release of the required version counts as meeting it: a strict
            # PEP 440 compare ranks every rc below its own final release and would
            # tell a dev build to upgrade to what it is already running.
            ("0.12.0rc8.dev32+g9268773", ">=0.12.0", True),
            ("0.11.0rc8.dev32+g9268773", ">=0.12.0", False),
        ],
    )
    def test_version_comparison(self, ns, monkeypatch, installed, spec, ok):
        monkeypatch.setattr(_requires, "_installed_version", lambda name: installed)
        res = check(ns, f"pkg:biopb-mcp{spec}")
        assert res["ok"] is ok
        if not ok:
            assert installed in res["unmet"][0]

    def test_an_unsupported_operator_is_unknown(self, ns, monkeypatch):
        # `<`, `!=`, `~=` are not in the vocabulary. Guessing is worse than asking.
        monkeypatch.setattr(_requires, "_installed_version", lambda name: "0.5.0")
        res = check(ns, "pkg:biopb-mcp~=0.1")
        assert res["unmet"] == [] and "only >= and ==" in res["unknown"][0]

    def test_a_broken_parent_package_does_not_raise(self, ns, monkeypatch):
        def boom(name):
            raise ModuleNotFoundError("no parent")

        monkeypatch.setattr("importlib.util.find_spec", boom)
        monkeypatch.setattr(_requires, "_installed_version", lambda name: None)
        assert check(ns, "pkg:whatever")["unmet"]


class TestUnknownVocabulary:
    def test_an_unrecognised_token_is_reported_not_passed(self, ns):
        # The vocabulary is curated in biopb-site and grows without this file
        # needing an edit. Silently treating a new token as met would hide exactly
        # the requirement the publisher went out of their way to declare.
        res = check(ns, "gpu")
        assert res["met"] == [] and res["unmet"] == []
        assert res["unknown"] == [
            "gpu — not a requirement this kernel knows how to check"
        ]

    def test_unknown_tokens_do_not_flip_ok(self, ns):
        assert check(ns, "gpu", "viewer")["ok"] is True


class TestErrorContainment:
    def test_a_raising_check_becomes_unknown_with_the_error(self, ns, monkeypatch):
        def boom(name):
            raise RuntimeError("record unreadable")

        monkeypatch.setattr(_requires, "_check_plugin", boom)
        res = check(ns, "plugin:x")
        assert res["unmet"] == []
        assert "the check itself failed" in res["unknown"][0]

    def test_one_bad_token_does_not_cost_the_others(self, ns, monkeypatch):
        def boom(name):
            raise RuntimeError("nope")

        monkeypatch.setattr(_requires, "_check_plugin", boom)
        ns.pop("da")
        res = check(ns, "plugin:x", "dask", "viewer")
        assert res["unmet"] and res["unknown"] and res["met"] == ["viewer"]


class TestInputTolerance:
    def test_a_find_skills_entry_can_be_passed_whole(self, ns):
        entry = {"id": "qc", "requires": ["viewer"]}
        assert _requires.check(entry, ns)["met"] == ["viewer"]

    def test_a_single_token_string_is_not_iterated_by_character(self, ns):
        assert _requires.check("viewer", ns)["met"] == ["viewer"]

    @pytest.mark.parametrize("empty", [None, [], {}, ""])
    def test_nothing_to_check_is_ok(self, ns, empty):
        res = _requires.check(empty, ns)
        assert res == {"ok": True, "met": [], "unmet": [], "unknown": []}

    def test_junk_is_tolerated(self, ns):
        assert _requires.check(42, ns)["ok"]

    def test_non_string_items_are_coerced_not_crashed_on(self, ns):
        assert _requires.check([None, 3, "viewer"], ns)["met"] == ["viewer"]


class TestNamespaceBinding:
    def test_the_checker_reads_the_namespace_live(self, ns):
        # Bound at bootstrap, called much later: a connection that came up in the
        # meantime must be seen, or the answer is stale by design.
        checker = _requires.make_checker(ns)
        ns["_conn"].client = None
        assert not checker(["tensor"])["ok"]
        ns["_conn"].client = object()
        assert checker(["tensor"])["ok"]

    def test_it_carries_the_docstring_the_agent_reads(self, ns):
        # inspect_object("check_skill_requirements") is the whole documentation for
        # this call, so an empty docstring would be a silent regression.
        doc = _requires.make_checker(ns).__doc__
        assert "requires" in doc and "consent" in doc
