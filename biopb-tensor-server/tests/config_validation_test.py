"""Config value validation: clamp at the read step, report at the strict
surfaces (biopb/biopb#34).

Out-of-range / bad-enum knobs are caught where a config is *read* -- the shared
``biopb._config_validate`` checker biopb-mcp and the control use too -- and
replaced with their defaults, with a warning naming the key. They never reach the
request path (the actual bug: ``downscale_factor=0`` -> ZeroDivisionError in
GetFlightInfo), and they never stop a supervised server from starting either.
``validate_config_dict`` is the strict half: same rules, reported per field, for
the admin form and ``biopb-tensor-server validate``.
"""

import logging

import pytest
from biopb_tensor_server.core.config import (
    CacheConfig,
    MetadataDbConfig,
    PrecacheConfig,
    PyramidConfig,
    ServerConfig,
    parse_config,
    validate_config_dict,
)


def _violations(caplog):
    return [m for m in caplog.messages if "Invalid config value" in m]


def _section_of(config, section):
    """The dataclass holding *section*'s fields (``server`` is the root)."""
    return config if section == "server" else getattr(config, section)


def _default_of(section, field):
    defaults = {
        "server": ServerConfig,
        "cache": CacheConfig,
        "pyramid": PyramidConfig,
        "precache": PrecacheConfig,
        "metadata_db": MetadataDbConfig,
    }
    return getattr(defaults[section](), field)


# --- the concrete failure modes named in the issue ---------------------------
# Raw on-disk configs, so the on-disk key (`cache.max_bytes`) and the dataclass
# field it feeds (`memory_max_bytes`) are both exercised.


@pytest.mark.parametrize(
    "raw, section, field",
    [
        ({"pyramid": {"downscale_factor": 0}}, "pyramid", "downscale_factor"),
        ({"pyramid": {"downscale_factor": 1}}, "pyramid", "downscale_factor"),
        (
            {"pyramid": {"pixel_budget_cubic_root": 0}},
            "pyramid",
            "pixel_budget_cubic_root",
        ),
        ({"pyramid": {"threshold": 0}}, "pyramid", "threshold"),
        ({"pyramid": {"reduction_method": "bogus"}}, "pyramid", "reduction_method"),
        # "precompute" is protocol vocabulary (request a native on-disk level),
        # not a way to compute a pyramid level -- invalid as config.
        (
            {"pyramid": {"reduction_method": "precompute"}},
            "pyramid",
            "reduction_method",
        ),
        (
            {"pyramid": {"reduction_method": "PRECOMPUTED"}},
            "pyramid",
            "reduction_method",
        ),
        ({"cache": {"backend": "bogus"}}, "cache", "backend"),
        ({"cache": {"max_bytes": 0}}, "cache", "memory_max_bytes"),
        ({"cache": {"file_max_total_gb": -1}}, "cache", "file_max_total_bytes"),
        ({"precache": {"backlog_high_water": 1.5}}, "precache", "backlog_high_water"),
        ({"precache": {"backlog_high_water": -0.1}}, "precache", "backlog_high_water"),
        (
            {"metadata_db": {"max_query_results": 0}},
            "metadata_db",
            "max_query_results",
        ),
        ({"metadata_db": {"query_timeout_ms": 0}}, "metadata_db", "query_timeout_ms"),
        ({"server": {"port": -1}}, "server", "port"),
        ({"server": {"port": 99999}}, "server", "port"),
        ({"server": {"rescan_interval": -1.0}}, "server", "rescan_interval"),
    ],
)
def test_bad_value_is_clamped_with_a_warning(raw, section, field, caplog):
    with caplog.at_level(logging.WARNING):
        config = parse_config(raw)  # must NOT raise: the server still starts
    msgs = _violations(caplog)
    assert msgs, "expected a validation warning"
    # The log line carries the full dotted path, so the key is findable in the file.
    assert f"{section}.{field}" in msgs[0]
    # ...and the value in force is the default, not the rejected one.
    assert getattr(_section_of(config, section), field) == _default_of(section, field)


def test_warning_describes_accepted_range_and_enum_and_the_default_used(caplog):
    with caplog.at_level(logging.WARNING):
        parse_config({"pyramid": {"downscale_factor": 0}, "cache": {"backend": "bad"}})
    joined = "\n".join(_violations(caplog))
    assert ">= 2" in joined  # range
    assert "file" in joined and "memory" in joined  # enum members
    assert "using the default" in joined  # what actually ran


# --- legitimate values that must pass through untouched ----------------------


def test_valid_defaults_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        parse_config({})
    assert not _violations(caplog)


@pytest.mark.parametrize(
    "raw, section, field, expected",
    [
        # full_rescan_interval <= 0 disables the periodic full scan (sentinel).
        (
            {"server": {"full_rescan_interval": 0.0}},
            "server",
            "full_rescan_interval",
            0.0,
        ),
        # port 0 = bind an OS-assigned ephemeral port (a sentinel, not a typo).
        ({"server": {"port": 0}}, "server", "port", 0),
        ({"server": {"port": 65535}}, "server", "port", 65535),
        ({"server": {"log_level": "debug"}}, "server", "log_level", "debug"),
        # reduction_method aliases + case-insensitivity (the computable subset;
        # "precompute" is protocol-only, tested above).
        (
            {"pyramid": {"reduction_method": "mean"}},
            "pyramid",
            "reduction_method",
            "mean",
        ),
        (
            {"pyramid": {"reduction_method": "STRIDE"}},
            "pyramid",
            "reduction_method",
            "STRIDE",
        ),
        # "linear" is a tolerated deprecated alias (folds to "area" at read time).
        (
            {"pyramid": {"reduction_method": "linear"}},
            "pyramid",
            "reduction_method",
            "linear",
        ),
        # boundaries are inclusive.
        (
            {"precache": {"backlog_high_water": 0.0}},
            "precache",
            "backlog_high_water",
            0.0,
        ),
        (
            {"precache": {"backlog_high_water": 1.0}},
            "precache",
            "backlog_high_water",
            1.0,
        ),
    ],
)
def test_legitimate_values_survive_silently(raw, section, field, expected, caplog):
    with caplog.at_level(logging.WARNING):
        config = parse_config(raw)
    assert not _violations(caplog)
    assert getattr(_section_of(config, section), field) == expected


def test_every_bad_section_is_reported_not_just_the_first(caplog):
    # The load path reports all of them in one pass -- fixing one and rediscovering
    # the next on the following start would be a miserable loop.
    with caplog.at_level(logging.WARNING):
        config = parse_config(
            {
                "server": {"port": 70000},
                "pyramid": {"downscale_factor": 0},
                "cache": {"backend": "nope"},
            }
        )
    joined = "\n".join(_violations(caplog))
    assert "server.port" in joined
    assert "pyramid.downscale_factor" in joined
    assert "cache.backend" in joined
    assert (config.port, config.pyramid.downscale_factor, config.cache.backend) == (
        ServerConfig().port,
        PyramidConfig().downscale_factor,
        CacheConfig().backend,
    )


# --- validate_config_dict: the endpoint's semantic gate ----------------------
# Same _CONSTRAINTS rules the server enforces at load, returned as structured
# {path, message} problems rather than raised, so the admin config-save endpoint
# can report every problem at once and accepts exactly what the server loads.


def test_validate_config_dict_valid_is_empty():
    assert validate_config_dict({"server": {"port": 8815, "log_level": "info"}}) == []


def test_validate_config_dict_flags_case_insensitive_enum():
    # The gap the JSON Schema cannot express (no hard `enum` for a folded set):
    # a bad log_level must still be caught here, on its on-disk path.
    problems = validate_config_dict({"server": {"log_level": "VERBOSE"}})
    assert [p["path"] for p in problems] == [["server", "log_level"]]
    assert "log_level" in problems[0]["message"]


def test_validate_config_dict_flags_reduction_method():
    problems = validate_config_dict({"pyramid": {"reduction_method": "bogus"}})
    assert [p["path"] for p in problems] == [["pyramid", "reduction_method"]]


def test_validate_config_dict_uses_ondisk_paths():
    # A field whose wire section diverges from the dataclass (memory_max_entries
    # is a CacheConfig field but lives at [cache] max_entries on disk) reports
    # the on-disk path, so it dedupes against the JSON Schema's path at the
    # endpoint.
    problems = validate_config_dict({"cache": {"max_entries": 0}})
    assert [p["path"] for p in problems] == [["cache", "max_entries"]]
    # The message names the on-disk key too, matching the path -- not the internal
    # `memory_max_entries` field the form has no name for.
    assert problems[0]["message"].startswith("max_entries=")
    assert "memory_max_entries" not in problems[0]["message"]


def test_validate_config_dict_ignores_removed_compute_section():
    # The [compute] section was removed with the GPU backend; it is
    # warn-and-ignore at parse time and must NOT be a validation problem.
    assert validate_config_dict({"compute": {"backend": "quantum"}}) == []


def test_validate_config_dict_reports_instead_of_raising():
    # The endpoint's surface returns problems where the load path raises, so a
    # bad form submission is a 422 listing them, not a 500.
    assert validate_config_dict({"server": {"port": 70000}})


def test_validate_config_dict_structural_error_is_reported():
    # A source without url can't even be constructed; surface it, don't crash.
    problems = validate_config_dict({"sources": [{"type": "zarr"}]})
    assert problems and problems[0]["path"] == []


def test_validate_config_dict_malformed_section_is_reported_not_raised():
    # A wrong-typed section makes parse_config walk a non-dict (str.get ->
    # AttributeError, not ValueError/TypeError). It must still be reported as a
    # root problem, never propagate -- otherwise the admin endpoint 500s instead
    # of returning a clean 422.
    problems = validate_config_dict({"server": "not-a-dict"})
    assert problems and problems[0]["path"] == []


# --- unknown-key warnings (the silent drop-to-default trap) ------------------


def _unknown(caplog):
    return [m for m in caplog.messages if m.startswith("Unknown config ")]


def test_misnamed_cache_key_warns_naming_the_right_key(caplog):
    # The reported trap: the dataclass field name `memory_max_entries` instead
    # of the file key `max_entries` -> silently keeps the default. Must warn and
    # point at the real key.
    with caplog.at_level(logging.WARNING):
        cfgobj = parse_config(
            {"cache": {"backend": "memory", "memory_max_entries": 1, "max_bytes": 1}}
        )
    msgs = _unknown(caplog)
    assert any("memory_max_entries" in m and "[cache]" in m for m in msgs)
    # The warning lists the accepted keys so the fix is discoverable.
    assert any("max_entries" in m and "max_bytes" in m for m in msgs)
    # The recognized key still took effect; the bogus one was ignored.
    assert cfgobj.cache.memory_max_bytes == 1
    assert cfgobj.cache.memory_max_entries == 1024  # default (key was dropped)


def test_unknown_top_level_section_warns(caplog):
    with caplog.at_level(logging.WARNING):
        parse_config({"kache": {"backend": "memory"}})
    assert any(
        "[kache]" in m and "Unknown config section" in m for m in _unknown(caplog)
    )


def test_unknown_keys_in_sources_and_profiles_warn(caplog):
    with caplog.at_level(logging.WARNING):
        parse_config(
            {
                "sources": [{"url": "/data/a.zarr", "kloud": True}],
                "credentials": {
                    "profiles": [{"name": "p", "storage_type": "s3", "secrt": "x"}]
                },
            }
        )
    joined = "\n".join(_unknown(caplog))
    assert "kloud" in joined and "[sources]" in joined
    assert "secrt" in joined and "[credentials.profiles]" in joined


def test_valid_config_does_not_warn_unknown(caplog):
    # A full config of legitimate keys -- including legacy aliases -- stays quiet.
    with caplog.at_level(logging.WARNING):
        parse_config(
            {
                "server": {
                    "host": "127.0.0.1",
                    "port": 8815,
                    "watcher_type": "off",  # legacy alias
                    "poll_interval": 15.0,  # legacy alias
                    "aggressive_dir_pruning": True,
                },
                "cache": {"backend": "memory", "max_entries": 1, "max_bytes": 1},
                "precache": {
                    "enabled": True,
                    "downscale_factor": 4,
                },  # back-compat knob
                "metadata_db": {"max_query_results": 100},
                "sources": [{"path": "/data/a.zarr", "cloud": True}],  # legacy `path`
            }
        )
    assert not _unknown(caplog)


# --- defaults are owned solely by the dataclasses (biopb/biopb#277 item A) ----


def _scalar_fields(cls):
    import dataclasses

    inst = cls()
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        value = getattr(inst, f.name)
        if dataclasses.is_dataclass(value) or isinstance(value, list):
            continue  # nested section / list, not a scalar default
        yield f.name


def test_empty_config_reproduces_dataclass_defaults():
    """An empty config must construct every scalar exactly as the dataclass
    default. If parse_config re-hardcoded a default that drifted from the
    dataclass, this diverges -- so no default is written in two places."""
    parsed = parse_config({})
    for section_obj, defaults in (
        (parsed, ServerConfig()),
        (parsed.cache, CacheConfig()),
        (parsed.pyramid, PyramidConfig()),
        (parsed.precache, PrecacheConfig()),
        (parsed.metadata_db, MetadataDbConfig()),
    ):
        for name in _scalar_fields(type(defaults)):
            assert getattr(section_obj, name) == getattr(defaults, name), (
                f"{type(defaults).__name__}.{name}: parse_config({{}}) gave "
                f"{getattr(section_obj, name)!r}, dataclass default is "
                f"{getattr(defaults, name)!r}"
            )


def test_present_keys_override_defaults_only_where_set():
    """A partial config carries only the keys it sets; the rest fall through to
    the dataclass defaults (including the on-disk MB->bytes alias)."""
    default_cache = CacheConfig()
    cfgobj = parse_config(
        {
            "server": {"port": 9000},
            "cache": {"file_max_segment_mb": 32},
        }
    )
    assert cfgobj.port == 9000
    assert cfgobj.host == ServerConfig().host  # untouched -> default
    assert cfgobj.cache.file_max_segment_bytes == 32 * 1024 * 1024
    assert cfgobj.cache.memory_max_bytes == default_cache.memory_max_bytes
