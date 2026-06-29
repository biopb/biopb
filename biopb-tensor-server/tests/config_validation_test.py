"""Declarative config value validation (biopb/biopb#34).

Out-of-range / bad-enum knobs are flagged at construction (warn during the TOML
deprecation window, raise under ``_STRICT_VALIDATION``) instead of blowing up
later on the request path. Enforcement lives in each dataclass's
``__post_init__``, so it covers both file formats and direct construction.
"""

import logging

import pytest
from biopb_tensor_server import config as cfg
from biopb_tensor_server.config import (
    CacheConfig,
    MetadataDbConfig,
    PrecacheConfig,
    PyramidConfig,
    ServerConfig,
    parse_config,
)


def _violations(caplog):
    return [m for m in caplog.messages if "Invalid config value" in m]


# --- the concrete failure modes named in the issue ---------------------------


@pytest.mark.parametrize(
    "factory, section, key",
    [
        (lambda: PyramidConfig(downscale_factor=0), "pyramid", "downscale_factor"),
        (lambda: PyramidConfig(downscale_factor=1), "pyramid", "downscale_factor"),
        (
            lambda: PyramidConfig(pixel_budget_cubic_root=0),
            "pyramid",
            "pixel_budget_cubic_root",
        ),
        (lambda: PyramidConfig(threshold=0), "pyramid", "threshold"),
        (
            lambda: PyramidConfig(reduction_method="bogus"),
            "pyramid",
            "reduction_method",
        ),
        (lambda: CacheConfig(backend="bogus"), "cache", "backend"),
        (lambda: CacheConfig(memory_max_bytes=0), "cache", "memory_max_bytes"),
        (lambda: CacheConfig(file_max_total_bytes=-1), "cache", "file_max_total_bytes"),
        (
            lambda: PrecacheConfig(backlog_high_water=1.5),
            "precache",
            "backlog_high_water",
        ),
        (
            lambda: PrecacheConfig(backlog_high_water=-0.1),
            "precache",
            "backlog_high_water",
        ),
        (
            lambda: MetadataDbConfig(max_query_results=0),
            "metadata_db",
            "max_query_results",
        ),
        (
            lambda: MetadataDbConfig(query_timeout_ms=0),
            "metadata_db",
            "query_timeout_ms",
        ),
        (lambda: ServerConfig(port=0), "server", "port"),
        (lambda: ServerConfig(port=99999), "server", "port"),
        (lambda: ServerConfig(compute_backend="bogus"), "server", "compute_backend"),
        (lambda: ServerConfig(rescan_interval=-1.0), "server", "rescan_interval"),
    ],
)
def test_bad_value_warns_naming_section_and_key(factory, section, key, caplog):
    with caplog.at_level(logging.WARNING):
        obj = factory()  # must NOT raise (warn-only window)
    msgs = _violations(caplog)
    assert msgs, "expected a validation warning"
    assert f"[{section}]" in msgs[0]
    assert key in msgs[0]
    # The offending value is left untouched (we only warn).
    assert getattr(obj, key) is not None


def test_warning_describes_accepted_range_and_enum(caplog):
    with caplog.at_level(logging.WARNING):
        PyramidConfig(downscale_factor=0)
        CacheConfig(backend="bogus")
    joined = "\n".join(_violations(caplog))
    assert ">= 2" in joined  # range
    assert "file" in joined and "memory" in joined  # enum members


# --- legitimate values that must NOT warn ------------------------------------


def test_valid_defaults_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        ServerConfig()  # constructs every nested config at its defaults
    assert not _violations(caplog)


@pytest.mark.parametrize(
    "factory",
    [
        # full_rescan_interval <= 0 disables the periodic full scan (sentinel).
        lambda: ServerConfig(full_rescan_interval=0.0),
        lambda: ServerConfig(full_rescan_interval=-1.0),
        # reduction_method aliases + case-insensitivity (mirrors normalize_*).
        lambda: PyramidConfig(reduction_method="mean"),
        lambda: PyramidConfig(reduction_method="PRECOMPUTED"),
        # log_level / compute_backend accepted forms.
        lambda: ServerConfig(log_level="debug"),
        lambda: ServerConfig(compute_backend="gpu"),
        # boundaries are inclusive.
        lambda: PrecacheConfig(backlog_high_water=0.0),
        lambda: PrecacheConfig(backlog_high_water=1.0),
        lambda: ServerConfig(port=65535),
    ],
)
def test_legitimate_values_are_silent(factory, caplog):
    with caplog.at_level(logging.WARNING):
        factory()
    assert not _violations(caplog)


# --- it flows through the file-load path -------------------------------------


def test_parse_config_warns_on_bad_value(caplog):
    with caplog.at_level(logging.WARNING):
        parse_config(
            {
                "server": {"port": 70000},
                "pyramid": {"downscale_factor": 0},
                "cache": {"backend": "nope"},
            }
        )
    joined = "\n".join(_violations(caplog))
    assert "[server]" in joined and "port" in joined
    assert "[pyramid]" in joined and "downscale_factor" in joined
    assert "[cache]" in joined and "backend" in joined


# --- strict mode raises (the post-migration end state) -----------------------


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setattr(cfg, "_STRICT_VALIDATION", True)
    with pytest.raises(ValueError, match="downscale_factor"):
        PyramidConfig(downscale_factor=0)


def test_strict_mode_allows_valid(monkeypatch):
    monkeypatch.setattr(cfg, "_STRICT_VALIDATION", True)
    ServerConfig()  # no raise
