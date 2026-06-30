"""save_config: the admin endpoint's config writer (biopb/biopb#237).

Covers the three behaviors the admin PUT route depends on: a raw-dict round-trip
that preserves unsurfaced keys, the sibling JSON Schema + relative ``$schema``
embed, and the legacy TOML -> JSON migration with backup. Also asserts the
embedded ``$schema`` meta key does not trip the unknown-key warning.
"""

import json
import logging
from pathlib import Path

from biopb_tensor_server.config import (
    CANONICAL_CONFIG_NAME,
    REDACTED_SENTINEL,
    SCHEMA_SIDECAR_NAME,
    load_config,
    redact_config_secrets,
    restore_redacted_secrets,
    save_config,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _config_with_creds() -> dict:
    return {
        "server": {"port": 8815},
        "credentials": {
            "profiles": [
                {
                    "name": "aws-prod",
                    "storage_type": "s3",
                    "key": "AKIA...",
                    "secret": "supersecret",
                    "token": "sess-tok",
                    "region": "us-east-1",
                }
            ]
        },
    }


def test_round_trips_raw_dict_preserving_unknown_keys(tmp_path):
    path = tmp_path / CANONICAL_CONFIG_NAME
    data = {
        "server": {"host": "127.0.0.1", "port": 9000},
        "cache": {"backend": "memory"},
        # An advanced / future key the form never surfaces -- must survive.
        "experimental_knob": {"deep": [1, 2, 3]},
    }

    returned = save_config(dict(data), path)
    assert returned == path

    on_disk = _read(path)
    assert on_disk["server"] == {"host": "127.0.0.1", "port": 9000}
    assert on_disk["experimental_knob"] == {"deep": [1, 2, 3]}


def test_writes_sibling_schema_and_relative_pointer(tmp_path):
    path = tmp_path / CANONICAL_CONFIG_NAME
    save_config({"server": {"port": 9000}}, path)

    on_disk = _read(path)
    assert on_disk["$schema"] == f"./{SCHEMA_SIDECAR_NAME}"

    schema_path = tmp_path / SCHEMA_SIDECAR_NAME
    assert schema_path.exists()
    schema = _read(schema_path)
    assert schema.get("$schema", "").startswith("https://json-schema.org/")
    assert "properties" in schema


def test_does_not_mutate_caller_dict(tmp_path):
    data = {"server": {"port": 9000}}
    save_config(data, tmp_path / CANONICAL_CONFIG_NAME)
    # The $schema embed happens on a copy, not the caller's dict.
    assert "$schema" not in data


def test_migrates_legacy_toml_to_json_with_backup(tmp_path):
    toml_path = tmp_path / "biopb.toml"
    toml_path.write_text("[server]\nport = 8815\n")

    returned = save_config({"server": {"port": 9000}}, toml_path)

    # Write redirected to the canonical JSON sibling...
    assert returned == tmp_path / CANONICAL_CONFIG_NAME
    assert returned.exists()
    assert _read(returned)["server"]["port"] == 9000
    # ...and the legacy file was backed up out of the way (no shadow warning).
    assert not toml_path.exists()
    assert (tmp_path / "biopb.toml.bak").exists()


def test_saved_config_loads_back_without_unknown_key_warning(tmp_path, caplog):
    path = tmp_path / CANONICAL_CONFIG_NAME
    save_config({"server": {"host": "127.0.0.1", "port": 9000}}, path)

    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)

    assert cfg.port == 9000
    # The embedded $schema must not be reported as an unknown section.
    assert not any("$schema" in rec.getMessage() for rec in caplog.records)
    assert not any(
        "Unknown config section" in rec.getMessage() for rec in caplog.records
    )


# --- credential-secret redaction (biopb/biopb#237) --------------------------


def test_redact_masks_credential_secrets_only():
    cfg = _config_with_creds()
    out = redact_config_secrets(cfg)
    prof = out["credentials"]["profiles"][0]
    assert prof["key"] == REDACTED_SENTINEL
    assert prof["secret"] == REDACTED_SENTINEL
    assert prof["token"] == REDACTED_SENTINEL
    # Non-secret fields pass through untouched.
    assert prof["region"] == "us-east-1"
    assert prof["name"] == "aws-prod"
    # The caller's dict is not mutated (deep copy).
    assert cfg["credentials"]["profiles"][0]["secret"] == "supersecret"


def test_redact_ignores_configs_without_credentials():
    assert redact_config_secrets({"server": {"port": 1}}) == {"server": {"port": 1}}


def test_restore_resolves_sentinel_from_existing_by_name():
    existing = _config_with_creds()
    redacted = redact_config_secrets(existing)
    # The form sends the redacted profile back unchanged, but edits the region.
    redacted["credentials"]["profiles"][0]["region"] = "eu-west-1"

    merged = restore_redacted_secrets(redacted, existing)
    prof = merged["credentials"]["profiles"][0]
    assert prof["secret"] == "supersecret"  # real value restored from disk
    assert prof["key"] == "AKIA..."
    assert prof["token"] == "sess-tok"
    assert prof["region"] == "eu-west-1"  # the genuine edit is kept


def test_restore_keeps_a_genuinely_new_secret():
    existing = _config_with_creds()
    incoming = _config_with_creds()
    incoming["credentials"]["profiles"][0]["secret"] = "rotated-secret"
    merged = restore_redacted_secrets(incoming, existing)
    assert merged["credentials"]["profiles"][0]["secret"] == "rotated-secret"


def test_restore_drops_sentinel_with_no_prior_value():
    # A brand-new profile (no match on disk) that still carries the sentinel must
    # never persist the literal mask.
    incoming = {
        "credentials": {
            "profiles": [
                {"name": "new", "storage_type": "s3", "secret": REDACTED_SENTINEL}
            ]
        }
    }
    merged = restore_redacted_secrets(incoming, {})
    assert "secret" not in merged["credentials"]["profiles"][0]
