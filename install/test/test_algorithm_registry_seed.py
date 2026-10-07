"""The installers' algorithm-registry seed: the off-site cellpose server goes in
only with consent, and only on a fresh install.

Both installers are held to the same cases: install.sh's
_seed_algorithm_registry and the engine's Set-AlgorithmRegistry. The seeded file
is a registry url entry, ``{"url": ...}`` (biopb_control._registry).
"""

from __future__ import annotations

import json

import pytest
from conftest import bash, ps_literal, pwsh, requires_posix, requires_pwsh, sh

CELLPOSE = {"url": "grpcs://cellpose.biopb.org:443"}


def _seed_sh(config_dir, *, env=None):
    bash(f"_seed_algorithm_registry {sh(config_dir)}", env=env)


def _seed_ps(config_dir, *, decline=False):
    flag = " -NoRemotePlugins" if decline else ""
    pwsh(f"Set-AlgorithmRegistry -ConfigDir {ps_literal(config_dir)}{flag}")


def _entries(config_dir):
    return sorted(p.name for p in (config_dir / "algorithms").iterdir())


# --- install.sh --------------------------------------------------------------


@requires_posix
def test_sh_unattended_opt_in_seeds_cellpose(tmp_path):
    _seed_sh(tmp_path, env={"NONINTERACTIVE": "1", "BIOPB_REMOTE_PLUGINS": "1"})
    seeded = tmp_path / "algorithms" / "cellpose.json"
    assert json.loads(seeded.read_text(encoding="utf-8")) == CELLPOSE
    assert _entries(tmp_path) == ["cellpose.json"]


@requires_posix
def test_sh_unattended_without_opt_in_leaves_an_empty_registry(tmp_path):
    _seed_sh(tmp_path, env={"NONINTERACTIVE": "1"})
    assert (tmp_path / "algorithms").is_dir()
    assert _entries(tmp_path) == []


@pytest.mark.parametrize("existing", ["algorithms", "mcp-config.json"])
@requires_posix
def test_sh_keeps_an_existing_registry_or_older_config(tmp_path, existing):
    """A rerun asks nothing: a registry, or an older mcp-config.json the control
    migrates, is the prior choice."""
    if existing == "algorithms":
        (tmp_path / "algorithms").mkdir()
    else:
        (tmp_path / "mcp-config.json").write_text("{}", encoding="utf-8")
    _seed_sh(tmp_path, env={"NONINTERACTIVE": "1", "BIOPB_REMOTE_PLUGINS": "1"})
    assert not (tmp_path / "algorithms" / "cellpose.json").exists()


# --- biopb-engine.ps1 ----------------------------------------------------------


@requires_pwsh
def test_ps_seeds_cellpose_by_default(tmp_path):
    _seed_ps(tmp_path)
    seeded = tmp_path / "algorithms" / "cellpose.json"
    assert json.loads(seeded.read_bytes()) == CELLPOSE
    assert not seeded.read_bytes().startswith(b"\xef\xbb\xbf")
    assert _entries(tmp_path) == ["cellpose.json"]


@requires_pwsh
def test_ps_no_remote_plugins_leaves_an_empty_registry(tmp_path):
    _seed_ps(tmp_path, decline=True)
    assert (tmp_path / "algorithms").is_dir()
    assert _entries(tmp_path) == []


@pytest.mark.parametrize("existing", ["algorithms", "mcp-config.json"])
@requires_pwsh
def test_ps_keeps_an_existing_registry_or_older_config(tmp_path, existing):
    if existing == "algorithms":
        (tmp_path / "algorithms").mkdir()
    else:
        (tmp_path / "mcp-config.json").write_text("{}", encoding="utf-8")
    _seed_ps(tmp_path)
    assert not (tmp_path / "algorithms" / "cellpose.json").exists()
