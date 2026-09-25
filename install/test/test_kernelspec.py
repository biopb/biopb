"""The Jupyter kernel spec both installers write and remove.

install.sh and biopb-engine.ps1 each carry the same ownership program, so the
cases run against both. The test runner's interpreter stands in for the biopb
env (it needs ipykernel, which CI adds), and JUPYTER_DATA_DIR points the
per-user kernel dir into tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest
from conftest import bash, ps_literal, pwsh, requires_posix, requires_pwsh, sh

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ipykernel") is None, reason="needs ipykernel"
)


def _bash(call, py, env):
    return bash(f"{call} {sh(py)}", env=env)


def _pwsh(call, py, env):
    call = {
        "_install_kernelspec": "Install-KernelSpec",
        "_remove_kernelspec": "Remove-KernelSpec",
    }[call]
    return pwsh(f"{call} -Python {ps_literal(py)}", env=env)


RUNNERS = [
    pytest.param(_bash, marks=requires_posix, id="install.sh"),
    pytest.param(_pwsh, marks=requires_pwsh, id="engine.ps1"),
]


@pytest.fixture
def env(tmp_path):
    home = {"HOME": str(tmp_path), "JUPYTER_DATA_DIR": str(tmp_path / "jupyter")}
    if os.name == "nt":
        home["USERPROFILE"] = str(tmp_path)
    return home


@pytest.fixture
def spec(tmp_path):
    return tmp_path / "jupyter" / "kernels" / "biopb" / "kernel.json"


@pytest.mark.parametrize("run", RUNNERS)
def test_install_points_the_spec_at_the_env(run, env, spec):
    run("_install_kernelspec", sys.executable, env)
    written = json.loads(spec.read_text())
    assert written["argv"][0] == sys.executable
    assert written["display_name"] == "Python (biopb)"


@pytest.mark.parametrize("run", RUNNERS)
def test_install_rewrites_its_own_spec(run, env, spec):
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps({"argv": [sys.executable], "display_name": "stale"}))
    run("_install_kernelspec", sys.executable, env)
    assert json.loads(spec.read_text())["display_name"] == "Python (biopb)"


@pytest.mark.parametrize("run", RUNNERS)
def test_a_users_own_spec_is_left_alone(run, env, spec):
    spec.parent.mkdir(parents=True)
    original = json.dumps(
        {"argv": [os.path.join(os.sep, "opt", "elsewhere", "python")]}
    )
    spec.write_text(original)
    run("_install_kernelspec", sys.executable, env)
    run("_remove_kernelspec", sys.executable, env)
    assert spec.read_text() == original


@pytest.mark.parametrize("run", RUNNERS)
def test_remove_deletes_its_own_spec(run, env, spec):
    run("_install_kernelspec", sys.executable, env)
    assert spec.exists()
    run("_remove_kernelspec", sys.executable, env)
    assert not spec.parent.exists()


@pytest.mark.parametrize("run", RUNNERS)
def test_install_can_be_skipped(run, env, spec):
    run("_install_kernelspec", sys.executable, {**env, "BIOPB_INSTALL_KERNELSPEC": "0"})
    assert not spec.exists()


@pytest.mark.parametrize("run", RUNNERS)
def test_an_env_without_jupyter_is_a_skip_not_an_abort(run, env, spec, tmp_path):
    """Best-effort: an interpreter that cannot answer must not end the install."""
    run("_install_kernelspec", str(tmp_path / "no-such-python"), env)
    assert not spec.exists()
