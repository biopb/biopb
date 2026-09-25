"""The optional `biopb tensor` / `biopb image` groups load their module on first use.

`biopb version` used to pay for `dask.array` (and pandas, scipy behind it)
because the subcommand modules were imported at registration. The group is now
a name and a help string until it is invoked or its help rendered; a broken
module is reported at that point, not at startup.
"""

import subprocess
import sys

import biopb.cli as cli
import typer
from typer.testing import CliRunner

runner = CliRunner()

_HEAVY = ("dask", "imageio", "grpc")


def test_importing_the_cli_loads_no_subcommand_module():
    # A fresh interpreter, so nothing already imported by the test session masks it.
    code = (
        "import sys, biopb.cli; "
        f"print(sorted(m for m in {_HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_top_level_help_lists_the_groups_without_loading_them():
    code = (
        "import sys; from typer.testing import CliRunner; import biopb.cli as cli; "
        "res = CliRunner().invoke(cli.app, ['--help']); "
        "assert res.exit_code == 0, res.output; "
        "assert 'tensor' in res.output and 'image' in res.output; "
        f"print(sorted(m for m in {_HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_the_group_help_lists_the_module_commands():
    res = runner.invoke(cli.app, ["tensor", "--help"])
    assert res.exit_code == 0, res.output
    assert "query" in res.output and "metadata" in res.output


def test_a_broken_module_is_reported_on_invocation_only():
    lazy = type(
        "_LazyBroken", (cli._LazySubcommands,), {"import_path": "biopb._no_such_cli"}
    )
    app = typer.Typer()
    app.add_typer(typer.Typer(), name="broken", help="A group.", cls=lazy)

    @app.command()
    def other():
        print("other ran")

    res = runner.invoke(app, ["other"])
    assert res.exit_code == 0 and "other ran" in res.output

    res = runner.invoke(app, ["broken", "anything", "--flag"])
    assert res.exit_code == 1
    assert "'broken' commands are unavailable" in res.output
    assert "biopb._no_such_cli" in res.output

    res = runner.invoke(app, ["broken", "--help"])
    assert res.exit_code == 0
    assert "unavailable" in res.output
