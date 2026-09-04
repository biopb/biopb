"""Hold both extras parsers to one table.

install.sh and biopb-engine.ps1 each parse ~/.config/biopb/extra-packages.txt,
independently, in two languages. In biopb/biopb#648 they had the same bug --
`#` treated as a comment marker anywhere on the line, so a PEP 508 direct
reference lost its URL fragment and resolved to something else entirely, with no
error to notice. Two implementations of one rule, and nothing making them agree.

This is what makes them agree. Every case in extras-contract.json runs through
BOTH parsers; a fix, a tweak or a rewrite on one side that the other does not
match fails here. Cases belong in the JSON, never inline in this file -- an
inline case is a case only one parser is held to, which is the hole this closes.
"""

from __future__ import annotations

import pytest
from conftest import (
    CONTRACT_CASES,
    bash,
    ps_literal,
    pwsh,
    requires_pwsh,
    write_extras,
)

CASE_IDS = [c["id"] for c in CONTRACT_CASES]


def _parse_bash(config_dir) -> tuple[int, list[str]]:
    """install.sh's _read_extra_packages, as (count, requirements).

    The count is read back separately because it is not a convenience: bash 3.2
    (still what macOS ships) treats expanding an EMPTY array under `set -u` as an
    unbound-variable error, so EXTRA_PACKAGES_COUNT -- not ${#EXTRA_PACKAGES[@]}
    -- gates every use of the array. A count that disagrees with the array is an
    installer that either drops the user's packages or aborts on an empty file.
    """
    out = bash(
        # CONFIG_DIR arrives through the environment rather than as an assignment
        # spliced into the script -- an interpolated path with a space in it would
        # otherwise turn one variable into two words.
        "_read_extra_packages\n"
        'printf "COUNT=%s\\n" "$EXTRA_PACKAGES_COUNT"\n'
        'if [ "$EXTRA_PACKAGES_COUNT" -gt 0 ]; then\n'
        '    printf "%s\\n" "${EXTRA_PACKAGES[@]}"\n'
        "fi\n",
        env={"CONFIG_DIR": str(config_dir)},
    )
    return _split(out.stdout)


def _parse_pwsh(config_dir) -> tuple[int, list[str]]:
    """biopb-engine.ps1's Read-ExtraPackages, in the same shape."""
    out = pwsh(
        f"$reqs = @(Read-ExtraPackages -ConfigDir {ps_literal(str(config_dir))})\n"
        '"COUNT=$($reqs.Count)"\n'
        "$reqs | ForEach-Object { $_ }\n"
    )
    return _split(out.stdout)


def _split(stdout: str) -> tuple[int, list[str]]:
    lines = stdout.splitlines()
    assert lines and lines[0].startswith("COUNT="), stdout
    return int(lines[0][len("COUNT=") :]), lines[1:]


PARSER_PARAMS = [
    pytest.param(_parse_bash, id="install.sh"),
    pytest.param(_parse_pwsh, id="biopb-engine.ps1", marks=requires_pwsh),
]


@pytest.mark.parametrize("parse", PARSER_PARAMS)
@pytest.mark.parametrize("case", CONTRACT_CASES, ids=CASE_IDS)
def test_contract(parse, case, tmp_path):
    write_extras(tmp_path, case["input"])
    count, requirements = parse(tmp_path)
    assert requirements == case["expected"], case["why"]
    assert count == len(case["expected"]), "count must agree with the array"


@pytest.mark.parametrize("parse", PARSER_PARAMS)
def test_no_file_is_no_extras(parse, tmp_path):
    """A user who never created the file gets an install, not an error.

    Distinct from the empty-file case in the contract: this is the state EVERY
    machine is in until someone writes the file, so the read has to return
    cleanly with the config dir itself untouched.
    """
    count, requirements = parse(tmp_path)
    assert (count, requirements) == (0, [])
    assert not (tmp_path / "extra-packages.txt").exists(), (
        "the file is the user's; never created for them"
    )


@pytest.mark.parametrize("parse", PARSER_PARAMS)
def test_missing_config_dir_is_no_extras(parse, tmp_path):
    """The config dir does not exist yet on a first install. Not an error either."""
    assert parse(tmp_path / "not-created-yet") == (0, [])


def test_contract_is_not_trivially_satisfiable():
    """Guard the guard: the table must still contain the case that motivated it.

    A contract file is only worth its CI seconds if the regression that created
    it is in there. If someone prunes the fragment cases, every parser test still
    passes and the drift protection is gone -- so fail here instead.
    """
    by_id = {c["id"]: c for c in CONTRACT_CASES}
    for required in (
        "git-subdirectory-fragment",
        "wheel-sha256-fragment",
        "fragment-plus-real-comment",
    ):
        assert required in by_id, (
            f"{required} is the #648 regression; it does not get removed"
        )
        assert "#" in by_id[required]["expected"][0], (
            "the fragment has to survive, or the case proves nothing"
        )
