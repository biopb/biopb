#!/usr/bin/env python3
"""Installer wheel-coverage check (manual / workbench only).

Mirrors the dependency set ``install/install.sh`` installs and verifies that, for
a given target platform, every resolved third-party package that ships
platform-specific wheels also ships one for that platform -- i.e.
``curl install.sh | bash`` would not silently fall back to compiling a C/Rust
extension from source. That fallback is exactly what broke Intel-macOS installs
when ``cryptography`` 49 dropped its x86_64/universal2 wheel (see ``install.sh``
and issue #45's sibling, #355).

It resolves the SAME requirements install.sh does, *including* the dependency
overrides install.sh applies (read out of install.sh itself by
``installer_overrides``), so a clean run means "an installer-equivalent resolve is
wheel-clean on this platform," not merely "the raw dependency graph is." Run from
the repo root.

Why not a uv ``--only-binary`` diff: ``--only-binary=:all:`` also rejects
pure-Python *sdist-only* packages (e.g. ``asciitree``), which build anywhere
without a compiler -- so it flags harmless packages. Instead we resolve
build-allowed (what install.sh actually does) and ask PyPI, per resolved package,
whether a wheel for the target exists. A package is flagged ONLY when it publishes
platform wheels for other platforms but none compatible with the target -- the
cryptography-49 shape. Pure-Python packages (a ``py3-none-any`` wheel, or no wheel
at all) are ignored.

A wheel counts only if its interpreter tags also fit ``--python-version``: a
pure-Python tag (``py3``), the exact ``cpXY``, or an ``abi3`` wheel built for that
minor or an earlier one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The format extras install.sh always installs (see install_biopb(): TENSOR_EXTRAS).
_BASE_TENSOR_EXTRAS = ["web", "vendor", "qptiff", "medical", "ndtiff"]

# install.sh adds the Zeiss CZI reader ([czi] -> bioio-czi -> pylibczirw) and TLS
# cert generation ([tls] -> cryptography) on every platform EXCEPT Intel macOS,
# where neither ships a wheel. Mirror that so the check reflects the set install.sh
# actually installs per target.
_CZI_TLS_UNAVAILABLE_TARGETS = {"x86_64-apple-darwin"}


def installer_requirements(target: str) -> list[str]:
    """The requirement set install.sh installs for ``target``.

    Local packages are given as paths so uv reads their real dependency metadata
    from this checkout rather than from a published release.
    """
    extras = list(_BASE_TENSOR_EXTRAS)
    if target not in _CZI_TLS_UNAVAILABLE_TARGETS:
        extras += ["czi", "tls"]
    return [
        ".[tensor]",
        f"./biopb-tensor-server[{','.join(extras)}]",
        "./biopb-mcp[napari]",
        "./biopb-control",
        "napari[all]",
    ]


_INSTALL_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "install.sh"
)

_MAX_MINOR = re.compile(r"^\s*MAX_MINOR=(\d+)\s*$", re.M)


def installer_python_version(install_sh: str = _INSTALL_SH) -> str:
    """The interpreter install.sh installs on (``3.12``): its MAX_MINOR."""
    with open(install_sh, encoding="utf-8") as fh:
        match = _MAX_MINOR.search(fh.read())
    if match is None:
        raise SystemExit(f"no MAX_MINOR= in {install_sh}")
    return f"3.{match.group(1)}"


# The single `printf ... > "$WHEELS_DIR/overrides.txt"` install.sh writes its
# --overrides file with.
_OVERRIDES_PRINTF = re.compile(
    r"printf\s+'([^']*)'\s*>\s*\"\$WHEELS_DIR/overrides\.txt"
)


def installer_overrides(install_sh: str = _INSTALL_SH) -> list[str]:
    """The override lines install.sh writes, read out of install.sh itself.

    A green run has to reflect the *installer's* resolve, not the unconstrained
    graph, so this file needs the same overrides. It reads them rather than
    restating them: the hand-copied list silently went stale when install.sh
    gained the `mcp<2` cap, and a workbench that resolves a closure the installer
    no longer produces is worse than no workbench. Parsing keeps the two in step
    by construction and works on any branch, whichever overrides that branch's
    install.sh happens to apply.

    Raises rather than degrading to an empty list: an unconstrained resolve looks
    like a pass here, so a parse that stops matching must be loud.
    """
    with open(install_sh, encoding="utf-8") as fh:
        text = fh.read()
    m = _OVERRIDES_PRINTF.search(text)
    if m is None:
        raise SystemExit(
            f"could not find the overrides printf in {install_sh}; "
            "update _OVERRIDES_PRINTF to match how install.sh writes overrides.txt"
        )
    lines = [line.strip() for line in m.group(1).split("\\n") if line.strip()]
    if not lines:
        raise SystemExit(f"parsed an empty override set from {install_sh}")
    return lines


def _mac_x86(tag: str) -> bool:
    return "macosx" in tag and ("x86_64" in tag or "universal2" in tag)


def _mac_arm(tag: str) -> bool:
    return "macosx" in tag and ("arm64" in tag or "universal2" in tag)


def _linux_x86(tag: str) -> bool:
    return (
        "manylinux" in tag or "musllinux" in tag or tag.startswith("linux")
    ) and "x86_64" in tag


def _win_amd64(tag: str) -> bool:
    return tag == "win_amd64"


# uv --python-platform target -> predicate deciding whether a wheel platform tag
# is compatible with that target.
TARGETS = {
    "x86_64-apple-darwin": _mac_x86,
    "aarch64-apple-darwin": _mac_arm,
    "x86_64-unknown-linux-gnu": _linux_x86,
    "x86_64-pc-windows-msvc": _win_amd64,
}

_PKG_LINE = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;]+)")


def resolve(target: str, python_version: str) -> dict[str, str]:
    """Resolve the installer's requirement set for ``target`` (build allowed).

    Returns {normalized_name: version} for every PyPI-pinned package. Local
    (path/file) requirements resolve to non-``name==version`` lines and are
    skipped -- they are this repo's own packages, never the wheel-gap risk.
    """
    reqs = installer_requirements(target)
    # install.sh passes its overrides via `--overrides <file>`; mirror that with a
    # temp file rather than folding them into the requirement list (an override is
    # not a plain requirement -- it rewrites another package's dep, e.g. dropping
    # pyjwt's `crypto` extra, which a bare requirement line cannot do).
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as ov:
        ov.write("\n".join(installer_overrides()) + "\n")
        override_path = ov.name
    try:
        proc = subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--python-version",
                python_version,
                "--python-platform",
                target,
                "--override",
                override_path,
                "-",
            ],
            input="\n".join(reqs) + "\n",
            text=True,
            capture_output=True,
        )
    finally:
        os.unlink(override_path)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"uv pip compile failed for {target}")
    pinned: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = _PKG_LINE.match(line.strip())
        if m:
            pinned[m.group(1).lower()] = m.group(2)
    return pinned


def interpreter_fits(python_tag: str, abi_tag: str, python_version: str) -> bool:
    """Whether a wheel's python/abi tags can run on ``python_version`` (``3.12``)."""
    minor = int(python_version.split(".")[1])
    for py in python_tag.split("."):
        if py.startswith("py"):
            return True
        if py.startswith("cp3"):
            built = int(py[3:])
            if built == minor or (built < minor and "abi3" in abi_tag.split(".")):
                return True
    return False


def wheel_platform_tags(name: str, version: str, python_version: str) -> dict | None:
    """Fetch the wheel platform tags for ``name==version`` from PyPI.

    Returns {"has_any": bool, "has_wheel": bool, "tags": [platform-tag, ...]}, or
    None if the package is not on PyPI (a local/unpublished package -> nothing to
    check). ``tags`` holds only the wheels that fit ``python_version``.
    """
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    tags: list[str] = []
    has_any = False
    has_wheel = False
    for f in data.get("urls", []):
        fn = f.get("filename", "")
        if not fn.endswith(".whl"):
            continue
        # wheel = name-version[-build]-python-abi-platform.whl; each of the last
        # three fields may be a '.'-joined set of tags.
        python_tag, abi_tag, platform_field = fn[:-4].split("-")[-3:]
        has_wheel = True
        if not interpreter_fits(python_tag, abi_tag, python_version):
            continue
        for tag in platform_field.split("."):
            if tag == "any":
                has_any = True
            tags.append(tag)
    return {"has_any": has_any, "has_wheel": has_wheel, "tags": tags}


def missing_wheel(name: str, version: str, compatible, python_version: str) -> bool:
    info = wheel_platform_tags(name, version, python_version)
    if info is None:  # local / not on PyPI
        return False
    if not info["has_wheel"]:  # sdist-only pure Python -> builds anywhere
        return False
    if info["has_any"]:  # py3-none-any wheel -> installable everywhere
        return False
    return not any(compatible(tag) for tag in info["tags"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument(
        "--python-version",
        default=installer_python_version(),
        help="default: the interpreter install.sh installs on (its MAX_MINOR)",
    )
    args = ap.parse_args()

    compatible = TARGETS[args.target]
    pinned = resolve(args.target, args.python_version)
    print(
        f"Resolved {len(pinned)} PyPI packages for {args.target} "
        f"(py{args.python_version})."
    )

    with ThreadPoolExecutor(max_workers=16) as pool:
        flagged = [
            (n, v)
            for (n, v), bad in zip(
                pinned.items(),
                pool.map(
                    lambda kv: missing_wheel(
                        kv[0], kv[1], compatible, args.python_version
                    ),
                    pinned.items(),
                ),
                strict=True,
            )
            if bad
        ]

    if flagged:
        print(
            f"\n✗ {len(flagged)} package(s) ship wheels for other platforms but "
            f"none for {args.target} -- installer would compile from source:"
        )
        for n, v in sorted(flagged):
            print(f"  - {n}=={v}")
        print(
            "\nThis is the failure mode that breaks `curl install.sh | bash` "
            "on this platform. Pin/override it in install.sh (see the pyjwt->drop-"
            "cryptography override precedent) or upstream a wheel."
        )
        sys.exit(1)

    print(f"✓ every resolved package that ships wheels has one for {args.target}.")


if __name__ == "__main__":
    main()
