#!/usr/bin/env python3
"""Installer wheel-coverage check (manual / workbench only).

Mirrors the dependency set ``install/install.sh`` installs and verifies that, for
a given target platform, every resolved third-party package that ships
platform-specific wheels also ships one for that platform -- i.e.
``curl install.sh | bash`` would not silently fall back to compiling a C/Rust
extension from source. That fallback is exactly what broke Intel-macOS installs
when ``cryptography`` 49 dropped its x86_64/universal2 wheel (see ``install.sh``
and issue #45's sibling, #355).

It resolves the SAME requirements install.sh does, *including* the per-platform
constraints install.sh applies (see ``INSTALLER_CONSTRAINTS``), so a clean run
means "an installer-equivalent resolve is wheel-clean on this platform," not
merely "the raw dependency graph is." Run from the repo root.

Why not a uv ``--only-binary`` diff: ``--only-binary=:all:`` also rejects
pure-Python *sdist-only* packages (e.g. ``asciitree``), which build anywhere
without a compiler -- so it flags harmless packages. Instead we resolve
build-allowed (what install.sh actually does) and ask PyPI, per resolved package,
whether a wheel for the target exists. A package is flagged ONLY when it publishes
platform wheels for other platforms but none compatible with the target -- the
cryptography-49 shape. Pure-Python packages (a ``py3-none-any`` wheel, or no wheel
at all) are ignored.

Limitation: matching is by platform tag only, not by Python-version tag; a package
whose only target-platform wheels are for a different Python version is not
flagged. In practice uv has already proven a target-compatible dist exists (the
resolve succeeded), so this only misses the rare "wheel exists but for another
interpreter" case.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The format extras install.sh always installs (see install_biopb(): TENSOR_EXTRAS).
_BASE_TENSOR_EXTRAS = ["web", "aics", "medical", "ndtiff", "hdf5"]

# install.sh adds the Zeiss CZI reader (the [czi] extra -> bioio-czi -> pylibczirw)
# on every platform EXCEPT Intel macOS, where pylibczirw ships no wheel. Mirror that
# so the check reflects the set install.sh actually installs per target.
_CZI_UNAVAILABLE_TARGETS = {"x86_64-apple-darwin"}


def installer_requirements(target: str) -> list[str]:
    """The requirement set install.sh installs for ``target``.

    Local packages are given as paths so uv reads their real dependency metadata
    from this checkout rather than from a published release.
    """
    extras = list(_BASE_TENSOR_EXTRAS)
    if target not in _CZI_UNAVAILABLE_TARGETS:
        extras.append("czi")
    return [
        ".[tensor]",
        f"./biopb-tensor-server[{','.join(extras)}]",
        "./biopb-mcp[mcp]",
        "./biopb-admin",
        "napari[all]",
    ]


# Extra constraints install.sh applies for specific platforms. Mirror them here so
# a green run reflects the *installer's* resolve, not the unconstrained graph.
INSTALLER_CONSTRAINTS = {
    # cryptography >= 49 ships arm64-only macOS wheels; install.sh caps it on Intel
    # macOS so uv picks 48.x's universal2 wheel. See install.sh + issue #355.
    "x86_64-apple-darwin": ["cryptography<49"],
}


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
    reqs = installer_requirements(target) + INSTALLER_CONSTRAINTS.get(target, [])
    proc = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--python-version",
            python_version,
            "--python-platform",
            target,
            "-",
        ],
        input="\n".join(reqs) + "\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"uv pip compile failed for {target}")
    pinned: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = _PKG_LINE.match(line.strip())
        if m:
            pinned[m.group(1).lower()] = m.group(2)
    return pinned


def wheel_platform_tags(name: str, version: str) -> dict | None:
    """Fetch the wheel platform tags for ``name==version`` from PyPI.

    Returns {"has_any": bool, "tags": [platform-tag, ...]}, or None if the package
    is not on PyPI (a local/unpublished package -> nothing to check).
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
    for f in data.get("urls", []):
        fn = f.get("filename", "")
        if not fn.endswith(".whl"):
            continue
        # wheel = name-version[-build]-python-abi-platform.whl; the platform field
        # (last) may be a '.'-joined set of tags.
        platform_field = fn[:-4].rsplit("-", 1)[-1]
        for tag in platform_field.split("."):
            if tag == "any":
                has_any = True
            tags.append(tag)
    return {"has_any": has_any, "tags": tags}


def missing_wheel(name: str, version: str, compatible) -> bool:
    info = wheel_platform_tags(name, version)
    if info is None:  # local / not on PyPI
        return False
    if not info["tags"]:  # sdist-only pure Python -> builds anywhere
        return False
    if info["has_any"]:  # py3-none-any wheel -> installable everywhere
        return False
    return not any(compatible(tag) for tag in info["tags"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--python-version", default="3.12")
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
                    lambda kv: missing_wheel(kv[0], kv[1], compatible), pinned.items()
                ),
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
            "on this platform. Pin/patch it in install.sh (see the "
            "cryptography<49 precedent) or upstream a wheel."
        )
        sys.exit(1)

    print(f"✓ every resolved package that ships wheels has one for {args.target}.")


if __name__ == "__main__":
    main()
