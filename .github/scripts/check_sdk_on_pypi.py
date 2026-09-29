"""Check that a locally built biopb SDK wheel is the one PyPI serves.

The installer takes the SDK from PyPI, so a product release may only ship
alongside an SDK that is already published. Given the built wheel and the SDK
version it should match (the nearest ``v*`` tag), this waits for that version
to appear on PyPI, downloads its wheel, and compares the two: every package
file byte for byte (except the setuptools_scm version stamp), plus the
dependency and entry-point metadata. Exits 1 on any difference.

Usage: check_sdk_on_pypi.py WHEEL VERSION [--wait SECONDS]
"""

import argparse
import email.parser
import hashlib
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/biopb/{version}/json"
# Differs between any two builds: the version stamp setuptools_scm writes.
VOLATILE = {"biopb/_version.py"}
METADATA_FIELDS = ("Requires-Python", "Requires-Dist", "Provides-Extra")


def pypi_wheel_url(version: str, wait: float) -> str:
    """Return the wheel URL for ``version``, polling until PyPI lists it."""
    deadline = time.monotonic() + wait
    while True:
        try:
            with urllib.request.urlopen(PYPI_JSON.format(version=version)) as r:
                files = json.load(r)["urls"]
            for f in files:
                if f["packagetype"] == "bdist_wheel":
                    return f["url"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        if time.monotonic() >= deadline:
            sys.exit(
                f"biopb {version} has no wheel on PyPI after {wait:.0f}s. "
                f"Push the v{version} tag (python-ci publishes it) and re-run."
            )
        time.sleep(30)


def contents(wheel: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Package-file digests and the dependency-relevant metadata of a wheel."""
    files: dict[str, str] = {}
    meta: dict[str, list[str]] = {}
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            top = name.split("/", 1)[0]
            if top.endswith(".dist-info"):
                rel = name.split("/", 1)[1]
                if rel == "METADATA":
                    msg = email.parser.Parser().parsestr(z.read(name).decode())
                    for field in METADATA_FIELDS:
                        meta[field] = sorted(msg.get_all(field) or [])
                elif rel == "entry_points.txt":
                    meta["entry_points"] = z.read(name).decode().split()
            elif name not in VOLATILE and not name.endswith("/"):
                files[name] = hashlib.sha256(z.read(name)).hexdigest()
    return files, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("version")
    ap.add_argument("--wait", type=float, default=0, help="seconds to poll PyPI")
    args = ap.parse_args()

    url = pypi_wheel_url(args.version, args.wait)
    with tempfile.TemporaryDirectory() as tmp:
        published = Path(tmp) / url.rsplit("/", 1)[1]
        urllib.request.urlretrieve(url, published)
        built_files, built_meta = contents(args.wheel)
        pub_files, pub_meta = contents(published)

    diffs = [
        f"{name}: {'added' if name not in pub_files else 'removed' if name not in built_files else 'changed'}"
        for name in sorted(built_files.keys() | pub_files.keys())
        if built_files.get(name) != pub_files.get(name)
    ]
    diffs += [
        f"METADATA {key}: built {built_meta.get(key)} != PyPI {pub_meta.get(key)}"
        for key in sorted(built_meta.keys() | pub_meta.keys())
        if built_meta.get(key) != pub_meta.get(key)
    ]
    if diffs:
        print(f"The SDK at this commit differs from biopb {args.version} on PyPI:")
        print("\n".join(f"  {d}" for d in diffs))
        print(
            "Cut a new SDK tag (v*) on this commit, or release from a commit "
            "whose SDK matches a published one."
        )
        return 1
    print(f"SDK matches biopb {args.version} on PyPI ({len(built_files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
