#!/usr/bin/env python3
"""Build the biopb sample-image bundle shipped with a release.

The installer seeds a fresh (config-less) install with this bundle so the product
opens onto real, populated microscopy data instead of an empty catalog — the
drag-drop / admin-page paths are how a user then adds their own data (see
docs/release-model.md). The bundle is produced ONCE here, in CI, so its bytes are
fixed and checksummed into the release; it is never fetched at install time.

Every image is drawn from ``skimage.data`` and is confirmed **CC0 / public
domain** (see the ``LICENSE`` column below and the emitted ``LICENSES.txt``). We
deliberately do NOT reuse the Fiji/ImageJ "Open Samples" set: its redistribution
status is an unresolved legal gray area (per the ImageJ maintainers themselves),
whereas skimage's data is individually license-vetted.

Each image is written as an **OME-TIFF** with an explicit ``axes`` string (and
physical pixel sizes where documented), so the tensor server's OME-TIFF adapter
surfaces it with correct Z/C/T/Y/X semantics — the PNG/JPG originals would
otherwise be blocked by discovery's generic-image guard (biopb/biopb#40) and
carry no axis metadata. Re-encoding with deflate keeps the files close to their
source size while making the whole set uniform.

Usage:
    python tools/build_samples.py --out samples/            # default lite set
    python tools/build_samples.py --out samples/ --include-large
    python tools/build_samples.py --out samples/ --only cells3d,cell   # dev

Run it in an isolated environment so it never perturbs the workspace, e.g.:
    uv run --no-project --with 'scikit-image==0.26.0' --with pooch --with tifffile \
        python tools/build_samples.py --out samples/
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """One curated sample image and everything needed to ship it."""

    name: str  # skimage.data loader name == output basename
    axes: str  # OME axis order the file is written in
    origin: str  # provenance / attribution for LICENSES.txt
    license: str = "CC0"  # every curated entry is CC0 / public domain
    large: bool = False  # excluded from the default (lite) bundle
    # The axis order skimage returns, when it differs from the OME order we want
    # to store. OME-TIFF keeps channel/Z/T as leading *planes*; a loader that
    # returns an interleaved trailing channel (e.g. lily is Y,X,C) must be
    # transposed to plane-leading (C,Y,X) or tifffile mis-reads it as RGB.
    # Leave empty when the loaded order already equals ``axes``.
    source_axes: str = ""
    # Documented physical pixel sizes (micrometers), written into the OME-XML
    # so napari renders a correct scale bar. Keys among X/Y/Z; omit when unknown.
    physical_um: dict[str, float] = field(default_factory=dict)


# The curated set. Kept small and microscopy-focused on purpose; the point is a
# credible first-run demo, not a benchmark corpus. Order = manifest order.
SAMPLES: list[Sample] = [
    # --- multidimensional microscopy (the large-data story) ---
    Sample(
        "cells3d",
        "ZCYX",
        "Allen Institute for Cell Science — 3D fluorescence (membrane + nuclei).",
        physical_um={"Z": 0.29, "Y": 0.26, "X": 0.26},
    ),
    Sample(
        "kidney",
        "ZYXS",
        "Genevieve Buckley, Monash Micro Imaging (2018) — confocal mouse kidney, 3-channel.",
    ),
    Sample(
        "lily",
        "CYX",
        "Genevieve Buckley, Monash Micro Imaging (2018) — 4-channel confocal, lily stem.",
        physical_um={"Y": 1.24, "X": 1.24},
        source_axes="YXC",
    ),
    Sample(
        "nickel_solidification",
        "TYX",
        "C. Gus Becker et al. (CSM/UCSB), APS beamline 32-ID-B, ANL (2019) — synchrotron x-radiograph time series.",
    ),
    Sample(
        "protein_transport",
        "TCYX",
        "Andrea Boni & Jan Ellenberg (EMBL) — NPC single-nucleus time series (Boni et al. JCB 2015).",
    ),
    # --- small 2D bio (detection / quick demos) ---
    Sample("human_mitosis", "YX", "David Root — dense nuclei, fluorescence."),
    Sample("cell", "YX", "Public domain (CC0) — single cell, transmitted light."),
    Sample("microaneurysms", "YX", "Andreas Maier — retinal fundus detail (CC0)."),
    Sample(
        "immunohistochemistry",
        "YXS",
        "No known copyright restrictions — IHC-stained tissue (RGB).",
    ),
    Sample("retina", "YXS", "Public domain (CC0), ISSN 2002-4436 — fundus (RGB)."),
    # --- large; opt-in only (~250 MB decoded) ---
    Sample(
        "palisades_of_vogt",
        "TYX",
        "Viacheslav Mazlin / Jules Scholler — in-vivo cornea time series (CC0).",
        large=True,
    ),
]


def _write_ome_tiff(sample: Sample, array, out_path: Path) -> None:
    """Write ``array`` as a compressed OME-TIFF carrying ``sample``'s axes."""
    import tifffile

    metadata: dict[str, object] = {"axes": sample.axes}
    for axis, size in sample.physical_um.items():
        metadata[f"PhysicalSize{axis}"] = size
        metadata[f"PhysicalSize{axis}Unit"] = "µm"

    # 'S' in the axis string means interleaved RGB samples (photometric rgb);
    # everything else is single-channel grayscale planes.
    photometric = "rgb" if "S" in sample.axes else "minisblack"
    tifffile.imwrite(
        out_path,
        array,
        ome=True,
        photometric=photometric,
        compression="zlib",
        metadata=metadata,
    )


def build(out_dir: Path, include_large: bool, only: set[str] | None) -> list[dict]:
    import numpy as np
    import skimage.data as skd

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for sample in SAMPLES:
        if only is not None and sample.name not in only:
            continue
        if (
            sample.large
            and not include_large
            and (only is None or sample.name not in only)
        ):
            print(f"  skip {sample.name} (large; use --include-large)")
            continue

        loader = getattr(skd, sample.name)
        array = np.asarray(loader())  # triggers the pooch download if needed
        if array.ndim != len(sample.axes):
            raise SystemExit(
                f"axes mismatch for {sample.name}: array ndim {array.ndim} "
                f"!= len(axes '{sample.axes}')"
            )
        if sample.source_axes:
            # Reorder the loader's native axes into the OME storage order.
            perm = [sample.source_axes.index(axis) for axis in sample.axes]
            array = np.transpose(array, perm)

        out_path = out_dir / f"{sample.name}.ome.tif"
        _write_ome_tiff(sample, array, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(
            f"  wrote {out_path.name:28} {str(array.dtype):8} "
            f"{sample.axes:6} {str(array.shape):22} {size_mb:7.2f} MB"
        )
        rows.append(
            {
                "file": out_path.name,
                "source": f"skimage.data.{sample.name}",
                "license": sample.license,
                "axes": sample.axes,
                "shape": "x".join(map(str, array.shape)),
                "dtype": str(array.dtype),
                "origin": sample.origin,
            }
        )

    if not rows:
        raise SystemExit("no samples selected — check --only")
    return rows


def write_manifest(out_dir: Path, rows: list[dict]) -> None:
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "source",
                "license",
                "axes",
                "shape",
                "dtype",
                "origin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_licenses(out_dir: Path, rows: list[dict]) -> None:
    lines = [
        "biopb sample images — licensing",
        "=" * 34,
        "",
        "Every image in this bundle is drawn from scikit-image's sample data",
        "(https://gitlab.com/scikit-image/data) and is released under the Creative",
        "Commons CC0 1.0 Universal Public Domain Dedication",
        "(https://creativecommons.org/publicdomain/zero/1.0/), i.e. free to copy,",
        "modify, and redistribute for any purpose. Each was re-encoded to OME-TIFF",
        "for biopb; the pixel data is otherwise unmodified.",
        "",
        "Per-image origin / attribution:",
        "",
    ]
    for row in rows:
        lines.append(f"* {row['file']}  [{row['license']}]")
        lines.append(f"    from {row['source']}")
        lines.append(f"    {row['origin']}")
        lines.append("")
    (out_dir / "LICENSES.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="also emit large opt-in samples (palisades_of_vogt, ~250 MB)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of loader names to build (dev/testing)",
    )
    args = parser.parse_args(argv)

    only = None
    if args.only:
        only = {name.strip() for name in args.only.split(",") if name.strip()}

    rows = build(args.out, args.include_large, only)
    write_manifest(args.out, rows)
    write_licenses(args.out, rows)
    total_mb = sum((args.out / r["file"]).stat().st_size for r in rows) / 1e6
    print(f"\n{len(rows)} images, {total_mb:.1f} MB total → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
