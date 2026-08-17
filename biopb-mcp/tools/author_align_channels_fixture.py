"""Author the `align-channels-from-landmarks` fixture into a curated tree.

Run **once**, by hand, and commit nothing but this file: the pixels it writes
are the fixture, and the tree is synced rather than generated per run. That is
the whole point of authoring-time perturbation — a benchmark that re-derives its
own data every run has no way to notice that the data changed, and a warp
applied at run time is a knob someone can turn between two results that get
compared.

What it makes, from one real acquisition:

* ``moving`` — a real nuclear-stain channel, unmodified.
* ``fixed``  — a real cytoplasmic-stain channel of the *same field*, resampled
  once through a known geometric map. Real pixels, real cross-modality
  intensity relationship (the two channels correlate 0.162, so this is not one
  image twice), and an exactly-known correspondence.
* ``moving_pts`` / ``fixed_pts`` — 18 correspondences at **real nuclei
  centroids**, spread by farthest-point sampling and displaced by 1.5 px of
  click noise, which is what a person clicking in BigWarp actually produces.
* ``probe_pts`` + the withheld ``probe_truth`` — 400 points the run must map.

The map is affine (6.5 deg rotation, anisotropic scale, shear, translation)
plus a smooth non-affine term, median displacement ~52 px. **Neither an affine
nor a thin-plate spline through the given points represents it exactly**, so no
route to the answer is an oracle and every one of them approximates. The
amplitude is inherited unchanged from the 2026-08-05 landmark-registration
prescreen (``docs/skill-candidates.md``) so the numbers stay comparable; it was
chosen there to separate method families, which makes it harder than a typical
section-to-section warp rather than representative of one.

Usage::

    python biopb-mcp/tools/author_align_channels_fixture.py \\
        --tensor-url grpc://localhost:8815 \\
        --source-id aics_ed3e82cd9a06 \\
        --out ~/biopb-fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAMESPACE = "tasks"
CASE_ID = "align-channels-from-landmarks"

SIZE = 960
N_LANDMARKS = 18
N_PROBES = 400
CLICK_SIGMA = 1.5
SEED = 11

#: Channel indices in the source acquisition: 3 is the nuclear stain (clean,
#: point-like, ~124 detectable nuclei), 0 the cytoplasmic one.
MOVING_CHANNEL = 3
FIXED_CHANNEL = 0

CENTER = np.array([SIZE / 2.0, SIZE / 2.0])
_THETA = np.deg2rad(6.5)
_ROT = np.array([[np.cos(_THETA), -np.sin(_THETA)], [np.sin(_THETA), np.cos(_THETA)]])
_SCALE_SHEAR = np.array([[1.045, 0.030], [0.0, 0.975]])
LINEAR = _ROT @ _SCALE_SHEAR
OFFSET = np.array([18.0, -12.0])


def nonlinear(pts: np.ndarray) -> np.ndarray:
    """The part no affine and no TPS-through-these-points can represent."""
    u, v = pts[:, 0] / SIZE, pts[:, 1] / SIZE
    d_r = 26.0 * np.sin(2 * np.pi * (0.9 * v + 0.25)) + 12.0 * np.cos(
        2 * np.pi * (1.1 * u)
    )
    d_c = 23.0 * np.cos(2 * np.pi * (0.8 * u + 0.40)) + 10.0 * np.sin(
        2 * np.pi * (1.2 * v)
    )
    return np.stack([d_r, d_c], axis=1)


def forward(pts) -> np.ndarray:
    """moving coordinates -> fixed coordinates."""
    pts = np.atleast_2d(np.asarray(pts, float))
    return (pts - CENTER) @ LINEAR.T + CENTER + OFFSET + nonlinear(pts)


def inverse(pts, iters: int = 60) -> np.ndarray:
    """Fixed-point iteration; the non-affine term is small and smooth, so it
    contracts. Verified against `forward` before anything is written."""
    pts = np.atleast_2d(np.asarray(pts, float))
    back = np.linalg.inv(LINEAR)
    x = (pts - CENTER - OFFSET) @ back.T + CENTER
    for _ in range(iters):
        x = (pts - nonlinear(x) - CENTER - OFFSET) @ back.T + CENTER
    return x


def _farthest_point(candidates: np.ndarray, n: int) -> np.ndarray:
    """Spread the clicks, because a clustered set is a different experiment.

    The prescreen ran both: 6 clustered clicks leave 94% of the probes outside
    their convex hull, which measures extrapolation rather than registration.
    This case ships the spread budget only.
    """
    chosen = [int(np.argmax(np.linalg.norm(candidates - CENTER, axis=1)))]
    for _ in range(n - 1):
        far = np.min(
            np.linalg.norm(candidates[:, None] - candidates[chosen][None], axis=-1),
            axis=1,
        )
        chosen.append(int(np.argmax(far)))
    return candidates[chosen]


def build(volume: np.ndarray) -> tuple[dict, dict, dict]:
    """`volume` is (C, Z, Y, X) from the acquisition. Returns data, truth, facts."""
    from skimage.feature import blob_log
    from skimage.transform import warp

    mip = volume.max(axis=1).astype(np.float32)
    top = (mip.shape[1] - SIZE) // 2, (mip.shape[2] - SIZE) // 2
    crop = mip[:, top[0] : top[0] + SIZE, top[1] : top[1] + SIZE]
    moving = np.ascontiguousarray(crop[MOVING_CHANNEL])
    cyto = np.ascontiguousarray(crop[FIXED_CHANNEL])

    err = np.abs(
        inverse(forward(np.array([[10.0, 10.0], [900.0, 700.0]])))
        - np.array([[10.0, 10.0], [900.0, 700.0]])
    ).max()
    if err > 1e-6:
        raise SystemExit(f"inverse map disagrees with forward by {err:.2e} px")

    fixed = warp(
        cyto, lambda xy: inverse(xy[:, ::-1])[:, ::-1], order=1, preserve_range=True
    ).astype(np.float32)

    unit = (moving - moving.min()) / (moving.max() - moving.min())
    blobs = blob_log(unit, min_sigma=6, max_sigma=14, num_sigma=5, threshold=0.055)
    centroids = blobs[:, :2]
    inside = ((centroids > 40).all(axis=1)) & ((centroids < SIZE - 40).all(axis=1))
    centroids = centroids[inside]
    if len(centroids) < N_LANDMARKS * 2:
        raise SystemExit(f"only {len(centroids)} nuclei found; need room to spread")

    rng = np.random.default_rng(SEED)
    truth_pts = _farthest_point(centroids, N_LANDMARKS)
    moving_pts = truth_pts + rng.normal(0, CLICK_SIGMA, truth_pts.shape)
    fixed_pts = forward(truth_pts) + rng.normal(0, CLICK_SIGMA, truth_pts.shape)
    probe_pts = rng.uniform(40, SIZE - 40, size=(N_PROBES, 2))

    data = {
        "moving": moving,
        "fixed": fixed,
        "moving_pts": moving_pts,
        "fixed_pts": fixed_pts,
        "probe_pts": probe_pts,
    }
    truth = {"probe_truth": forward(probe_pts), "landmark_truth": truth_pts}
    disp = np.linalg.norm(truth["probe_truth"] - probe_pts, axis=1)
    facts = {
        "nuclei_detected": int(len(centroids)),
        "displacement_px": {
            "min": round(float(disp.min()), 2),
            "median": round(float(np.median(disp)), 2),
            "max": round(float(disp.max()), 2),
        },
        "channel_correlation": round(
            float(
                np.corrcoef(moving.ravel(), np.ascontiguousarray(cyto).ravel())[0, 1]
            ),
            3,
        ),
    }
    return data, truth, facts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays(mapping: dict) -> dict:
    return {
        k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
        for k, v in mapping.items()
    }


def write(root: Path, data: dict, truth: dict, facts: dict, source: dict) -> None:
    here = root / NAMESPACE / CASE_ID
    here.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(here / "data.npz", **data)
    np.savez_compressed(here / "truth.npz", **truth)
    (here / "case.json").write_text(
        json.dumps(
            {
                "about": (
                    "A real cross-modality confocal pair -- nuclear stain against "
                    "cytoplasmic stain of the same field -- with the second channel "
                    "resampled once through a known geometric map, and 18 "
                    "correspondences clicked at real nuclei centroids."
                ),
                "data": dict.fromkeys(data, "data.npz"),
                "truth": dict.fromkeys(truth, "truth.npz"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"fixtures": []}
    )
    entry = {
        "skill": NAMESPACE,
        "case_id": CASE_ID,
        "citation": source["citation"],
        "provenance": (
            f"Channels {MOVING_CHANNEL} (nuclear, unmodified) and {FIXED_CHANNEL} "
            f"(cytoplasmic, resampled) of {source['source_id']} "
            f"({source['source_url']}), max-projected over z and centre-cropped to "
            f"{SIZE}x{SIZE}. The fixed channel carries a known affine "
            "(6.5deg rotation, anisotropic scale 1.045/0.975, shear 0.030, "
            "translation [18,-12]) plus a smooth non-affine term, applied ONCE "
            "here at authoring time by biopb-mcp/tools/"
            f"{Path(__file__).name}; the run never perturbs anything. Neither an "
            "affine nor a TPS through the shipped correspondences represents that "
            "map exactly, so no route to the answer is an oracle. Amplitude is "
            "inherited from the 2026-08-05 landmark-registration prescreen "
            "(docs/skill-candidates.md), where it was chosen to separate method "
            "families -- harder than a typical section-to-section warp, not a "
            f"sample of one. Measured: {json.dumps(facts)}."
        ),
        "files": {
            "data.npz": {
                "sha256": _sha256(here / "data.npz"),
                "arrays": _arrays(data),
            },
            "truth.npz": {
                "sha256": _sha256(here / "truth.npz"),
                "arrays": _arrays(truth),
            },
        },
    }
    kept = [
        f
        for f in manifest.get("fixtures", [])
        if not (f.get("skill") == NAMESPACE and f.get("case_id") == CASE_ID)
    ]
    manifest["fixtures"] = [*kept, entry]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {here}")
    for k, v in facts.items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor-url", default="grpc://localhost:8815")
    ap.add_argument("--source-id", default="aics_ed3e82cd9a06")
    ap.add_argument("--array-id", default=None, help="defaults to <source-id>/Image:0")
    ap.add_argument("--citation", default="")
    ap.add_argument("--out", type=Path, default=Path("~/biopb-fixtures"))
    args = ap.parse_args()

    from biopb.tensor import TensorFlightClient

    client = TensorFlightClient(args.tensor_url)
    array_id = args.array_id or f"{args.source_id}/Image:0"
    volume = np.asarray(client.get_tensor(array_id))[0]

    url = ""
    try:
        rows = client.query_sources(
            f"SELECT source_url FROM sources WHERE source_id = '{args.source_id}'",
            format="pandas",
        )
        url = str(rows.iloc[0].source_url) if len(rows) else ""
    except Exception:  # noqa: BLE001 -- provenance is best-effort, the pixels are not
        pass

    citation = args.citation.strip()
    if not citation:
        raise SystemExit(
            "--citation is required. Real data comes from someone, and the "
            "manifest refuses an entry that does not say who."
        )
    data, truth, facts = build(volume)
    write(
        args.out.expanduser(),
        data,
        truth,
        facts,
        {"source_id": args.source_id, "source_url": url, "citation": citation},
    )


if __name__ == "__main__":
    main()
