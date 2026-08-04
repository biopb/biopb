"""`skeleton-network-metrics` as benchmark data: how long is this network?

The skill takes a segmented filament network and returns its length, its branch
count and its junction count. Two facts are withheld, and they fail in different
directions:

* **the voxel spacing** — 0.1 um laterally and 0.5 um between planes, the
  ordinary confocal compromise. Nothing in a boolean mask says how thick a plane
  is, and a run that assumes cubic voxels reports **79.8%** of the true length;
* **what a real side branch is** — the operator's filaments never branch shorter
  than about two microns, and everything below that is roughness on the mask.
  Nothing in the pixels separates a two-voxel bump from a real bud, so a run that
  counts what the skeleton hands it reports **23 branches against 11**.

The third thing this case measures needs no asking, which is why the skill is
worth its length: **the distance between two adjacent skeleton voxels is not the
length of that filament**. Summing steps along a digitised path overestimates,
and on anisotropic voxels it overestimates badly, because a single z step buys
0.5 um of measured length for a rise the filament made over several microns.
That is `skan`'s `branch_distance` column, so it is also what a run that reaches
for the obvious library lands on.

Measured on this fixture, against a truth of 132.47 um, 11 branches and 5
junctions:

  =========================================  ==========  ========  =========
  route                                      length err  branches  junctions
  =========================================  ==========  ========  =========
  the whole procedure                              4.2%        11          5
  the same, without spur pruning                   2.6%        23         10
  the same, without junction clustering            3.7%        14          8
  step-summed length (`branch_distance`)          27.2%        11          5
  spacing ignored, cubic 0.1 um voxels            20.2%        11          5
  no graph at all, skeleton voxel count           23.6%        25         29
  =========================================  ==========  ========  =========

`TOLERANCE` sits in the gaps. The three knobs the procedure carries all have
plateaux far wider than the gap they have to clear: the pruning length gives the
same answer anywhere from 1 to 5 um, the junction-merge distance from 0.5 to
3 um, and the chord step from 0.8 to 3 um (97.3% to 106.2% of truth), so the
case scores whether the *decisions* were made, not whether a constant was
guessed.

The network is rejection-sampled so that no two non-incident filaments pass
within 2 um of each other and no two meeting at a node do so at less than 50
degrees. A crossing would be a junction the mask has and the generating graph
does not, and the truth would silently be wrong — which is how the first draft
of this fixture scored its own reference at 122%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .._benchmark import Case, Layer
from .._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_scalar,
    save_png,
)
from .._respondent import Persona

SKILL = "skeleton-network-metrics"
CASE_ID = "mitochondria-on-anisotropic-voxels"

#: From the table above, not from taste. The reference scores 4.2% / 0 / 0 and
#: the nearest wrong route scores 20.2% / 3 / 3, so each limit sits about three
#: times above a clean run and comfortably under every ablation. The two counts
#: are exact requirements written as half a branch, because counts are integers
#: and the engine's limits are strictly positive.
TOLERANCE = {
    "length_error_pct": 12.0,
    "branch_error": 1.5,
    "junction_error": 1.5,
}

SPACING = (0.5, 0.1, 0.1)  #: um per voxel, z:xy = 5:1
SHAPE = (48, 320, 320)  #: 24 x 32 x 32 um
RADIUS_UM = 0.5  #: filament radius; 1 um across, like a mitochondrion
#: Between non-incident filaments, and at a shared node. Both exist to keep the
#: mask's topology equal to the generating graph's — see the module docstring.
CLEARANCE_UM = 2.0
MIN_ANGLE_DEG = 50.0
N_NODES = 13
N_CHORDS = 3  #: chords beyond the tree, so the network has loops to get wrong
N_BUMPS = 60  #: lumps on the surface: the ragged mask a real threshold gives
GRAPH_SEED = 3
BUMP_SEED = 5


def _segment_distance(p0, p1, q0, q1) -> float:
    """Closest approach of two line segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-9:
        sc, tc = 0.0, (e / c if c > 1e-9 else 0.0)
    else:
        tc = float(np.clip((a * e - b * d) / den, 0, 1))
        sc = float(np.clip((b * tc - d) / a, 0, 1))
    return float(np.linalg.norm(w + sc * u - tc * v))


@dataclass(frozen=True)
class MitochondrialNetwork:
    """A filament network whose length only the microscope settings fix."""

    shape: tuple[int, int, int] = SHAPE
    spacing: tuple[float, float, float] = SPACING

    # --- the network itself, in microns ------------------------------------

    def _graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        rng = np.random.default_rng(GRAPH_SEED)
        extent = np.array(
            [n * s for n, s in zip(self.shape, self.spacing, strict=True)]
        )
        margin = np.array([3.0, 3.5, 3.5])
        nodes: list[np.ndarray] = []
        while len(nodes) < N_NODES:
            p = rng.uniform(margin, extent - margin)
            if all(np.linalg.norm(p - q) > 7.0 for q in nodes):
                nodes.append(p)
        pts = np.array(nodes)
        edges: list[tuple[int, int]] = []
        min_angle = np.deg2rad(MIN_ANGLE_DEG)

        def clear(a: int, b: int) -> bool:
            """Would this edge leave the mask's topology equal to the graph's?"""
            if a == b or (a, b) in edges or (b, a) in edges:
                return False
            if np.linalg.norm(pts[a] - pts[b]) > 16.0:
                return False
            for c, e in edges:
                shared = {a, b} & {c, e}
                if len(shared) == 2:
                    return False
                if len(shared) == 1:
                    # They already touch. What matters is that they part
                    # company: two tubes leaving a node at a shallow angle stay
                    # merged for microns, and the skeleton cuts the corner.
                    s = shared.pop()
                    u = pts[(a + b) - s] - pts[s]
                    v = pts[(c + e) - s] - pts[s]
                    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                    if np.arccos(np.clip(cos, -1, 1)) < min_angle:
                        return False
                    continue
                if _segment_distance(pts[a], pts[b], pts[c], pts[e]) < CLEARANCE_UM:
                    return False
            return True

        joined, rest = [0], list(range(1, N_NODES))
        while rest:
            nearest = sorted(
                (float(np.linalg.norm(pts[a] - pts[b])), a, b)
                for a in joined
                for b in rest
            )
            for _, a, b in nearest:
                if clear(a, b):
                    edges.append((a, b))
                    joined.append(b)
                    rest.remove(b)
                    break
            else:  # pragma: no cover - the seed is pinned and this one closes
                raise RuntimeError(f"no uncrowded edge reaches {rest}")

        chords = 0
        for a in range(N_NODES):
            for b in range(a + 1, N_NODES):
                if chords < N_CHORDS and clear(a, b):
                    edges.append((a, b))
                    chords += 1
        if chords < N_CHORDS:  # pragma: no cover - same
            raise RuntimeError(f"only {chords} of {N_CHORDS} loops fit")
        return pts, edges

    def _truth(self, pts, edges) -> dict[str, float | int]:
        """What the network *is*, in the terms the skill reports.

        A node of degree 2 is a bend in one filament, not a branch point, so it
        does not divide a branch — which is why `n_branches` is the edge count
        less those, and not the edge count.
        """
        degree = np.zeros(len(pts), int)
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        return {
            "total_length_um": sum(
                float(np.linalg.norm(pts[a] - pts[b])) for a, b in edges
            ),
            "n_branches": len(edges) - int((degree == 2).sum()),
            "n_junctions": int((degree >= 3).sum()),
            "n_endpoints": int((degree == 1).sum()),
            "n_loops": len(edges) - len(pts) + 1,
        }

    # --- and as voxels -----------------------------------------------------

    def _positions(self) -> np.ndarray:
        """Physical coordinate of every voxel centre, in microns."""
        zz, yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1], 0 : self.shape[2]]
        return np.stack(
            [zz * self.spacing[0], yy * self.spacing[1], xx * self.spacing[2]],
            axis=-1,
        ).astype(np.float32)

    def _rasterize(self, pts, edges) -> np.ndarray:
        where = self._positions()
        mask = np.zeros(self.shape, bool)
        for a, b in edges:
            p0 = pts[a].astype(np.float32)
            p1 = pts[b].astype(np.float32)
            along = p1 - p0
            length = float(np.linalg.norm(along))
            unit = along / length
            offset = where - p0
            t = np.clip(offset @ unit, 0, length)
            mask |= np.linalg.norm(offset - t[..., None] * unit, axis=-1) <= RADIUS_UM

        # Lumps stuck to the surface. A real threshold gives a ragged mask, and
        # every lump is a spur on the skeleton -- the population step 5 prunes.
        # Rasterized in a local box: a full-volume distance per lump is 60x the
        # cost of the network itself.
        rng = np.random.default_rng(BUMP_SEED)
        surface = np.argwhere(mask)
        span = np.array([int(np.ceil(1.2 / s)) for s in self.spacing])
        for _ in range(N_BUMPS):
            centre = surface[rng.integers(len(surface))]
            lo = np.maximum(centre - span, 0)
            hi = np.minimum(centre + span + 1, self.shape)
            box = where[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            here = (centre * np.array(self.spacing, np.float32)).astype(np.float32)
            mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] |= np.linalg.norm(
                box - here, axis=-1
            ) <= rng.uniform(0.6, 1.0)
        return mask

    def __call__(self) -> Fixture:
        pts, edges = self._graph()
        truth = self._truth(pts, edges)
        mask = self._rasterize(pts, edges)

        # The stack the mask came from: blurred by a PSF that is wider between
        # planes than within one, so the picture agrees with the voxel size the
        # operator will quote. It is here to be looked at, not measured.
        rng = np.random.default_rng(11)
        signal = ndi.gaussian_filter(mask.astype(np.float32), sigma=(0.6, 1.1, 1.1))
        image = 120.0 + 900.0 * signal
        image = image + rng.normal(0, np.sqrt(np.maximum(image, 1.0)) * 0.5)
        image = (image + rng.normal(0, 4.0, self.shape)).astype(np.float32)

        # The two properties the case rests on, checked before anyone pays for a
        # run. Neither is visible from the arrays alone, which is the point.
        assert truth["n_junctions"] >= 4, "too few junctions to score clustering"
        assert truth["n_loops"] >= 2, "a network with no loops is a tree"
        components = ndi.label(mask, structure=np.ones((3, 3, 3)))[1]
        assert components == 1, (
            f"the mask is in {components} pieces, so a run could report the "
            "length of one of them and the truth would not say so"
        )
        assert self.spacing[0] / self.spacing[2] >= 4, (
            "the z step is not coarse enough for cubic voxels to be a wrong "
            "answer, and that assumption is what the case is built to catch"
        )

        return Fixture(
            provenance=(
                f"procedural: {N_NODES} nodes and {len(edges)} filaments of radius "
                f"{RADIUS_UM} um in {self.shape[0]}x{self.shape[1]}x{self.shape[2]} "
                f"voxels at {self.spacing} um, graph seed {GRAPH_SEED}, "
                f"{N_BUMPS} surface lumps at seed {BUMP_SEED}"
            ),
            about=(
                f"A mitochondrial network of {truth['total_length_um']:.1f} um in "
                f"{truth['n_branches']} branches meeting at {truth['n_junctions']} "
                f"junctions, with {truth['n_loops']} loops and "
                f"{truth['n_endpoints']} free ends. The stack is sampled at "
                f"{self.spacing[1]} um laterally and {self.spacing[0]} um between "
                "planes, and nothing in a boolean mask says so: measured as cubic "
                "voxels the same network is 79.8% as long. Its mask is ragged, as "
                "a thresholded one is, and the lumps become side branches two "
                "voxels long that no filter in the pixels tells from real ones — "
                "counted, they take 11 branches to 23."
            ),
            data={"mitochondria": image, "mask": mask.astype(np.uint8)},
            truth={
                # The private facts: the geometry the voxels were drawn from,
                # and the spacing that turns voxels back into microns.
                "total_length_um": truth["total_length_um"],
                "n_branches": truth["n_branches"],
                "n_junctions": truth["n_junctions"],
                "n_endpoints": truth["n_endpoints"],
                "n_loops": truth["n_loops"],
                "spacing_um": np.array(self.spacing, float),
                "nodes_um": pts,
                "edges": np.array(edges, int),
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Every metric reads a scalar the truth carries by name, which is what an
    annotated real network carries too — someone tracing a stack by hand
    produces a length and two counts and nothing else — so the case survives a
    curated substitution without changing what is measured.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    want_length = fixture.truth.get("total_length_um")
    if want_length is None:
        why = "the fixture carries no network length"
        metrics.append(
            Metric(
                "length_error_pct", None, limits["length_error_pct"], unavailable=why
            )
        )
    else:
        got, why = read_scalar(attempt, "total_length_um")
        if got is None:
            metrics.append(
                Metric(
                    "length_error_pct",
                    None,
                    limits["length_error_pct"],
                    unavailable=why,
                )
            )
        else:
            want = float(want_length)
            metrics.append(
                Metric(
                    "length_error_pct",
                    100.0 * abs(got - want) / max(abs(want), 1e-12),
                    limits["length_error_pct"],
                    unit="%",
                )
            )
            detail["length_reported_um"] = round(got, 2)
            detail["length_true_um"] = round(want, 2)

    for key, limit_name in (
        ("n_branches", "branch_error"),
        ("n_junctions", "junction_error"),
    ):
        want_count = fixture.truth.get(key)
        if want_count is None:
            metrics.append(
                Metric(
                    limit_name,
                    None,
                    limits[limit_name],
                    unavailable=f"the fixture carries no {key}",
                )
            )
            continue
        got, why = read_scalar(attempt, key)
        if got is None:
            metrics.append(
                Metric(limit_name, None, limits[limit_name], unavailable=why)
            )
            continue
        metrics.append(
            Metric(
                limit_name,
                abs(got - float(want_count)),
                limits[limit_name],
                unit=" off",
            )
        )
        detail[f"{key}_reported"] = int(round(got))
        detail[f"{key}_true"] = int(want_count)

    for extra in ("n_loops", "n_endpoints"):
        # Not scored, and reported anyway: a run whose length and counts are
        # right but whose loop count is not has built a different graph, and
        # that is worth seeing in the report rather than discovering later.
        got, _why = read_scalar(attempt, extra)
        if got is not None:
            detail[f"{extra}_reported"] = int(round(got))
            detail[f"{extra}_true"] = int(fixture.truth.get(extra, -1))

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    image = fixture.data.get("mitochondria")
    mask = fixture.data.get("mask")
    if image is None or mask is None:
        return
    image = np.asarray(image, float)
    mask = np.asarray(mask, float)

    # Through z, and through x: the second is the view that shows how few
    # planes the network actually spans, which is the whole difficulty.
    save_png(image.max(axis=0), where / "network-xy.png")
    save_png(mask.max(axis=0), where / "mask-xy.png")
    save_png(np.repeat(mask.max(axis=2), 5, axis=0), where / "mask-zy-scaled.png")

    rows = ["quantity,truth,reported"]
    for key in (
        "total_length_um",
        "n_branches",
        "n_junctions",
        "n_endpoints",
        "n_loops",
    ):
        want = fixture.truth.get(key)
        got = attempt.arrays.get(key)
        got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
        rows.append(f"{key},{'' if want is None else f'{float(want):g}'},{got}")
    (where / "network.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A confocal stack of a cell's mitochondria is open in the napari viewer:

  mitochondria  float32 (48, 320, 320) — the raw stack, one channel
  mask          uint8   (48, 320, 320) — that network already segmented, 1 inside

There is no tensor server in this session, so those layers are the only source
of pixels. The segmentation is done; do not re-segment it, and do not clean it
up beyond what your measurement needs.

I need the size and the shape of this network: how much of it there is, and how
branched it is.

The person who acquired and segmented this is here and can answer questions. To
ask them something, reply in plain text without calling a tool, and their answer
will come back as the next message.

When you are finished, leave three names bound in the kernel namespace:

  total_length_um   the total length of the network, in microns
  n_branches        how many branches it is made of
  n_junctions       how many points those branches meet at

Those names are how your result is collected, so bind them exactly.
"""

#: Note what is *not* here: nothing about skeletons, graphs, pruning or how a
#: length is accumulated. This person knows their microscope and their
#: mitochondria. A persona that knew the procedure could answer a question the
#: agent never properly asked, and the numeric result would stop meaning what it
#: appears to.
CELL_BIOLOGIST = Persona(
    name="operator-mitochondrial-network",
    facts={
        "what one voxel is": (
            "0.1 microns in x and y. The z step was 0.5 microns — I did not "
            "oversample in z, the stacks are big enough already"
        ),
        "how thick a mitochondrion is": (
            "about a micron across. They are all much the same thickness, which "
            "is why I can threshold them at all"
        ),
        "what the shortest real side branch is": (
            "two microns, near enough. Nothing real is shorter than that — if it "
            "were, it would be a new tubule and I would see it move"
        ),
        "what the small lumps on the outline are": (
            "that is the threshold, not the biology. The outline is ragged "
            "everywhere and I have stopped trying to fix it"
        ),
        "whether the stack has been changed since it was acquired": (
            "no. That is the raw stack and a threshold on it, nothing else — no "
            "deconvolution and no resampling"
        ),
        "what the numbers are for": (
            "comparing treated cells against untreated ones. The stacks were not "
            "all taken the same way, so the numbers have to be in microns to be "
            "worth anything"
        ),
        "how many pieces the network is in": (
            "one, in this cell. In the treated cells it falls apart into "
            "fragments, and that is exactly what I am trying to measure"
        ),
    },
    background=(
        "You imaged mitochondria in a cultured cell on a confocal microscope and "
        "thresholded the stack yourself. You are happy to answer questions about "
        "the sample, the microscope and the segmentation."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id=CASE_ID,
    task=TASK,
    persona=CELL_BIOLOGIST,
    fixture=Procedural(MitochondrialNetwork()),
    layers=(
        Layer("mitochondria", "mitochondria"),
        Layer("mask", "mask", kind="labels"),
    ),
    collect={
        "total_length_um": "total_length_um",
        "n_branches": "n_branches",
        "n_junctions": "n_junctions",
    },
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="skeleton",
    # It must be able to answer: the fixture withholds the voxel size and the
    # length below which a branch is not biology, and this person knows both.
    persona_must_know=("0.1 microns", "0.5 microns", "two microns", "ragged"),
    # And it must not know the method — only the sample and the microscope.
    persona_must_not_know=(
        "skeleton",
        "skan",
        "prune",
        "spur",
        "medial axis",
        "branch_distance",
        "chord",
        "degree",
        "connected component",
    ),
)
