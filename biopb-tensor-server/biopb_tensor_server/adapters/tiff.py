"""TIFF sequence adapters for tensor storage.

Handles plain TIFF file sequences and legacy MicroManager datasets.
OME-TIFF files are handled by OmeTiffAdapter (pure-tifffile).
"""

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._scale import (
    MICRON,
    mm_summary_scale,
    scale_by_label,
    unit_to_um,
)
from biopb_tensor_server.core.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    SourceClaim,
    _is_offline_placeholder,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

# TIFF ResolutionUnit code -> micrometres per unit (2 = inch, 3 = centimetre).
# Code 1 ("no absolute unit") carries only an aspect ratio and is excluded.
_RESUNIT_TO_UM = {2: 25400.0, 3: 10000.0}

# Sentinel for "physical scale not computed yet" -- distinct from a computed
# ``None`` (no usable calibration), so the memoized result is never recomputed.
_SCALE_UNSET = object()


def _tiff_pixel_size_um(page, tag_name: str, imagej_unit_um) -> Optional[float]:
    """One axis' pixel size in µm from a TIFF ``X``/``YResolution`` tag.

    A resolution tag is a *density* (pixels per unit), so the pixel size is its
    reciprocal times the unit's µm length. The unit is the ImageJ calibration
    unit when the file carries one (``imagej_unit_um`` = its µm factor), else the
    ``ResolutionUnit`` tag (inch / centimetre) -- but only when that tag is
    actually present. Returns ``None`` when the resolution tag is absent or zero,
    when the unit is unusable (e.g. ResolutionUnit 1, aspect-ratio only), and --
    deliberately -- when ``ResolutionUnit`` is *missing*: the TIFF spec defaults
    it to inch, but an omitted unit in practice means the density is an
    uncalibrated aspect ratio as often as it means inches, so assuming inch just
    fabricates a bogus micron size. Better no scale than a wrong one.
    """
    tag = page.tags.get(tag_name)
    if tag is None:
        return None
    val = tag.value
    if isinstance(val, tuple) and len(val) == 2:
        num, den = val
        if not num or not den:
            return None
        density = num / den  # pixels per unit
    else:
        try:
            density = float(val)
        except (TypeError, ValueError):
            return None
    if not density:
        return None
    per_pixel = 1.0 / density  # units per pixel, in the resolution's own unit

    if imagej_unit_um is not None:
        return per_pixel * imagej_unit_um
    ru_tag = page.tags.get("ResolutionUnit")
    if ru_tag is None:
        return None  # no explicit unit -- don't assume inch (see docstring)
    factor = _RESUNIT_TO_UM.get(int(ru_tag.value))
    if factor is None:
        return None
    return per_pixel * factor


# OME-TIFF naming patterns owned by the file-level OmeTiffAdapter. Excluded only
# OFF cloud (see ``exclude_ome`` below): OmeTiffAdapter claims individual .ome.tif
# FILES, not the directory, so if this directory-claiming adapter grouped a folder
# of .ome.tif into a plain sequence it would prune the subtree before any OME-XML /
# multi-scene parsing ran. Under a cloud root OmeTiffAdapter is disabled
# (``if ctx.cloud_root: return None``) and every .ome.tif degrades to a per-file
# aics source anyway, so deferring to it there is pointless -- grouping the
# directory into one sequence is the better fallback.
#
# MicroManager img_* patterns are deliberately NOT excluded. This adapter is
# registered *below* MicroManagerLegacyAdapter (see adapters/__init__.py), and
# registration order is load-bearing priority, so a valid MicroManager dataset is
# already claimed -- and its subtree pruned -- before this adapter is ever probed.
# The only img_* directories that reach here are ones MM declined (no metadata.txt
# or a corrupt/truncated one from an aborted acquisition). Grouping their frames
# into one sequence is exactly the wanted fallback, instead of letting the walk
# descend and register every frame as its own per-file aics source.
_TIFF_EXCLUDE_PATTERNS = {
    "*.ome.tif",
    "*.ome.tiff",
}


def _filter_tiff_candidates(files: List[Path], exclude_ome: bool = True) -> List[Path]:
    """Drop files matching OME-TIFF naming patterns owned by OmeTiffAdapter.

    ``exclude_ome`` is ``False`` under a cloud root, where OmeTiffAdapter is
    disabled and grouping .ome.tif into one sequence beats per-file fallback.
    """
    if not exclude_ome:
        return list(files)
    return [f for f in files if not any(f.match(p) for p in _TIFF_EXCLUDE_PATTERNS)]


# Minimum TIFFs for a directory to be *claimed* as a stacked sequence; below it
# each TIFF falls back to its own source. Set purposefully high to avoid
# false-positives: a wrong claim prunes the subtree and can leave the data
# unreadable, whereas a false-negative degrades gracefully to per-file sources.
# Claim-time only -- an explicitly-configured small sequence still opens at
# resolve (see _MIN_PATTERN_FILES).
_MIN_TIFF_FILES = 30


def _mask_and_digits(name: str) -> Tuple[str, List[int]]:
    """Split a filename into its structural mask and its numeric fields.

    Each run of digits is replaced by ``#`` to form a *mask* (so files that
    belong to the same sequence share a mask), and the integer value of each
    run is returned in order.

    ``"s1-0001_bf.tif"`` -> ``("s#-#_bf.tif", [1, 1])``
    """
    digits = [int(m.group()) for m in re.finditer(r"\d+", name)]
    mask = re.sub(r"\d+", "#", name)
    return mask, digits


def _natural_key(name: str) -> List[Tuple[int, Any]]:
    """Sort key ordering embedded numbers numerically (``img_2`` < ``img_10``).

    Splits ``name`` into alternating digit / non-digit chunks. Each chunk is
    wrapped as ``(0, int)`` or ``(1, str)`` so numbers sort before text at any
    position and int/str never compare directly (no ``TypeError`` on names of
    differing structure). Gives the stacked file axis a sensible *default* order;
    the authoritative interpretation of that axis is the agent's, via
    :meth:`TiffSequenceAdapter.get_metadata`.
    """
    return [
        (0, int(t)) if t.isdigit() else (1, t.lower())
        for t in re.findall(r"\d+|\D+", name)
    ]


# Filename-only coherence gate: does a directory hold a *coherent* set of related
# files, or an incidental grab-bag? It does not parse what the filename fields
# *mean* (the agent's job) -- only whether the set hangs together. Two signals,
# either sufficient:
#   (a) one digit-template (mask) shared by >=_COHERENT_FRACTION of the names --
#       catches numbered sequences, incl. tiny stems like ``a1/a2/a3``;
#   (b) a non-trivial common stem shared by >=_COHERENT_FRACTION -- catches sets
#       varying by a token, e.g. ``sp_0001_{red,green,blue}`` or MetaMorph
#       ``.._w1DIC_.. / .._w2GFP_..``.
# The threshold is a near-total super-majority because a real sequence is almost
# entirely one pattern; this stops a few strays from dragging an unrelated set in.
# A bare no-number/no-stem set (``red/green/blue.tif``) is indistinguishable from a
# grab-bag by filename alone, so it is left to per-file fallback.
_MIN_STEM = 3  # chars; a shorter shared prefix is too weak to imply coherence
_COHERENT_FRACTION = 0.9  # share of names that must fit one mask/stem to cohere

# Floor for the pattern check itself, distinct from the claim floor
# (_MIN_TIFF_FILES): under a few names a shared mask/stem is trivially met and
# means nothing. Governs the resolve-time gate, so an explicitly-configured small
# sequence can still open.
_MIN_PATTERN_FILES = 3


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def _looks_like_tiff_sequence(names: List[str]) -> bool:
    """Filename-only coherence gate (see the comment above). Pure, no I/O."""
    n = len(names)
    if n < _MIN_PATTERN_FILES:
        return False
    threshold = math.ceil(_COHERENT_FRACTION * n)

    # (a) a digit-template (mask) shared by >=90% of the names.
    mask_counts: Dict[str, int] = {}
    for nm in names:
        mask, _ = _mask_and_digits(nm)
        mask_counts[mask] = mask_counts.get(mask, 0) + 1
    if max(mask_counts.values()) >= threshold:
        return True

    # (b) a non-trivial common stem shared by >=90%. A prefix shared by that many
    # names is contiguous once the names are sorted, so the LCP of the first and
    # last entry of each threshold-sized window covers every candidate prefix.
    ordered = sorted(nm.lower() for nm in names)
    for i in range(n - threshold + 1):
        if _common_prefix_len(ordered[i], ordered[i + threshold - 1]) >= _MIN_STEM:
            return True
    return False


def _group_tiff_sequence(
    files: List[Path], exclude_ome: bool = True
) -> Optional[List[Path]]:
    """Group plain-TIFF files into one ordered sequence by a single varying field.

    .. note::
       Retained for unit coverage and as a single-field ordering reference. Under
       the stack-all policy (#215) it is no longer the claim gate: ``claim`` now
       claims any directory with enough coherent TIFFs and ``__init__`` stacks
       them (normalizing dtype/shape), delegating axis semantics to the agent.

    Files are bucketed by their digit-run mask; the dominant (largest) bucket is
    inspected for exactly one numeric field that varies across its members (all
    other numeric tokens, e.g. the ``s1`` in ``s1-0001_bf.tif``, must be
    constant). The bucket is sorted by that varying field.

    ``exclude_ome`` is forwarded to :func:`_filter_tiff_candidates`. claim() sets
    it to ``not ctx.cloud_root`` (OME ownership arbitration vs OmeTiffAdapter);
    read passes ``False`` (the directory is already claimed, so all TIFFs are
    members). See the call sites for the rationale.

    Returns the sorted file list, or ``None`` if no valid single-varying-field
    sequence of at least three files exists. Never returns an empty list.
    """
    candidates = _filter_tiff_candidates(files, exclude_ome=exclude_ome)
    if len(candidates) < 3:
        return None

    # Bucket files by structural mask. Files with no digit runs get their own
    # singleton masks and fall out below (they can never form a varying field).
    groups: Dict[str, List[Tuple[Path, List[int]]]] = {}
    for f in candidates:
        mask, digits = _mask_and_digits(f.name)
        groups.setdefault(mask, []).append((f, digits))

    # Dominant bucket: most files, tie-broken by mask string for determinism.
    best_mask = max(groups, key=lambda m: (len(groups[m]), m))
    members = groups[best_mask]
    if len(members) < 3:
        return None

    n_fields = len(members[0][1])  # all members share the mask -> same count
    if n_fields == 0:
        return None

    varying = [
        pos
        for pos in range(n_fields)
        if len({digits[pos] for _, digits in members}) > 1
    ]
    if len(varying) != 1:
        return None

    vpos = varying[0]
    members.sort(key=lambda fd: fd[1][vpos])
    return [f for f, _ in members]


def _resolve_aszarr_axes(axes: str, ndim: int) -> Tuple[int, int, Optional[int]]:
    """Locate the (Y, X, page) axes within a ``series[0].aszarr()`` array.

    tifffile orders a series' axes as ``[sequence…] Y X [samples]``: an IFD/page
    *sequence* axis (present only when a file holds several pages) leads, the
    spatial pair ``Y X`` sits in the middle, and a *samples* axis (``S``/``Q`` --
    RGB, or a singleton extrasample) trails ``X``. So the spatial pair is NOT
    reliably the last two dims: a trailing samples axis (``YXS``/``YXQ``, e.g.
    what ``imwrite`` produces from a ``(Y, X, 1)`` array) pushes ``shape[-2:]``
    onto ``(X, samples)``. Reading off that mistaken pair, and then treating the
    3-D shape as ``(page, Y, X)``, collapses each plane to a single column
    (biopb/biopb#220). Keying off the axes string instead is read-free and
    matches the descriptor, which already takes ``Y``/``X`` from ``page.shape[:2]``.

    Returns ``(y_ax, x_ax, page_ax)``; ``page_ax`` is the leading sequence axis,
    or ``None`` for a single-page file. The caller fixes every other (samples)
    axis at index 0.
    """
    a = axes.upper()
    if "Y" in a and "X" in a:
        y_ax, x_ax = a.index("Y"), a.index("X")
    else:  # opaque axes -> fall back to the legacy "spatial pair is last two"
        y_ax, x_ax = ndim - 2, ndim - 1
    # The page/sequence axis (if any) precedes the spatial pair; samples trail it.
    page_ax = next(
        (i for i in range(ndim) if i not in (y_ax, x_ax) and i < min(y_ax, x_ax)),
        None,
    )
    return y_ax, x_ax, page_ax


def _read_aszarr_plane(
    zarr_arr: Any,
    y_ax: int,
    x_ax: int,
    page_ax: Optional[int],
    page_idx: int,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """Read one (Y, X) plane from an aszarr zarr array at ``page_idx``.

    Indexes via the resolved axis positions (biopb/biopb#220): Y/X take the
    requested slices, a leading page axis takes ``page_idx``, and every other axis
    (e.g. a trailing RGB samples axis) is fixed at 0 -- so a ``YXS``/``YXQ`` frame
    is never read off the wrong axis. Transposes if tifffile ever emits X before Y.
    Callers resolve axes once via :func:`_resolve_aszarr_axes`; the shared indexing
    lives here so the sequence and MicroManager read paths cannot drift apart again.
    """
    index: List[Any] = [0] * zarr_arr.ndim
    index[y_ax] = y_slice
    index[x_ax] = x_slice
    if page_ax is not None:
        index[page_ax] = page_idx
    plane = zarr_arr[tuple(index)]
    if y_ax > x_ax:  # tifffile emits Y before X; transpose if ever not
        plane = plane.T
    return plane


class _PerFileTiffLockMixin:
    """Per-file read locks: serialize reads of the SAME file, parallelize others.

    A single adapter-wide lock made one slow/stalled file read (a cloud/VM I/O
    stall) freeze every other plane the async viewer and precache worker asked
    for -- turning one hiccup into a multi-second freeze across unrelated planes.
    Keyed per file, a stall holds only its own file's lock. The registry holds one
    small ``Lock`` per distinct file and never needs pruning. Call
    :meth:`_init_file_locks` from ``__init__`` before any read.
    """

    def _init_file_locks(self) -> None:
        self._locks_guard = threading.Lock()
        self._file_locks: Dict[Path, threading.Lock] = {}

    def _lock_for(self, path: Path) -> threading.Lock:
        """Return the per-file lock for ``path``, creating it on first use."""
        with self._locks_guard:
            lock = self._file_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[path] = lock
            return lock


# =============================================================================
# TiffSequenceAdapter - Plain TIFF sequences (no metadata)
# =============================================================================


class TiffSequenceAdapter(_PerFileTiffLockMixin, SourceAdapter, TensorAdapter):
    """Adapter for plain TIFF file sequences in a directory (no metadata).

    Handles datasets where multiple TIFF files form a single logical image:
    - ND000_aligned.tiff, ND001_aligned.tiff, ND002_aligned.tiff, ...

    This is a single-tensor source that tracks file list and index mapping.
    Uses TiffFile().aszarr() for true tile-level lazy reading.

    Unlike MultiFileOmeTiffAdapter, this does NOT:
    - Parse OME-XML or MicroManager metadata
    - Handle 5D/6D coordinate mapping
    - Discover related files via companion files

    Stack-all policy (#215): every TIFF in the directory that can share a tensor
    is stacked along an opaque file axis (label ``i``); the axis's semantic
    structure (channel / time / site / z -- e.g. MetaMorph ``_w/_s/_t`` or
    ``_red/_green/_blue``) is deliberately NOT inferred here. Instead the per-file
    names are exposed via ``get_metadata`` so a downstream agent can parse them
    and reshape / relabel. Differing dtype and spatial size are normalized into
    the stack (#198: widest dtype, zero-pad to the max plane); only a differing
    page count (or an unreadable file) is left out, and listed as a sibling. This
    is metadata-free, never silently drops files, and avoids guessing axes wrong.

    Shape:
    - (num_files, Y, X) for single-page files -> ['i', 'y', 'x']
    - (num_files, pages, Y, X) for multi-page files -> ['i', 'z', 'y', 'x']
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories holding several plain TIFFs (stack-all, #215).

        Claim when at least ``_MIN_TIFF_FILES`` TIFFs are present (excluding OME
        names owned by OmeTiffAdapter) AND their names cohere as a related set
        (see ``_looks_like_tiff_sequence``). The previous single-varying-numeric-
        field requirement is gone: directories whose filenames encode several axes
        (channel x site x time, ``_red/_green/_blue``, MetaMorph ``_w/_s/_t``) are
        now claimed too. ``__init__`` stacks the dominant *shape* group and
        exposes per-file names for the agent to interpret -- no filename-pattern
        inference, and no silent channel drop. The coherence gate only avoids
        welding an incidental grab-bag of unrelated TIFFs into one tensor; it does
        NOT parse what the fields mean.

        Claiming is intentionally metadata-free — no per-file reads, so discovery
        scans stay cheap. Stackability (shape/dtype/pages) is resolved lazily in
        ``__init__`` (where every file is opened anyway), not here.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if enough TIFFs are present
        """
        # Only support local directories for now (remote glob/stat is expensive)
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Gather all TIFF files. Match common extension cases (MetaMorph and
        # other instrument exports use uppercase ``.TIF``), deduping by path so
        # case-insensitive filesystems don't double-count. Route through ctx.glob
        # so the snapshot's cached child listing serves the match (biopb#65).
        seen: Dict[str, Path] = {}
        for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for t in ctx.glob(pat):
                seen[str(t._path)] = t._path
        tiff_files = list(seen.values())

        # OME guards arbitrate ownership against the file-level OmeTiffAdapter, and
        # apply only OFF cloud. Under a cloud root OmeTiffAdapter is disabled (it
        # returns None for every .ome.tif / .companion.ome), so the files it would
        # normally own degrade to per-file aics sources; deferring to it there is
        # pointless and grouping the directory is the better fallback. This is a
        # claim-time-only decision: once the directory is claimed, __init__ reads
        # ALL TIFFs (exclude_ome=False) since OmeTiffAdapter is already locked out.
        exclude_ome = not ctx.cloud_root

        # OME companion guard: a *.companion.ome marks this directory as a
        # multi-file OME-TIFF set owned by the file-level OmeTiffAdapter, so don't
        # claim it out from under that adapter. MicroManager metadata.txt /
        # DisplaySettings.json are deliberately NOT guarded here -- MicroManager-
        # LegacyAdapter has higher priority and prunes any valid MM dataset before
        # this adapter runs, so a metadata file that survives to here belongs to a
        # dataset MM could not parse (e.g. a truncated metadata.txt from an aborted
        # acquisition); falling back to a plain sequence claim is preferable to
        # registering every frame as its own per-file source. Route through
        # ctx.glob (not ctx._path.glob) so the snapshot's cached child listing
        # serves the match without re-reading the directory (biopb/biopb#65).
        if exclude_ome and ctx.glob("*.companion.ome"):
            return None

        # Stack-all claim gate (#215): enough plain TIFFs present AND their names
        # cohere as a related set (not a grab-bag). No filename-template *parsing*
        # -- multi-field names are claimed too and sorted out by the agent at read
        # time -- only a coherence check. Metadata-free: filenames only, no reads.
        candidates = _filter_tiff_candidates(tiff_files, exclude_ome=exclude_ome)
        if len(candidates) < _MIN_TIFF_FILES:
            return None
        if not _looks_like_tiff_sequence([p.name for p in candidates]):
            return None

        # Dir-claiming policy (biopb/biopb): the directory IS the dataset
        # boundary. Claiming the dir prunes its whole subtree, so the interior
        # TIFFs are never independently walked and need not be recorded as
        # members -- that would only duplicate the prune and pin a brittle
        # membership.
        state.try_claim_path(ctx.path_str)

        # Cloud-storage phase 2 (biopb/biopb#173): ``__init__`` opens EVERY file
        # in the sequence to validate dimension consistency (see below). Under a
        # cloud root those members may be dehydrated placeholders, so building the
        # adapter eagerly would recall the whole sequence -- one synced-folder
        # round-trip per file (~hundreds of ms each on OneDrive Files-On-Demand),
        # serialized -- during the synchronous startup scan. That wedges health at
        # STARTING for minutes (a 749-file sequence measured at ~4 min) before
        # ``mark_ready()`` is ever reached, and the per-file content skip the cloud
        # policy applies elsewhere never fires because the sequence claim records
        # only the directory (``is_file()`` is False, so ``_claim_is_unresolved``
        # has no member to test). Claim the directory provisionally and defer
        # construction to first access, where the sequence is hydrated and opening
        # every file is expected -- the same strategy MicroManagerLegacyAdapter
        # already uses for its placeholder metadata.txt. Gated on ``cloud_root``
        # (not residency) so the deferral still holds at resolve, when the files
        # are resident -- consistent with OmeTiffAdapter and the cloud opt-in's
        # "don't open it eagerly" contract. An unresolved source also advertises no
        # tensors, so the precache worker skips it instead of warming (recalling)
        # every chunk in the background.
        return SourceClaim(
            source_type="tiff-sequence",
            primary_path=ctx.path_str,
            unresolved=ctx.cloud_root,
        )

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "TiffSequenceAdapter":
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or "", source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize TIFF sequence adapter.

        Args:
            directory: Path to directory containing TIFF sequence
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from file count)
        """
        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "tiff-sequence"
        # Per-file read locks (see _PerFileTiffLockMixin): reads of the same TIFF
        # serialize while reads of different files run in parallel, so one slow
        # read can't freeze every other frame.
        self._init_file_locks()
        # Memoized physical scale: computing it reopens members[0], so cache the
        # result (incl. a None) after the first call -- see _physical_scale.
        self._physical_scale_cache: Any = _SCALE_UNSET

        # Gather every TIFF in the claimed directory. Unlike claim(), read does
        # NOT exclude OME names: the OME exclusion is a claim-time ownership
        # decision; by now the directory is claimed and its subtree pruned, so
        # every TIFF here is a member. Natural-sort for a stable, numeric-aware
        # default order (``img_2`` before ``img_10``); what that order *means* is
        # the agent's to decide (see get_metadata), not ours.
        all_tiffs = sorted(
            (
                p
                for p in self.directory.iterdir()
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
            ),
            key=lambda p: _natural_key(p.name),
        )
        if not all_tiffs:
            raise ValueError(f"No TIFF files found in {directory}")

        # Stack-all policy (#215) + per-file normalization (#198). A dense tensor
        # has exactly one hard constraint: every member must contribute the same
        # number of pages, since the page count sets the tensor's ndim (whether a
        # file adds a page axis at all) and so cannot be reconciled. Differing
        # dtype and spatial size, by contrast, ARE normalized at read time -- the
        # descriptor takes the widest dtype (np.result_type: promote *up* so no
        # member's values clip) and the per-axis max plane (smaller frames
        # zero-pad in get_data). So we bucket only by page count; files with a
        # different page count are not stacked but are surfaced via
        # get_metadata(), as are unreadable ones. We do not parse what the file
        # axis means (channel / time / site / z): that is delegated to the agent,
        # which gets the per-file names alongside the array.
        #
        # Exception policy: OSError (a missing file, or a cloud recall that fails
        # on a network blip) is a transport failure -> re-raise, so it surfaces as
        # a retryable error instead of silently shrinking the stack. Any other
        # error means the file opened at the I/O level but is not a stackable
        # image (TiffFileError, or a corrupt header that makes ``pages[0]`` raise)
        # -> demote to a sibling. Each file is probed exactly once.
        probes: Dict[Path, Tuple[Any, ...]] = {}  # h, w, dtype, npages, tiled, tw, tl
        buckets: Dict[int, List[Path]] = {}
        unreadable: List[Path] = []
        for p in all_tiffs:
            try:
                with tifffile.TiffFile(str(p)) as tf:
                    page = tf.pages[0]
                    probe = (
                        page.shape[0],
                        page.shape[1],
                        str(page.dtype),
                        len(tf.pages),
                        bool(page.is_tiled),
                        page.tilewidth,
                        page.tilelength,
                    )
            except OSError:
                raise  # transport / recall failure -- retryable, do not swallow
            except Exception:
                unreadable.append(p)  # not a valid image -- demote to sibling
                continue
            probes[p] = probe
            buckets.setdefault(probe[3], []).append(p)

        if not buckets:
            raise ValueError(f"No readable TIFF files in {directory}")

        # Dominant bucket: most files; tie-broken toward the higher page count for
        # determinism. Members keep all_tiffs' natural order.
        n_pages_per_file, members = max(
            buckets.items(), key=lambda kv: (len(kv[1]), kv[0])
        )

        # Coherence gate at resolve too (mirrors claim()): the stacked members must
        # look like a related set. Filename-only; the one mismatch normalization
        # can't fix, so it raises rather than stack nonsense. Two messages: if the
        # whole directory's names cohere but a page-count split (or an unreadable
        # file) left the dominant subset too small, blame the split; otherwise blame
        # the names. Gate the page-count message on the coherence of ALL files (not
        # the subset), so it is never claimed for a grab-bag with mixed page counts
        # -- reachable because an explicit-config source skips the claim-time gate.
        if not _looks_like_tiff_sequence([p.name for p in members]):
            names_cohere = _looks_like_tiff_sequence([p.name for p in all_tiffs])
            if (len(buckets) > 1 or unreadable) and names_cohere:
                bucket_summary = ", ".join(
                    f"{len(v)} file(s)x{k}pg"
                    for k, v in sorted(
                        buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])
                    )
                )
                unreadable_note = (
                    f", plus {len(unreadable)} unreadable" if unreadable else ""
                )
                raise ValueError(
                    f"Cannot stack the TIFFs in {directory} into one sequence: "
                    f"they split into {len(buckets)} page-count group(s) "
                    f"[{bucket_summary}]{unreadable_note}, and the largest group "
                    f"({len(members)} file(s) with {n_pages_per_file} page(s) "
                    f"each) is too small to form a sequence on its own. Files in "
                    f"a sequence must share a page count (the directory's "
                    f"filenames do cohere; it is the page-count split that "
                    f"prevents stacking)."
                )
            raise ValueError(
                f"TIFF files in {directory} do not look like one sequence "
                f"(no shared filename template or stem across the stacked files)"
            )

        self._tiff_files = members
        stacked = set(members)
        self._unstacked_files = [p for p in all_tiffs if p not in stacked] + unreadable

        # Normalized descriptor geometry across the members (#198):
        #  - dtype: the widest dtype, promoting up so no member's values clip;
        #  - spatial: the per-axis max plane (smaller frames zero-pad in get_data).
        members_info = [probes[p] for p in members]
        spatial_h = max(i[0] for i in members_info)
        spatial_w = max(i[1] for i in members_info)
        self._dtype = str(np.result_type(*(np.dtype(i[2]) for i in members_info)))

        # Tile / chunk geometry from members[0] (best effort; tiling may vary
        # across members, but the chunk grid is only a hint -- get_data reads each
        # file's own zarr and pads to the requested extent regardless).
        _, _, _, _, m0_tiled, m0_tw, m0_tl = probes[members[0]]
        self.is_tiled = m0_tiled
        if self.is_tiled:
            self.tile_width = m0_tw
            self.tile_length = m0_tl
            self._spatial_chunk = [m0_tl, m0_tw]
        else:
            self._spatial_chunk = [spatial_h, spatial_w]

        self._file_ifd_map = [(p, n_pages_per_file) for p in members]

        n_files = len(members)
        spatial_shape = [spatial_h, spatial_w]
        if n_pages_per_file > 1:
            # Multi-page files: (num_files, pages, Y, X). File axis is opaque;
            # page axis keeps the conventional 'z' default.
            self.full_shape = [n_files, n_pages_per_file] + spatial_shape
            self.chunk_shape = [1, 1] + self._spatial_chunk
            self.dim_labels = dim_labels if dim_labels else ["i", "z", "y", "x"]
        else:
            # Single-page files: (num_files, Y, X). 'i' = opaque file/stack axis.
            self.full_shape = [n_files] + spatial_shape
            self.chunk_shape = [1] + self._spatial_chunk
            self.dim_labels = dim_labels if dim_labels else ["i", "y", "x"]

        # Total IFDs for coordinate mapping
        self._total_ifds = sum(n for _, n in self._file_ifd_map)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def _read_padded_plane(
        self,
        zarr_arr: Any,
        y_ax: int,
        x_ax: int,
        page_ax: Optional[int],
        page_idx: int,
        y_slice: slice,
        x_slice: slice,
        fh: int,
        fw: int,
    ) -> np.ndarray:
        """Read one plane into a zero-padded, dtype-promoted buffer (#198).

        The buffer is sized to the requested ``y_slice`` / ``x_slice`` extent and
        typed as the descriptor dtype, so a frame smaller than the per-axis max
        (``fh`` x ``fw`` is *this* file's plane size) reads back zero-padded and a
        narrower-dtype frame is cast up. Returns a ``(out_h, out_w)`` array. The
        clamped inner read goes through the shared :func:`_read_aszarr_plane`
        (axis resolution incl. the #220 samples-axis fix); this wrapper only adds
        the pad/cast that a heterogeneous stack needs.
        """
        ys = y_slice.start or 0
        ye = y_slice.stop if y_slice.stop is not None else fh
        xs = x_slice.start or 0
        xe = x_slice.stop if x_slice.stop is not None else fw
        plane = np.zeros((ye - ys, xe - xs), dtype=self._dtype)
        ry, rx = min(ye, fh), min(xe, fw)
        if ry > ys and rx > xs:
            data = _read_aszarr_plane(
                zarr_arr, y_ax, x_ax, page_ax, page_idx, slice(ys, ry), slice(xs, rx)
            )
            plane[: ry - ys, : rx - xs] = data
        return plane

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )

        # Slice math (no I/O) needs no lock; only the per-file read below is
        # synchronized, and per file -- so a slow read of one frame no longer
        # blocks reads of other frames.
        pages_per_file = self._file_ifd_map[0][1] if self._file_ifd_map else 1
        original_ndim = len(slices)

        # Build slice tuple: (file_slice, [page_slice], y_slice, x_slice)
        if original_ndim == 4:
            file_slice, page_slice, y_slice, x_slice = slices
        elif original_ndim == 3:
            file_slice, y_slice, x_slice = slices
            page_slice = slice(0, pages_per_file) if pages_per_file > 1 else slice(0, 1)
        else:
            # Handle other dimensionalities
            file_slice = slices[0]
            page_slice = slice(0, pages_per_file) if len(slices) > 3 else slice(0, 1)
            y_slice = slices[-2] if len(slices) >= 2 else slice(None)
            x_slice = slices[-1] if len(slices) >= 1 else slice(None)

        # Determine which files and pages to read
        file_indices = range(
            file_slice.start or 0,
            min(file_slice.stop or len(self._file_ifd_map), len(self._file_ifd_map)),
        )
        page_indices = range(
            page_slice.start or 0,
            min(page_slice.stop or pages_per_file, pages_per_file),
        )

        n_files = (file_slice.stop or len(self._file_ifd_map)) - (file_slice.start or 0)
        n_pages = (page_slice.stop or pages_per_file) - (page_slice.start or 0)

        # Read data via aszarr() for tile-level access. Each plane is read
        # into a zero-filled buffer of the requested extent and cast to the
        # descriptor dtype (#198): a frame smaller than the per-axis max reads
        # back zero-padded (bottom/right), and a narrower-dtype frame is
        # promoted up -- so heterogeneous members stack uniformly.
        result_pages = []
        for file_idx in file_indices:
            file_path, n_pages_in_file = self._file_ifd_map[file_idx]
            # Per-file lock (not adapter-wide): serialize concurrent reads of the
            # SAME file while different files read in parallel. A scrub reads one
            # file per call, so this is a brief, usually-uncontended acquire -- and
            # a stalled read holds only this file's lock, not every frame's.
            with self._lock_for(file_path):
                with tifffile.TiffFile(str(file_path)) as tf:
                    series = tf.series[0]
                    zarr_arr = zarr.open_array(series.aszarr(), mode="r")
                    y_ax, x_ax, page_ax = _resolve_aszarr_axes(
                        series.axes, zarr_arr.ndim
                    )
                    fh, fw = zarr_arr.shape[y_ax], zarr_arr.shape[x_ax]
                    for page_idx in page_indices:
                        if page_idx < n_pages_in_file:
                            result_pages.append(
                                self._read_padded_plane(
                                    zarr_arr,
                                    y_ax,
                                    x_ax,
                                    page_ax,
                                    page_idx,
                                    y_slice,
                                    x_slice,
                                    fh,
                                    fw,
                                )
                            )

        # Stack into result array
        if result_pages:
            result_4d = np.stack(result_pages, axis=0)
            h, w = result_4d.shape[-2:]
            result_4d = result_4d.reshape(n_files, n_pages, h, w)
        else:
            result_4d = np.array([])

        # Reshape to match original ndim
        if original_ndim == 3:
            if pages_per_file > 1:
                result = result_4d.reshape(n_files * n_pages, h, w)
            else:
                result = result_4d.squeeze(axis=1)
        elif original_ndim == 4:
            result = result_4d
        else:
            result = result_4d

        return result

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (µm) from the first member's TIFF resolution tags.

        Memoized: computing it reopens ``members[0]`` to read its page header, so
        the result (a value *or* a ``None``) is cached after the first call and
        every later open reuses it instead of reopening the TIFF. See
        :meth:`_compute_physical_scale` for the projection itself.
        """
        if self._physical_scale_cache is _SCALE_UNSET:
            self._physical_scale_cache = self._compute_physical_scale()
        return self._physical_scale_cache

    def _compute_physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Read the physical scale off ``members[0]`` (see :meth:`_physical_scale`).

        X/Y come from ``X``/``YResolution`` (+ ``ResolutionUnit``, or the ImageJ
        calibration unit when present); Z, only for a multi-page stack, from the
        ImageJ ``spacing`` field. The opaque file axis (``i``) and any axis
        without a tag get ``0.0`` / ``""``. One page-header read of ``members[0]``
        under its per-file lock. Returns ``None`` when no usable resolution is
        present.
        """
        if not self._tiff_files:
            return None
        try:
            import tifffile

            path = self._tiff_files[0]
            with self._lock_for(path):
                with tifffile.TiffFile(str(path)) as tf:
                    page = tf.pages[0]
                    ij = tf.imagej_metadata or {}
                    ij_unit_um = unit_to_um(ij.get("unit"))
                    x_um = _tiff_pixel_size_um(page, "XResolution", ij_unit_um)
                    y_um = _tiff_pixel_size_um(page, "YResolution", ij_unit_um)
                    z_um = None
                    spacing = ij.get("spacing")
                    if spacing is not None and ij_unit_um is not None:
                        try:
                            z_um = float(spacing) * ij_unit_um
                        except (TypeError, ValueError):
                            z_um = None
            return scale_by_label(
                self.dim_labels, {"x": x_um, "y": y_um, "z": z_um}, MICRON
            )
        except Exception:
            logger.debug(
                "tiff-sequence: physical scale unavailable for %s",
                self._source_url,
                exc_info=True,
            )
            return None

    def get_metadata(self) -> dict:
        """Expose per-file provenance so the agent can interpret the file axis.

        The file axis (label ``i``) is an opaque stack of every uniformly-shaped
        TIFF in the directory; its semantic structure (channel / time / site / z
        -- e.g. MetaMorph ``_w/_s/_t`` or ``_red/_green/_blue``) is intentionally
        NOT inferred here. ``files`` lists the stacked members index-aligned to
        axis 0, so a downstream agent can parse the names and reshape / relabel as
        needed. ``unstacked_files`` lists TIFFs in the directory that could not
        join the stack -- a different page count (the one mismatch normalization
        can't fix) or an unreadable file -- present for completeness so nothing is
        silently dropped. (Differing dtype and spatial size do NOT land here: they
        are normalized into the stack per #198.)
        """
        md: Dict[str, Any] = {"files": [p.name for p in self._tiff_files]}
        if self._unstacked_files:
            md["unstacked_files"] = [p.name for p in self._unstacked_files]
        return md


# =============================================================================
# MicroManagerLegacyAdapter - Legacy MicroManager datasets with JSON metadata
# =============================================================================


class MicroManagerLegacyAdapter(_PerFileTiffLockMixin, SourceAdapter, TensorAdapter):
    """Adapter for legacy MicroManager datasets with JSON metadata.

    Handles MicroManager v1 v2 datasets with metadata.txt containing:
    - Summary: IntendedDimensions, AxisOrder, Channels, Slices, Frames, Positions
    - Coords-Default/<filename>: PositionIndex, TimeIndex, ChannelIndex, SliceIndex
    - Metadata-Default/<filename>: UUID, Width, Height

    This adapter supports full 5D/6D datasets: (position, time, channel, z, y, x).

    Uses TiffFile().aszarr() for true tile-level lazy reading.
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories with MicroManager metadata files.

        Handles both:
        - MicroManager v1: metadata.txt directly in directory
        - MicroManager v2: DisplaySettings.json at root, metadata.txt in subdirectory

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if MicroManager format
        """
        # Only support local directories for now
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Check for v1 metadata.txt directly in this directory
        v1_metadata = None
        if ctx.join("metadata.txt").exists():
            v1_metadata = ctx.join("metadata.txt")._path

        # Check for v2 DisplaySettings.json (need to find subdirectory with metadata.txt)
        v2_data_dir = None
        if v1_metadata is None and ctx.join("DisplaySettings.json").exists():
            # Look for subdirectories containing metadata.txt
            for subdir in ctx._path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    candidate_metadata = subdir / "metadata.txt"
                    if candidate_metadata.exists():
                        v2_data_dir = subdir
                        v1_metadata = candidate_metadata
                        break

        if v1_metadata is None:
            return None

        # Cloud-storage phase 2: a non-resident metadata.txt placeholder cannot be
        # read+parsed without a whole-file recall (or it blocks offline). Defer
        # format validation and the coordinate-map build: the directory was
        # recognized structurally (a metadata.txt plus img_* TIFFs, all recall-
        # free), so claim it provisionally and resolve on first access. The
        # authoritative Coords map is rebuilt then.
        #
        # Deferral is gated on the marker's *residency*, not on ctx.cloud_root --
        # deliberately unlike TiffSequenceAdapter above (cloud_root-gated). The two
        # are the codebase's two cloud-defer categories: MM recognizes the dataset
        # from one cheap marker file (metadata.txt), so residency of that marker is
        # the right signal and a resident MM dataset resolves eagerly (like OME-Zarr
        # .zattrs / a single DICOM header); a plain sequence has no marker and its
        # __init__ opens *every* file, so it must defer wholesale under cloud. See
        # cloud_phase2_test.py, which patches _is_offline_placeholder in this module.
        if _is_offline_placeholder(v1_metadata):
            # Dir-claiming policy: claim the dir (+ metadata.txt marker) only.
            # The dir claim prunes its whole subtree, so the interior img_*
            # TIFFs are never independently walked -- no need to enumerate them
            # as members (which under cloud would also be a recall-free but
            # divergent best-effort glob vs the Coords map).
            state.try_claim_path(ctx.path_str)
            state.try_claim_path(str(v1_metadata))
            if v2_data_dir:
                state.try_claim_path(str(v2_data_dir))
            return SourceClaim(
                source_type="micromanager-legacy",
                primary_path=ctx.path_str,
                unresolved=True,
            )

        # Parse metadata to confirm it's MicroManager v1 format
        try:
            content = v1_metadata.read_text()
            data = json.loads(content)

            # Check for MicroManager v1 format markers
            has_coords = any(k.startswith("Coords-") for k in data)
            has_summary = "Summary" in data

            if not (has_coords or has_summary):
                return None
        except (json.JSONDecodeError, OSError):
            return None

        # Discover TIFF files (in current dir or v2 data subdirectory)
        search_dir = v2_data_dir if v2_data_dir else ctx._path
        tiff_patterns = [
            "img_channel*.tif",
            "img_channel*.tiff",
            "img_pos*_channel*.tif",
            "img_pos*_channel*.tiff",
            "img_*.tif",
            "img_*.tiff",
        ]
        tiff_files = []
        seen = set()
        for pattern in tiff_patterns:
            for f in search_dir.glob(pattern):
                if f not in seen:
                    seen.add(f)
                    tiff_files.append(f)

        if not tiff_files:
            return None

        # Dir-claiming policy: the glob above is kept only to *confirm* this is a
        # real MicroManager dataset (TIFFs present); the dir claim prunes the
        # subtree, so the interior TIFFs need not be recorded as members.
        state.try_claim_path(ctx.path_str)
        state.try_claim_path(str(v1_metadata))
        if v2_data_dir:
            state.try_claim_path(str(v2_data_dir))

        return SourceClaim(
            source_type="micromanager-legacy",
            primary_path=ctx.path_str,
        )

    @staticmethod
    def _find_metadata_file(directory: Path) -> Optional[Path]:
        """Find MicroManager metadata file in directory.

        Handles both v1 (metadata.txt directly) and v2 (DisplaySettings.json at root,
        metadata.txt in subdirectory) formats.

        Args:
            directory: Path to directory to search

        Returns:
            Path to metadata.txt if found, None otherwise
        """
        # Check for v1 format: metadata.txt directly in directory
        v1_metadata = directory / "metadata.txt"
        if v1_metadata.exists():
            return v1_metadata

        # Check for v2 format: DisplaySettings.json at root, metadata.txt in subdirectory
        display_settings = directory / "DisplaySettings.json"
        if display_settings.exists():
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    candidate_metadata = subdir / "metadata.txt"
                    if candidate_metadata.exists():
                        return candidate_metadata

        return None

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "MicroManagerLegacyAdapter":
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or "", source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize MicroManager legacy adapter.

        Args:
            directory: Path to directory containing MicroManager dataset
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from metadata)
        """
        import json

        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "micromanager-legacy"
        # Per-file read locks (see _PerFileTiffLockMixin): a stalled read of one
        # plane holds only its own file's lock, not every plane's.
        self._init_file_locks()

        # Find and parse metadata file
        metadata_file = self._find_metadata_file(self.directory)
        if metadata_file is None:
            raise ValueError(f"No MicroManager metadata file found in {directory}")

        with open(metadata_file) as f:
            self._raw_metadata = json.load(f)

        # Parse Summary to get dimensions
        summary = self._raw_metadata.get("Summary", {})
        intended = summary.get("IntendedDimensions", {})

        # Get dimension sizes
        self._n_positions = intended.get("position", summary.get("Positions", 1))
        self._n_times = intended.get("time", summary.get("Frames", 1))
        self._n_channels = intended.get("channel", summary.get("Channels", 1))
        self._n_z = intended.get("z", summary.get("Slices", 1))

        # Build coordinate map: (p, t, c, z) -> filename
        self._coord_map: Dict[Tuple[int, int, int, int], Path] = {}
        self._file_list: List[Path] = []

        for key in self._raw_metadata:
            if key.startswith("Coords-"):
                coords = self._raw_metadata[key]
                # Extract path from key (Coords-Default/<filename> or Coords-<filename>)
                coords_prefix_len = len("Coords-")
                filepath_in_key = key[coords_prefix_len:]

                # Get indices
                pos_idx = coords.get("PositionIndex", 0)
                time_idx = coords.get("TimeIndex", 0)
                chan_idx = coords.get("ChannelIndex", 0)
                slice_idx = coords.get("SliceIndex", 0)

                # The Coords key path is written relative to the acquisition
                # *parent*, so for a "separate image files" acquisition it carries
                # a leading position-folder segment (e.g. "Default/img_...").
                # Which directory this source was claimed at decides how that
                # resolves: at the acquisition root (a DisplaySettings.json fired
                # the v2 claim there) the leading segment is real, but with no
                # DisplaySettings.json the root is invisible and the position
                # folder's own metadata.txt claims it directly -- so self.directory
                # *is* that folder and the segment doubles it (biopb/biopb#314).
                # Try the path as-is, then retry after stripping a leading segment
                # equal to this directory's own name, so both rootings land on the
                # same file. (v1 keys carry no prefix and resolve as-is.)
                file_path = self.directory / filepath_in_key
                if not file_path.exists():
                    parts = Path(filepath_in_key).parts
                    if len(parts) > 1 and parts[0] == self.directory.name:
                        file_path = self.directory.joinpath(*parts[1:])
                if file_path.exists():
                    self._coord_map[(pos_idx, time_idx, chan_idx, slice_idx)] = (
                        file_path
                    )
                    self._file_list.append(file_path)

        if not self._file_list:
            raise ValueError(f"No MicroManager TIFF files found in {directory}")

        # Sort file list for consistent ordering
        self._file_list = sorted(set(self._file_list))

        # Open first file to get shape and dtype
        with tifffile.TiffFile(str(self._file_list[0])) as tf:
            first_page = tf.pages[0]
            self._dtype = str(first_page.dtype)
            self._height = first_page.shape[0]
            self._width = first_page.shape[1]

            # Tile info
            if first_page.is_tiled:
                self.is_tiled = True
                self.tile_width = first_page.tilewidth
                self.tile_length = first_page.tilelength
                self._spatial_chunk = [self.tile_length, self.tile_width]
            else:
                self.is_tiled = False
                self._spatial_chunk = [self._height, self._width]

        # Get axis order from metadata
        axis_order = summary.get("AxisOrder", ["position", "time", "channel", "z"])
        if isinstance(axis_order, str):
            axis_order = [a.strip() for a in axis_order.split(",")]
        self._axis_order = axis_order

        # Map axis names to their counts
        axis_counts = {
            "position": self._n_positions,
            "time": self._n_times,
            "channel": self._n_channels,
            "z": self._n_z,
        }

        # Build shape following axis_order from metadata
        shape_axes = [a.lower() for a in axis_order]
        initial_shape = [axis_counts.get(a.lower(), 1) for a in axis_order] + [
            self._height,
            self._width,
        ]

        # Remove singleton dimensions while preserving axis correspondence
        # Only remove from non-spatial dimensions (keep y, x)
        self.full_shape = []
        self._shape_axes = []
        for i, (size, axis) in enumerate(zip(initial_shape, shape_axes, strict=False)):
            if i < len(shape_axes):  # Non-spatial axis
                if size > 1:  # Keep non-singleton dimensions
                    self.full_shape.append(size)
                    self._shape_axes.append(axis)
            else:  # This shouldn't happen, shape_axes is shorter
                pass

        # Always append spatial dimensions
        self.full_shape.extend([self._height, self._width])

        # Chunk shape
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self._spatial_chunk

        # Dimension labels (from remaining axes after singleton removal)
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            axis_alias = {
                "position": "p",
                "pos": "p",
                "time": "t",
                "frame": "t",
                "channel": "c",
                "z": "z",
                "slice": "z",
            }
            self.dim_labels = []
            for axis in self._shape_axes:
                label = axis_alias.get(axis.lower(), axis.lower()[0])
                self.dim_labels.append(label)
            self.dim_labels.extend(["y", "x"])

        # Build index for efficient lookups
        self._build_file_index()

    def _build_file_index(self) -> None:
        """Build index mapping for efficient coordinate lookups."""
        # Create reverse lookup: global_index -> file_path
        # For single-position/time/z datasets, this simplifies access
        self._index_to_file: Dict[int, Path] = {}

        if len(self.full_shape) == 3:
            # (channels, y, x) - single index is channel
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0 and z == 0:
                    self._index_to_file[chan] = file_path
        elif len(self.full_shape) == 4:
            # (channels, z, y, x) or similar
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0:
                    # Index = chan * n_z + z (assuming channel-z order)
                    idx = chan * self._n_z + z
                    self._index_to_file[idx] = file_path

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )

        # Slice math (no I/O) needs no lock; all state read below is immutable
        # after __init__. Only the per-file read is synchronized, and per file --
        # so a slow read of one plane no longer blocks reads of other planes.
        ndim = len(self.full_shape)

        # Extract spatial slices (last 2 axes)
        y_slice = slices[-2] if len(slices) >= 2 else slice(None)
        x_slice = slices[-1] if len(slices) >= 1 else slice(None)

        # Determine coordinate ranges based on shape
        if ndim == 3:
            # (channels, y, x)
            chan_slice = slices[0]
            pos_range = [0]
            time_range = [0]
            chan_range = range(
                chan_slice.start or 0, chan_slice.stop or self._n_channels
            )
            z_range = [0]
        elif ndim == 4:
            # (channels, z, y, x) or (time, channels, y, x) depending on axis_order
            if "z" in self.dim_labels[:2]:
                # (channels, z, y, x) format
                chan_slice = slices[0]
                z_slice = slices[1]
                pos_range = [0]
                time_range = [0]
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            else:
                # (time, channels, y, x) format
                time_slice = slices[0]
                chan_slice = slices[1]
                pos_range = [0]
                time_range = range(
                    time_slice.start or 0, time_slice.stop or self._n_times
                )
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = [0]
        elif ndim == 5:
            # (time, channels, z, y, x) or (position, channels, z, y, x)
            if "p" in self.dim_labels:
                pos_slice = slices[0]
                chan_slice = slices[1]
                z_slice = slices[2]
                pos_range = range(
                    pos_slice.start or 0, pos_slice.stop or self._n_positions
                )
                time_range = [0]
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            else:
                time_slice = slices[0]
                chan_slice = slices[1]
                z_slice = slices[2]
                pos_range = [0]
                time_range = range(
                    time_slice.start or 0, time_slice.stop or self._n_times
                )
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
        elif ndim == 6:
            # (position, time, channels, z, y, x)
            pos_slice = slices[0]
            time_slice = slices[1]
            chan_slice = slices[2]
            z_slice = slices[3]
            pos_range = range(pos_slice.start or 0, pos_slice.stop or self._n_positions)
            time_range = range(time_slice.start or 0, time_slice.stop or self._n_times)
            chan_range = range(
                chan_slice.start or 0, chan_slice.stop or self._n_channels
            )
            z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
        else:
            # Fallback - assume all dimensions except last 2 are 1
            pos_range = [0]
            time_range = [0]
            chan_range = [0]
            z_range = [0]

        # Collect data for each coordinate
        result_pages = []
        for pos_idx in pos_range:
            for time_idx in time_range:
                for chan_idx in chan_range:
                    for slice_idx in z_range:
                        coord = (pos_idx, time_idx, chan_idx, slice_idx)
                        file_path = self._coord_map.get(coord)

                        if file_path is None:
                            # Create empty array for missing data
                            h = (
                                y_slice.stop - (y_slice.start or 0)
                                if y_slice.stop
                                else self._height
                            )
                            w = (
                                x_slice.stop - (x_slice.start or 0)
                                if x_slice.stop
                                else self._width
                            )
                            result_pages.append(np.zeros((h, w), dtype=self._dtype))
                            continue

                        # Per-file lock (not adapter-wide): a stalled read holds
                        # only this file's lock, not every plane's.
                        with self._lock_for(file_path):
                            with tifffile.TiffFile(str(file_path)) as tf:
                                series = tf.series[0]
                                zarr_arr = zarr.open_array(series.aszarr(), mode="r")
                                # Each MM file holds one plane (page 0); resolve Y/X
                                # from the axes string so an RGB (YXS) frame reads the
                                # plane, not row 0 (biopb/biopb#220). Shared read path
                                # with TiffSequenceAdapter.
                                y_ax, x_ax, page_ax = _resolve_aszarr_axes(
                                    series.axes, zarr_arr.ndim
                                )
                                result_pages.append(
                                    _read_aszarr_plane(
                                        zarr_arr,
                                        y_ax,
                                        x_ax,
                                        page_ax,
                                        0,
                                        y_slice,
                                        x_slice,
                                    )
                                )

        # Build output array with proper shape
        if not result_pages:
            return np.array([])

        # Stack pages and reshape to match the expected shape
        n_pos = len(pos_range)
        n_time = len(time_range)
        n_chan = len(chan_range)
        n_z = len(z_range)

        h = result_pages[0].shape[0] if result_pages else 0
        w = result_pages[0].shape[1] if result_pages else 0

        # Stack all pages and reshape
        result = np.stack(result_pages, axis=0)

        # Reshape to (pos, time, chan, z, h, w) based on actual ranges
        if ndim == 2:
            # Genuine 2-D [y, x] dataset (all non-spatial axes singleton and
            # dropped from full_shape). result_pages holds exactly one page,
            # so drop the leading stack axis to match the descriptor rank.
            result = result.reshape(h, w)
        elif ndim == 3:
            result = result.reshape(n_chan, h, w)
        elif ndim == 4:
            result = result.reshape(n_chan * n_z, h, w)
            if "z" in self.dim_labels[:2]:
                result = result.reshape(n_chan, n_z, h, w)
            else:
                result = result.reshape(n_time, n_chan, h, w)
        elif ndim == 5:
            if "p" in self.dim_labels:
                result = result.reshape(n_pos, n_chan, n_z, h, w)
            else:
                result = result.reshape(n_time, n_chan, n_z, h, w)
        elif ndim == 6:
            result = result.reshape(n_pos, n_time, n_chan, n_z, h, w)

        return result

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (µm) from the MicroManager ``Summary`` metadata.

        ``PixelSize_um`` (isotropic X/Y) and the z-step onto the ``x`` / ``y`` /
        ``z`` axes; position / time / channel axes get ``0.0`` / ``""``.
        """
        summary = self._raw_metadata.get("Summary", {})
        return mm_summary_scale(summary, self.dim_labels)

    def get_metadata(self) -> dict:
        """Return parsed MicroManager metadata."""
        return self._raw_metadata
