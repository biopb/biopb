"""Connected components on a chunked dask array, linked across chunk boundaries.

``scipy.ndimage.label`` applied per block is the obvious way to label a dask array
and it is wrong at every seam: the two halves of an object that straddles a chunk
boundary are numbered independently, and nothing reconciles them. The result is a
plausible label image whose object count is too high by however many objects touch
a boundary — no error, no warning, and the number is what you were computing. A
halo does not fix it: ``map_overlap`` gives each block a view of its neighbours,
but the two halves still receive numbers from two independent labellings.

Measured on 8 discs centred exactly on a chunk boundary, with the array split
exactly there::

    chunked_label.label(mask)                 -> 8 objects     (all 8 span both chunks)
    map_blocks(scipy.ndimage.label)           -> 16 objects    (every disc counted twice)

So this does the reconciliation properly: label each block independently, find
which block-local labels are the same object by looking at the two-pixel-thick
slab spanning each chunk face, resolve those pairings into global components, and
renumber. Everything stays lazy — ``label`` returns a dask array with the input's
chunking, and nothing is computed until you ask.

Two callables, reached through the module the agent gets bound
(``chunked_label``):

- ``label(mask, structure=None)`` — ``(labels, n)``, the relabeled dask array and
  the object count.
- ``object_sizes(labels, n)`` — pixel count per label, as a lazy 1-D dask array.

**This labels a *binary* mask, and it merges anything touching.** Connectivity is
the only thing it knows about; a nonzero pixel is foreground. Handing it an
existing *instance* segmentation does not preserve those instances — two touching
cells labelled 1 and 2 come back as one object, by design, exactly as
``scipy.ndimage.label`` would. Reconciling instance labels across tiles is a
different problem (match by overlap in the halo, then renumber) and this is not
it.

**What it costs.** Global numbering cannot be decided from one block, so computing
*any* part of the result labels the *whole* array: every block's labels are held
while the component graph is resolved. Budget the labeled array (4 bytes/pixel),
not just one chunk of it. Under the distributed scheduler that spills rather than
fails, but it is the reason this is not free on an array that only just fits.

**Chunk count drives the cost far more than pixel count does.** The work per
boundary is tiny, and there is one graph task per boundary, so the price is
task overhead and it climbs faster than linearly: on the same 2048x2048 field,
16 blocks took 0.07s, 256 took 4.2s, and 1024 took 128s. Prefer fewer, larger
chunks — rechunk up before labelling rather than accepting whatever the read
path handed you.

The approach — per-block labelling, face slabs, a sparse component graph — follows
``dask_image.ndmeasure.label`` (Dask developers, BSD-3-Clause), reimplemented here
so biopb does not take the dependency: ``dask-image``'s only unique contribution
for our purposes is this one function, and it pulls ``pims`` for a file-reading
path biopb does not use (see issue #671).
"""

# Private aliases keep the module's own surface to its public API, so
# `inspect_object("chunked_label")` shows the agent two callables rather than
# every scipy/dask handle this file imported. Style, not protection: as a kernel
# plugin this module is bound under one name.
import dask as _dask
import dask.array as da
import numpy as np
import scipy.ndimage as _ndi
import scipy.sparse as _sparse
import scipy.sparse.csgraph as _csgraph

__all__ = ["label", "object_sizes"]

# Whatever scipy labels with, so the relabeled array round-trips through
# `scipy.ndimage.label` without a widening cast on every block.
LABEL_DTYPE = _ndi.label(np.zeros(1, dtype=np.uint8))[0].dtype


def _label_block(block, structure):
    """Label one block on its own. Returns ``(labels, count)``, labels from 1."""
    labeled, count = _ndi.label(block, structure)
    return labeled.astype(LABEL_DTYPE), int(count)


def _offsets(counts):
    """Prefix sums of the per-block counts, so block ``k`` starts at ``offsets[k]``."""
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _shift(labeled, offsets, index):
    """Make block ``index``'s labels globally unique by adding its prefix offset.

    Background stays 0 — the offset is added only where a label is present, which
    is what keeps 0 meaning "nothing" rather than becoming the first block's
    offset.
    """
    offset = offsets[index]
    if offset == 0:
        return labeled
    shifted = labeled.copy()
    np.add(shifted, LABEL_DTYPE.type(offset), out=shifted, where=shifted > 0)
    return shifted


def _face_edges(blocks, offset, structure):
    """Pair up labels that are the same object across one block boundary.

    ``blocks`` are the labelled blocks meeting at that boundary — two for an
    axis-aligned face, ``2**k`` for a corner where the direction moves along ``k``
    axes — given in C order of their position in the little ``offset``-shaped
    grid. Each contributes only the one plane it touches the boundary with, so the
    slab assembled here is two pixels thick along every axis the direction moves
    along and full width along the rest.

    Labelling *that slab* with the same structuring element is what decides
    connectivity, and it is why a diagonal connection across the seam needs no
    special case: scipy answers it on the slab exactly as it would in the interior.

    Every original label sharing a slab component is the same object, so emit them
    as consecutive pairs (``a-b``, ``b-c``): the transitive closure is the
    component graph's job, not this function's.

    Returns a ``(2, M)`` array of label pairs, possibly empty.
    """
    ndim = len(offset)
    grid = tuple(2 if step else 1 for step in offset)
    pieces = np.empty(grid, dtype=object)
    # strict: the caller owes exactly one block per grid position, and a short
    # list would silently drop a side of the boundary rather than fail.
    for block, position in zip(blocks, np.ndindex(*grid), strict=True):
        pieces[position] = block[
            tuple(
                (slice(-1, None) if position[d] == 0 else slice(0, 1))
                if offset[d]
                else slice(None)
                for d in range(ndim)
            )
        ]
    face = np.block(pieces.tolist())

    components, _ = _ndi.label(face, structure)
    pairs = np.stack([components.ravel(), face.ravel()], axis=1)
    # A pixel is only evidence if it is foreground on both readings; background
    # carries no identity and would otherwise join every object to every other.
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if pairs.size == 0:
        return np.zeros((2, 0), dtype=LABEL_DTYPE)
    # Sorted by component, then label: rows adjacent within a component name two
    # labels known to be connected.
    pairs = np.unique(pairs, axis=0)
    same = np.flatnonzero(np.diff(pairs[:, 0]) == 0)
    return np.stack([pairs[same, 1], pairs[same + 1, 1]], axis=0).astype(LABEL_DTYPE)


def _global_lut(edge_arrays, total):
    """Resolve the pairings into a lookup table from block label -> final label.

    Entry 0 stays 0. The rest are renumbered contiguously from 1, so the result
    counts objects the way ``scipy.ndimage.label`` does on an array that fits.
    """
    size = int(total) + 1
    parts = [e for e in edge_arrays if e.size]
    if parts:
        edges = np.concatenate(parts, axis=1)
        i, j = edges[0], edges[1]
    else:
        i = j = np.zeros(0, dtype=np.int64)
    graph = _sparse.coo_matrix(
        (np.ones(i.size, dtype=np.int8), (i, j)), shape=(size, size)
    ).tocsr()
    _, components = _csgraph.connected_components(graph, directed=False)
    lut = np.zeros(size, dtype=LABEL_DTYPE)
    if size > 1:
        # Node 0 is isolated (no edge ever names it), so its component is its own
        # and renumbering labels 1.. cannot collide with the background.
        _, contiguous = np.unique(components[1:], return_inverse=True)
        lut[1:] = contiguous.ravel() + 1
    return lut


def _take(block, lut):
    """Apply the lookup table to one block."""
    return lut[block]


def _count_from_lut(lut):
    return np.int64(lut.max())


def label(mask, structure=None):
    """Label connected components of ``mask``, linking objects across chunks.

    A drop-in replacement for ``scipy.ndimage.label`` on a chunked dask array:
    same connectivity semantics, same numbering-from-1, but objects spanning a
    chunk boundary get **one** label instead of one per block.

    Args:
        mask: Foreground mask (dask or numpy array, any dimensionality). Nonzero
            is foreground — see the module docstring on why an instance
            segmentation is not a valid input.
        structure: Connectivity, shaped ``(3,) * ndim`` as
            ``scipy.ndimage.label`` takes it. Default is face connectivity
            (``generate_binary_structure(ndim, 1)``); pass ``np.ones((3,) * ndim)``
            to include diagonals.

    Returns:
        ``(labels, n)`` — a lazy dask array chunked like ``mask``, and the object
        count as a lazy 0-d dask array. Nothing has been computed yet; calling
        ``int(n)`` or ``labels.compute()`` runs the labelling over the whole array
        (see the module docstring on what that costs).
    """
    image = da.asarray(mask)
    if structure is None:
        structure = _ndi.generate_binary_structure(image.ndim, 1)

    # 1. Label every block on its own, then make the numbering globally unique by
    #    offsetting each block by the count of all blocks before it. Labelling is
    #    delayed once per block and reused: the count and the labels come out of
    #    the same task, so no block is labelled twice.
    label_block = _dask.delayed(_label_block, nout=2, pure=True)
    slices = list(da.core.slices_from_chunks(image.chunks))
    labeled, counts = [], []
    for block_slice in slices:
        block_labels, block_count = label_block(image[block_slice], structure)
        labeled.append(block_labels)
        counts.append(block_count)
    offsets = _dask.delayed(_offsets, pure=True)(counts)

    # `slices_from_chunks` and `np.ndindex` both walk the block grid in C order,
    # so position k in `labeled`/`counts` is the block at the k-th index.
    shift = _dask.delayed(_shift, pure=True)
    shifted = [
        shift(block_labels, offsets, k) for k, block_labels in enumerate(labeled)
    ]
    shapes = [tuple(s.stop - s.start for s in bs) for bs in slices]

    # 2. Two blocks agree about an object iff it shows up on both sides of the
    #    slab spanning their shared boundary.
    #
    #    Directions come from the *structure*, not from the axes. Under diagonal
    #    connectivity two objects can meet at a corner where 2**ndim blocks touch,
    #    and no axis-aligned face holds both of those pixels — that miscount is
    #    silent and rare enough to survive a hand-checked test, so it is worth
    #    stating why the loop looks like this. Widening the slab on every axis the
    #    direction moves along puts the whole corner in one face.
    #
    #    Only forward directions are walked, so each boundary is visited once.
    #    That still covers the *backward* diagonal (block (i,j+1) meeting block
    #    (i+1,j)): the corner slab spans all four blocks, so labelling it answers
    #    both diagonals at once.
    #
    #    Faces read the block tasks directly instead of slicing a reassembled
    #    array, so each one depends on the 2**k blocks it actually touches.
    #    Slicing is the obvious alternative and does not hold up: at 216 blocks
    #    it measured ~48x slower (3.8s against 182s) and segfaulted on one run in
    #    three while dask was still planning the graph, the cost of every face
    #    reaching back through the whole reassembled array.
    directions = [
        offset
        for offset in np.array(np.where(structure)).T - 1
        if offset.min() >= 0 and offset.max() > 0
    ]
    face_edges = _dask.delayed(_face_edges, pure=True)
    edges = []
    for index in np.ndindex(*image.numblocks):
        for offset in directions:
            grid = tuple(2 if step else 1 for step in offset)
            if any(index[d] + grid[d] > image.numblocks[d] for d in range(image.ndim)):
                continue
            corner = [
                shifted[
                    int(
                        np.ravel_multi_index(
                            tuple(index[d] + position[d] for d in range(image.ndim)),
                            image.numblocks,
                        )
                    )
                ]
                for position in np.ndindex(*grid)
            ]
            edges.append(face_edges(corner, tuple(int(s) for s in offset), structure))

    # 3. One graph over every block label, one pass of connected components, one
    #    lookup table applied to every block. The table is applied to the same
    #    `shifted` nodes the faces were read from, so no block is labelled twice.
    total = _dask.delayed(sum, pure=True)(counts)
    lut = _dask.delayed(_global_lut, pure=True)(edges, total)

    take = _dask.delayed(_take, pure=True)
    final = np.empty(image.numblocks, dtype=object)
    for k, index in enumerate(np.ndindex(*image.numblocks)):
        final[index] = da.from_delayed(
            take(shifted[k], lut), shape=shapes[k], dtype=LABEL_DTYPE
        )
    relabeled = da.block(final.tolist())
    n = da.from_delayed(
        _dask.delayed(_count_from_lut, pure=True)(lut), shape=(), dtype=np.int64
    )
    return relabeled, n


def object_sizes(labels, n_labels):
    """Pixel count for each label in ``labels``, as a lazy 1-D dask array.

    Args:
        labels: A label array, as returned by :func:`label`.
        n_labels: How many labels there are — the ``n`` from :func:`label`.
            Concrete, not lazy: the histogram's length has to be known to build
            it, so compute ``n`` first (``int(n)``).

    Returns:
        A lazy dask array of length ``n_labels``, where element ``i`` is the size
        of label ``i + 1``. Background is dropped rather than returned as element
        0, so the array lines up with labels 1..n and a size of 0 means a label
        that no longer has any pixels, not the background.
    """
    count = int(n_labels)
    histogram = da.bincount(da.asarray(labels).ravel(), minlength=count + 1)
    return histogram[1 : count + 1]
