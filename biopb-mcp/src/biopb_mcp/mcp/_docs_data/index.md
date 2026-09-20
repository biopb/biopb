# biopb docs

Read one with `read_doc(id)`. This file is itself a doc: edit it with
`write_doc("index", old=…, new=…)` — regroup, re-hook, drop a line to retire a
doc of your own, and add a shipped id to the `ignored:` line to stop listing
one you do not want.

## Read first

- kernel: the namespace, kernel plugins, long-running jobs, where a compute runs
- data: pyramids, laziness, axis order and rank — read before moving pixels
- client: browse the catalog, load a tensor, upload a result
- viewer: layers, camera, dims, annotation layers, mouse events
- ops: the server-side image-processing ops in `ops`

## Procedures

- flatfield: correct uneven illumination across a set of tiles or fields
- stitch-tiles: assemble a grid of overlapping tiles into one mosaic
- align-stack-by-features: register serial sections that shifted and turned
- drift-correction: correct stage drift in a time-lapse before measuring
- deconvolve-widefield: deblur a widefield z-stack, and check it restored something
- pixel-classifier-segmentation: segment a field by training on a few scribbles
- count-foci-per-cell: count puncta inside each segmented cell, zeros included
- track-objects: follow segmented objects through a time-lapse, lineage included
- detect-filaments: trace filament centrelines and measure their width
- skeleton-network-metrics: length and branching of a filament network, in µm
- ratiometric-fret: a bleedthrough-corrected FRET ratio comparable between conditions
- measure-smlm-resolution: localize an SMLM stack and measure its resolution by FRC

## Writing

- authoring: what a procedure doc must contain, and the checkpoint types

ignored:
