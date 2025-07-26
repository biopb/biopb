# BioPB
A place for collecting protobuf/gRPC definitions for bio-research data. Currently it has only two packages

1. `biopb.ome` Microscopy data representation modeled after [OME-XML](https://ome-model.readthedocs.io/en/stable/ome-xml/index.html).
2. `biopb.image` Image processing protocols. Current focus is single-cell segmentation, designed originally for the [Lacss](https://github.com/jiyuuchc/lacss/) project.

## Public Servers

* `cellpose.biopb.org:443`
  - Model: [Cellpose](https://www.cellpose.org/) single-cellsegmentation (cyto3 variant). 2D and 3D.
* `lacss.biopb.org:443`
  - Model: [LACSS](https://github.com/jiyuuchc/lacss) base variant. 2D and 3D.
* `osilab.biopb.org:443`
  - Model: segformer as reported in [NIPS challenge paper](https://www.nature.com/articles/s41592-024-02233-6). 2D only.

## Related project
* [`napari-biopb`](https://github.com/jiyuuchc/napari-biopb) is a [Napari](https://napari.org) widget and a `biopb.image` client, allowing users to perform 2D/3D single-cell segmentation within the Napari environement.
* [`trackmate-lacss`](https://github.com/jiyuuchc/TrackMate-Lacss) is a [`FIJI`](https://imagej.net/software/fiji/) plugin and a `biopb.image` client, designed as a cell detector/segmentor for [trackmate](https://imagej.net/plugins/trackmate/index). It works with any `biopb.image` servers, such as those listed above.

## Documentation
[Documentation](https://biopb.github.io/biopb/)
