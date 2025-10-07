# BioPB
A place for collecting protobuf/gRPC definitions for bio-imaging data. Currently it has only two packages

1. `biopb.ome` Microscopy data representation modeled after [OME-XML](https://ome-model.readthedocs.io/en/stable/ome-xml/index.html).
2. `biopb.image` Image processing protocols. Current focus is single-cell segmentation, designed originally for the [Lacss](https://github.com/jiyuuchc/lacss/) project.


## Documentation
[Documentation](https://buf.build/jiyuuchc/biopb/)

## Python binding
A python binding of schema is included in this repo. The package additionally implements some utility functions for data conversion between _numpy_ <--> _protobuf_.

``` sh
pip install biopb
```

## Related project
* [`napari-biopb`](https://github.com/biopb/napari-biopb) is a [napari](https://napari.org) widget and a `biopb.image` client, allowing users to perform 2D/3D single-cell segmentation within the Napari environement.
* [`trackmate-lacss`](https://github.com/biopb/TrackMate-Lacss) is a [`FIJI`](https://imagej.net/software/fiji/) plugin and a `biopb.image` client, designed as a cell detector/segmentor for [`trackmate`](https://imagej.net/plugins/trackmate/index). It works with any `biopb.image` servers.
* [`biopb-server`](https://github.com/biopb/biopb-server) implement ready-to-deploy biopb servers (as Docker containers).