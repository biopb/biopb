# BioPB
A place for collecting protobuf/gRPC definitions for bio-imaging data. Currently it has only two packages

1. `biopb.image` Image processing protocols. Current focus is single-cell segmentation, designed originally for the [Lacss](https://github.com/jiyuuchc/lacss/) project.
2. `biopb.ome` Microscopy data representation modeled after [OME-XML](https://ome-model.readthedocs.io/en/stable/ome-xml/index.html).


## Documentation
[Documentation](https://buf.build/jiyuuchc/biopb/)

## Language bindings
### Python

A python binding of schema is included in this repo. The package additionally implements some utility functions for data conversion between _numpy_ <--> _protobuf_.

``` sh
pip install biopb
```
### Java
The Java binding contains util functions based on imglib2. To use in Maven based project:

![Maven Central Version](https://img.shields.io/maven-central/v/io.github.jiyuuchc/biopb)

```xml
<dependencies>
<dependency>
<groupId>io.github.jiyuuchc</groupId>
<artifactId>biopb</artifactId>
<version>CURRENT_VERSION</version>
</dependency>
</dependencies>
```

### Other languages
For all other languages use the automatically generated SDK from [buf.build](https://buf.build/jiyuuchc/biopb/sdks/main:protobuf)

## Related project
* [`napari-biopb`](https://github.com/biopb/napari-biopb) is a [napari](https://napari.org) widget and a `biopb.image` client, allowing users to perform 2D/3D single-cell segmentation within the Napari environement.
* [`trackmate-bipb`](https://github.com/biopb/trackmate-biopb) is a [`FIJI`](https://imagej.net/software/fiji/) plugin and a `biopb.image` client, designed as a cell detector/segmentor for [`trackmate`](https://imagej.net/plugins/trackmate/index), but also works a stand-alone unit.
* [`biopb-server`](https://github.com/biopb/biopb-server) implement ready-to-deploy biopb servers (as Docker containers).
