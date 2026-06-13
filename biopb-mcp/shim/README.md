# napari-biopb (deprecated)

**`napari-biopb` has been renamed to [`biopb-mcp`](https://github.com/biopb/biopb/tree/main/biopb-mcp).**

This package is a backward-compatibility shim. Installing it pulls in
`biopb-mcp` and re-exports its top-level API, so existing
`import napari_biopb` code keeps working. It will not receive further updates.

Please switch your dependency:

```bash
pip install biopb-mcp
```

and update imports from `napari_biopb` to `biopb_mcp`.

## Publishing this shim

This is a separate distribution from the main project — build and upload it
on its own:

```bash
cd shim
python -m build
twine upload dist/*
```

Before uploading, make sure the `version` in `pyproject.toml` is **higher**
than the last `napari-biopb` release already on PyPI (the last git tag was
`v0.5.0`).
