# biopb-mcp desktop-bundle packaging

PyInstaller inputs for building biopb-mcp as a standalone desktop app (a Windows
`.exe` / macOS `.app` that opens napari with the Tensor Browser). These are a
**manual/local** build path — the shipped product installs from the `release-v*`
wheel via `install.sh` / `install.ps1`, not from this bundle — so nothing in CI
consumes these files; they are kept for developers who build the desktop app.

| File | Role |
|------|------|
| `biopb-mcp.spec` | PyInstaller spec (entry point, hooks, excludes, icon, Win version info, macOS `.app` bundle) |
| `main.py` | Bundle entry point — opens a napari `Viewer` with the Tensor Browser widget |
| `hooks/` | Per-dependency PyInstaller hooks (napari, vispy, imageio, ipykernel, OpenGL, metadata) |
| `logo.ico` / `logo.icns` | Windows / macOS application icons |

## Build

Every input path in the spec is anchored to `SPECPATH` (the spec's own
directory), so you can invoke it from anywhere:

```sh
# from the repo root, with the biopb-mcp env active
pyinstaller biopb-mcp/packaging/biopb-mcp.spec
```

Output lands under `build/` and `dist/` in the current working directory (both
git-ignored).
