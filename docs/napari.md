# Napari Plugin System and Core Components Overview

## Plugin System

Napari uses the **npe2** (Napari Plugin Engine 2) framework for managing plugins. Key points:

- **Discovery at Startup**: Plugins are discovered and loaded when napari starts, via `napari/plugins/__init__.py:_initialize_plugins()`. The npe2 `PluginManager` scans the environment for installed plugins, reads their `napari.yaml` manifests, and imports the plugin modules directly into the main napari process.
- **Same Process & Event Loop**: Plugins run in the **same Python interpreter** and share the **Qt event loop** as the napari GUI. There is no subprocess sandboxing by default.
- **Contribution Types**: Plugins can contribute:
  - **Readers/Writers** (`napari_get_reader`, `napari_get_writer`)
  - **Widgets** (appearing in the plugin dock or via `View > Plugins`)
  - **Commands** (menu items, toolbar buttons, shortcuts)
  - **Menus** (submenus under `Plugins`)
  - **Sample Data** (`napari_provide_sample_data`)
  - **Themes** (`napari_contribute_theme`)
  - **Hooks** (via npe2's hookspec/hookimpl mechanism, though napari currently defines few hooks)
- **Lazy Execution**: While the plugin code is loaded at startup, the actual execution of contributions (e.g., running a command or initializing a widget) occurs only when the user triggers it (e.g., clicks a menu item). However, import‑time side effects run during discovery and can block startup if they are long‑running or raise exceptions.
- **Threading & Blocking**: Because plugins share the main (GUI) thread, any blocking work performed at import time or in a contribution callback will freeze the UI. Plugin authors should offload heavy work to `QThread`, `QtConcurrent`, or asyncio tasks.
- **Configuration**: Users can disable plugins via `Settings > Plugins`; disabled plugins are skipped during discovery.

## Core Components of Napari

Napari’s architecture centers around a **Viewer** model and a **Qt-based GUI**. The main core components include:

| Component | Location | Purpose |
|-----------|----------|---------|
| **ViewerModel** | `src/napari/components/viewer_model.py` | Central state object holding layers, dims, camera, cursor, etc.; provides the public API (`viewer.add_image`, `viewer.window`, etc.). |
| **LayerList** | `src/napari/components/layerlist.py` | Ordered, observable collection of `Layer` objects; emits events on addition, removal, reordering, selection changes. |
| **Layers** | `src/napari/layers/` | Base `Layer` class and concrete types (`Image`, `Labels`, `Points`, `Shapes`, `Vectors`, `Surface`, `Tracks`). Each manages its own data, visual properties, and events. |
| **Dims** | `src/napari/components/dims.py` | Manages dimensional slicing, display, and world‑view coordinate conversion. |
| **Camera** | `src/napari/components/camera.py` | Controls 2D/3D view (zoom, pan, center) for the canvas. |
| **Canvas / QtViewer** | `src/napari/_qt/qt_viewer.py` | Embeds a `VisPy` canvas (or QtConsole) and handles mouse/keyboard events, console creation, and docking widgets. |
| **Window** | `src/napari/window.py` | Top‑level `QMainWindow` that assembles the menu bar, toolbars, status bar, layer list, plugin dock, and the central canvas. |
| **Event System** | Built on `PyQt5` signals and a custom `Event`/`Emitter` model (`src/napari/utils/event.py`). Enables loose coupling between components. |
| **Settings** | `src/napari/settings.py` | Persistent configuration (via `QSettings`) controlling plugins, themes, shortcuts, etc. |
| **Console** | `napari_console` package (imported lazily via `QtViewer._get_console`) | Embedded IPython QtConsole that shares the same interpreter as napari; used for interactive exploration. |
| **App Model** | Uses `app_model` for declarative command definitions (menus, palettes, shortcuts). |
| **Utilities** | Various helpers in `src/napari/utils/` (translations, notifications, progress, theme, etc.). |

These components interact primarily through the `ViewerModel` and its event system, allowing plugins to extend functionality by registering contributors that hook into the same extension points used by napari’s built‑in features.
