"""Tensor browser widget for napari.

Provides a tree-based UI to browse biopb.tensor datastore catalog and add
selected tensors as dask arrays to the napari viewer. Supports authentication
tokens and search filtering.

Uses pure Qt for complex UI (tree widget, custom layouts).
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Set
from urllib.parse import urlparse

from biopb.tensor import ResolveCancelled
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor
from qtpy.QtCore import QRect, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._connection import TensorConnection
from .._tensor_utils import add_tensor_layer

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


# ==============================================================================
# Tree Building Utilities (adapted from JS SourceTree.tsx)
# ==============================================================================


class _TreeNode:
    """Internal tree node for building the source tree."""

    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: str,  # "folder" or "source"
        depth: int,
        source: DataSourceDescriptor | None = None,
    ):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.depth = depth
        self.source = source
        self.children: List[_TreeNode] = []


def _get_path_parts(url: str) -> List[str]:
    """Extract tree path parts from source_url.

    Splits on both POSIX (``/``) and Windows (``\\``) separators so a catalog
    indexed on Windows — whose ``source_url`` is a backslash path like
    ``C:\\Users\\me\\img.tif`` — builds the same folder tree as a POSIX one
    instead of collapsing into a single flat leaf (the whole path as one name).
    A leading drive-letter token (``C:``) is dropped: ``urlparse`` reads it as a
    URL scheme so it is usually already gone, but when it survives it is just
    noise in the folder hierarchy.

    For an authority URL (a non-empty netloc — remote ``tensor-server`` mirrors
    ``grpc://host:port/remote/path``, ``s3://bucket/key``, …) the endpoint
    ``<scheme>://<netloc>`` is emitted as the FIRST part, so mirrored sources nest
    by their remote filepath under an endpoint root rather than collapsing into a
    flat ``grpc:`` node (biopb/biopb#297). A local ``file://`` url has an empty
    netloc, so it is unchanged (still just its path).

    Mirror of the web viewer's ``getPathParts`` in
    ``biopb-tensor-server/packages/web/src/components/SourceTree.tsx`` — keep the
    two behaviorally in lockstep.
    """
    if not url:
        return []
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None
    if parsed is not None and parsed.scheme and parsed.netloc:
        path_parts = [p for p in re.split(r"[\\/]+", parsed.path) if p]
        return [f"{parsed.scheme}://{parsed.netloc}"] + path_parts
    raw = (parsed.path if parsed is not None else None) or url
    parts = [p for p in re.split(r"[\\/]+", raw) if p]
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    return parts


def _format_shape(shape: List[int]) -> str:
    """Format shape as compact string."""
    return "×".join(str(s) for s in shape)


def _tensor_short_name(array_id: str) -> str:
    """Get short name for tensor from its array_id."""
    parts = [p for p in array_id.split("/") if p]
    return parts[-1] if parts else array_id


# Leading glyph and tooltip for the per-source residency indicator. A cloud
# marks a source whose content is not local; resident sources stay unadorned.
_RESIDENCY_GLYPH = "☁"
_REMOTE_TOOLTIP = (
    "Not resident — content is remote or not yet local; "
    "reading it may be slow or block until it is hydrated. "
    "(Cloud / remote source support is experimental.)"
)
_RESIDENT_TOOLTIP = "Resident — content is local and cheap to read"


def _is_unresolved(src: DataSourceDescriptor) -> bool:
    """A source whose content has not been resolved yet: its field list is empty,
    so shape/dtype are unknown until the server hydrates it (the cloud / synced-
    folder case). Resolving such a source downloads its whole file, so it is an
    explicit, blocking action rather than something browsing triggers."""
    return len(src.tensors) == 0


class _ResolveWorker(QThread):
    """Runs the blocking ``TensorConnection.resolve_source`` off the GUI thread.

    Resolving a cloud source downloads the whole file (a recall that can take
    minutes), so it must not run on the Qt event loop. This thread does the work
    and reports back via signals; the widget keeps a modal progress dialog up
    until one fires, so the user is blocked from other actions but the app stays
    painted. Server heartbeats are relayed via :attr:`progress`, and the dialog's
    Cancel button calls :meth:`request_cancel` — a cooperative stop checked at
    each heartbeat (so it takes effect within one heartbeat interval; the
    server-side recall finishes and is cached, so a later resolve coalesces).
    """

    resolved = Signal(object)  # the refreshed DataSourceDescriptor
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)  # ResolveProgress

    def __init__(self, conn: TensorConnection, source_id: str):
        super().__init__()
        self._conn = conn
        self._source_id = source_id
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the running resolve to stop (thread-safe, idempotent)."""
        self._cancel.set()

    def run(self):
        try:
            descriptor = self._conn.resolve_source(
                self._source_id,
                on_progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except ResolveCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        self.resolved.emit(descriptor)


# Source types whose data lives across many files under one directory (the
# dir-claimed formats). Only these benefit from hydrate-ahead; a single-file
# source's bytes were already recalled by resolve, so warm is a server-side
# no-op there and we don't bother offering it.
_MULTIFILE_SOURCE_TYPES = frozenset(
    {
        "zarr",
        "ome-zarr",
        "ome-zarr-hcs",
        "ndtiff",
        "tiff-sequence",
        "micromanager-legacy",
    }
)


def _is_multifile_source(src: DataSourceDescriptor) -> bool:
    """A resolved, directory-backed multi-file source -- the case where member
    data files recall lazily onto the read path, so hydrate-ahead helps."""
    return not _is_unresolved(src) and src.source_type in _MULTIFILE_SOURCE_TYPES


# Item-data role carrying a hydrate-ahead ("warm") progress fraction on a source
# row: 0..1 = determinate, ``_WARM_INDETERMINATE`` = counts not known yet,
# absent/None = not warming. ``_WarmProgressDelegate`` paints a translucent fill
# across the row to that fraction instead of floating a progress dialog
# (biopb/biopb#202); cancel is offered from the context menu.
_WARM_ROLE = Qt.ItemDataRole.UserRole + 10
_WARM_INDETERMINATE = -1.0
# Translucent accent painted behind a hydrating row; the row's normal text, shape
# badge and residency glyph render on top unchanged.
_WARM_FILL = QColor(64, 132, 223, 60)


class _WarmProgressDelegate(QStyledItemDelegate):
    """Paints a hydrate-ahead progress fill behind a source row.

    The fraction lives on the item at :data:`_WARM_ROLE`; a row without it renders
    normally. A determinate fraction (0..1) fills the left portion of the row; the
    indeterminate sentinel tints the whole row faintly until the first server
    count arrives. The fill is drawn *under* the default item paint so the
    existing name / shape badge / residency glyph stay legible -- the bar is the
    only progress affordance (no percentage text). biopb/biopb#202.
    """

    def paint(self, painter, option, index):
        fraction = index.data(_WARM_ROLE)
        if fraction is not None:
            rect = QRect(option.rect)
            if fraction >= 0:
                rect.setWidth(int(rect.width() * max(0.0, min(1.0, fraction))))
            painter.fillRect(rect, _WARM_FILL)
        super().paint(painter, option, index)


class _WarmWorker(QThread):
    """Runs the blocking ``TensorConnection.warm_source`` off the GUI thread.

    Warming asks the server to recall all of a resolved source's member files
    (server-side; no pixels cross the wire), which can take minutes, so it must
    not run on the Qt event loop. Unlike resolve, warm is presented *non-modally*
    -- the user keeps browsing while it runs. Server progress (files/bytes) is
    relayed via :attr:`progress`; :meth:`request_cancel` cooperatively stops it
    (the client closes the stream, the server halts the recall).
    """

    warmed = Signal(object)  # terminal WarmProgress
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)  # WarmProgress

    def __init__(self, conn: TensorConnection, source_id: str):
        super().__init__()
        self._conn = conn
        self._source_id = source_id
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the running warm to stop (thread-safe, idempotent)."""
        self._cancel.set()

    def run(self):
        try:
            done = self._conn.warm_source(
                self._source_id,
                on_progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except ResolveCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        self.warmed.emit(done)


@dataclass
class _WarmState:
    """In-flight hydrate-ahead warm for one source (UI state).

    Holds the :class:`_WarmWorker` reference used to cancel from the context menu
    and the last progress ``fraction`` (``None`` = indeterminate); the fraction is
    kept here, not only on the tree item, so the inline bar can be re-applied after
    the tree is cleared and rebuilt. GC ownership of the QThread lives separately
    in ``_warm_retain`` (held until ``finished``), because this state is dropped
    earlier -- on ``warmed``/``cancelled``/``failed`` -- to flip the menu back.
    """

    worker: _WarmWorker
    fraction: float | None = None


# A dropped folder with more than this many filesystem entries prompts a
# confirmation before the recursive scan is sent (a footgun-stopper for dropping
# a home/root folder by mistake). Counted client-side: drag-drop is gated to a
# localhost server, so the client shares the server's disk. Coarse on purpose --
# it counts entries, not resulting sources.
_LARGE_DROP_ENTRY_THRESHOLD = 2000


def _is_onedrive_dir_name(name: str) -> bool:
    """True for a OneDrive root directory name (``OneDrive`` / ``OneDrive - <Org>``).

    A deliberate small copy of the server's discovery skip (biopb_tensor_server
    .discovery._is_skippable_system_dir): the server declines to walk these trees,
    which is exactly what makes a source added from inside one worth a heads-up.
    Kept independent rather than imported because the tensor-server package is not
    a runtime dependency of this widget.
    """
    low = name.lower()
    return (
        low == "onedrive" or low.startswith("onedrive -") or low.startswith("onedrive-")
    )


def _cloud_drop_warning(path: str) -> str | None:
    """Warning text if *path* sits in a cloud-synced folder, else ``None``.

    Name-based and best-effort: it inspects the path components only (never opens
    a file, so it cannot itself trigger a cloud recall). Detects OneDrive -- the
    common Windows case, and exactly the subtree the server's monitored-tree walk
    skips. Split on both separators so a Windows path survives whatever the drop
    delivered.
    """
    components = [c for c in re.split(r"[\\/]+", path) if c]
    if not any(_is_onedrive_dir_name(c) for c in components):
        return None
    name = os.path.basename(path.rstrip("/\\")) or path
    return (
        f"“{name}” is inside OneDrive, a cloud-synced folder.\n\n"
        "Cloud-synced source support is experimental.\n\n"
        "It will be indexed while its files are downloaded to this PC, but if "
        "OneDrive Files On-Demand later dehydrates them to free up space, reads "
        "can become slow or fail until Windows re-downloads them. Marking the "
        "files “Always keep on this device” avoids that.\n\n"
        "Add anyway?"
    )


def _dir_exceeds_entry_threshold(path: str) -> bool:
    """True if *path* is a directory holding more than the large-drop threshold.

    A non-directory (a single file/dataset) is never "large". Short-circuits at
    the threshold, so it stays cheap even on an enormous tree. This is the only
    size gate on a drop — the server does not re-check it (a direct SDK caller
    passing a path is trusted as explicit intent) — so a walk error (permission,
    race) is treated as "not large" and the drop proceeds unconfirmed rather
    than being blocked.
    """
    if not os.path.isdir(path):
        return False
    try:
        count = 0
        for _root, dirs, files in os.walk(path):
            count += len(dirs) + len(files)
            if count > _LARGE_DROP_ENTRY_THRESHOLD:
                return True
    except OSError:
        return False
    return False


class _AddSourceWorker(QThread):
    """Runs ``TensorConnection.add_source`` for one dropped path off the GUI thread.

    Registering a dropped file/dir asks the server to discover + catalog it,
    which for a plain folder is a slow recursive walk that may add many sources,
    so it must not run on the Qt event loop. Presented *non-modally* (like warm):
    the user keeps using the viewer while sources appear. Per-source progress is
    relayed via :attr:`progress`; :meth:`request_cancel` cooperatively stops the
    walk (the client closes the stream; sources already registered stay).

    One drop == one path == one ``add_source`` call == one terminal result
    ``(added, already_present, failed)``. Multi-item drops are refused upstream
    (``_local_paths_from_mime``), so there is no cross-path aggregation here — a
    single call keeps the progress count monotone. An oversized folder is caught
    *before* this worker starts, by the widget's client-side confirm prompt
    (``_confirm_large_drop``), so the walk only runs once the user has agreed.
    """

    progress = Signal(object)  # AddSourceProgress
    done = Signal(object)  # (added, already_present, failed)
    failed = Signal(str)

    def __init__(self, conn: TensorConnection, path: str):
        super().__init__()
        self._conn = conn
        self._path = path
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the running add to stop (thread-safe, idempotent)."""
        self._cancel.set()

    def run(self):
        try:
            result = self._conn.add_source(
                self._path,
                on_progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except Exception as exc:  # surface the SDK/server message to the user
            self.failed.emit(str(exc))
            return
        added = list(result.added)
        already = list(result.already_present)
        failed = [(f.path, f.reason) for f in result.failed]
        self.done.emit((added, already, failed))


def _human_bytes(n: int) -> str:
    """Compact human-readable byte size (e.g. ``4.2 GB``)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _residency_state(src: DataSourceDescriptor) -> str | None:
    """Residency of a source's content, or ``None`` when the server didn't report it.

    Returns ``"resident"`` (content local, cheap to read), ``"remote"`` (not
    local -- remote or dehydrated, slow or blocking to read), or ``None`` when
    ``data_resident`` is unset (a server predating the field; residency unknown,
    so the UI shows no indicator rather than guessing).
    """
    if not src.HasField("data_resident"):
        return None
    return "resident" if src.data_resident else "remote"


def _build_tree(sources: Dict[str, DataSourceDescriptor]) -> _TreeNode:
    """Build hierarchical tree from sources based on source_url paths."""
    root = _TreeNode(node_id="", name="", node_type="folder", depth=0)

    for src in sources.values():
        parts = _get_path_parts(src.source_url)
        if not parts:
            # No path parts, add directly to root
            root.children.append(
                _TreeNode(
                    node_id=src.source_id,
                    name=src.source_id,
                    node_type="source",
                    depth=1,
                    source=src,
                )
            )
            continue

        # Navigate/create folder path
        current = root
        for i in range(len(parts) - 1):
            part = parts[i]
            child = next(
                (
                    c
                    for c in current.children
                    if c.node_type == "folder" and c.name == part
                ),
                None,
            )
            if not child:
                child = _TreeNode(
                    node_id=current.node_id + "/" + part,
                    name=part,
                    node_type="folder",
                    depth=current.depth + 1,
                )
                current.children.append(child)
            current = child

        # Add source as leaf
        source_name = parts[-1]
        current.children.append(
            _TreeNode(
                node_id=src.source_id,
                name=source_name,
                node_type="source",
                depth=current.depth + 1,
                source=src,
            )
        )

    # Sort children: folders first, then sources, both alphabetically
    def sort_children(node: _TreeNode):
        node.children.sort(
            key=lambda c: (0 if c.node_type == "folder" else 1, c.name.lower())
        )
        for child in node.children:
            sort_children(child)

    sort_children(root)

    # Flatten paths: merge folders that have only one folder child
    def flatten_paths(node: _TreeNode):
        for child in node.children:
            if child.node_type == "folder":
                flatten_paths(child)
                # Flatten while single folder child
                while (
                    len(child.children) == 1 and child.children[0].node_type == "folder"
                ):
                    grandchild = child.children[0]
                    child.name = child.name + "/" + grandchild.name
                    child.node_id = grandchild.node_id
                    child.children = grandchild.children
                    for gc in child.children:
                        gc.depth = child.depth + 1
                    flatten_paths(child)

    flatten_paths(root)
    return root


def _filter_tree(
    node: _TreeNode,
    matching_ids: Set[str],
    expanded_folders: Set[str],
) -> _TreeNode | None:
    """Filter tree to show only matching sources, auto-expand folders."""
    if node.node_type == "source":
        if node.node_id in matching_ids:
            return node
        return None

    # Folder: filter children
    filtered_children: List[_TreeNode] = []
    for child in node.children:
        filtered = _filter_tree(child, matching_ids, expanded_folders)
        if filtered:
            filtered_children.append(filtered)
            # Auto-expand folders containing matches
            if filtered.node_type == "source" or filtered.children:
                expanded_folders.add(node.node_id)

    if not filtered_children:
        return None

    result = _TreeNode(
        node_id=node.node_id,
        name=node.name,
        node_type=node.node_type,
        depth=node.depth,
    )
    result.children = filtered_children
    return result


# ==============================================================================
# Metadata Dialog
# ==============================================================================


def _is_empty_for_display(value) -> bool:
    """Check if a value is empty for display purposes.

    Filters out null, empty arrays, empty objects, and nested empty structures.
    """
    if value is None:
        return True
    if isinstance(value, list):
        if not value:
            return True
        return all(_is_empty_for_display(v) for v in value)
    if isinstance(value, dict):
        if not value:
            return True
        return all(_is_empty_for_display(v) for v in value.values())
    return isinstance(value, str) and not value.strip()


def _filter_empty_metadata(metadata: Dict) -> Dict:
    """Filter out empty items from metadata dict."""
    if not metadata:
        return {}

    filtered = {}
    for key, value in metadata.items():
        if not _is_empty_for_display(value):
            filtered[key] = value

    return filtered


class MetadataDialog(QDialog):
    """Dialog to display source and tensor metadata."""

    def __init__(
        self,
        parent: QWidget,
        source: DataSourceDescriptor,
        tensor_id: str | None = None,
        metadata: Dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setMinimumSize(600, 500)
        self.resize(650, 600)

        layout = QVBoxLayout(self)

        # Compact header: source path - shape dtype
        header_layout = QHBoxLayout()

        # Source URL path (use stem)
        url_parts = _get_path_parts(source.source_url)
        url_display = "/" + "/".join(url_parts) if url_parts else source.source_id
        url_label = QLabel(url_display)
        url_label.setStyleSheet("color: #60a5fa; font-weight: bold;")
        header_layout.addWidget(url_label)

        # Tensor info inline
        tensor_desc = None
        if tensor_id:
            tensor_desc = next(
                (t for t in source.tensors if t.array_id == tensor_id), None
            )
        elif len(source.tensors) == 1 and source.tensors[0]:
            tensor_desc = source.tensors[0]

        if tensor_desc:
            header_layout.addWidget(QLabel("—"))
            shape_str = _format_shape(tensor_desc.shape)
            shape_label = QLabel(shape_str)
            shape_label.setStyleSheet("color: #a78bfa;")
            header_layout.addWidget(shape_label)

            dtype_label = QLabel(tensor_desc.dtype)
            dtype_label.setStyleSheet("color: #fbbf24;")
            header_layout.addWidget(dtype_label)

        # Source-level residency badge (omitted when the server didn't report it)
        residency = _residency_state(source)
        if residency == "remote":
            res_label = QLabel(f"{_RESIDENCY_GLYPH} remote")
            res_label.setStyleSheet("color: #888;")
            res_label.setToolTip(_REMOTE_TOOLTIP)
            header_layout.addWidget(res_label)
        elif residency == "resident":
            res_label = QLabel("● local")
            res_label.setStyleSheet("color: #34d399;")
            res_label.setToolTip(_RESIDENT_TOOLTIP)
            header_layout.addWidget(res_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Metadata section
        layout.addWidget(QLabel("Metadata"))
        meta_text = QTextEdit()
        meta_text.setReadOnly(True)
        meta_text.setStyleSheet(
            "QTextEdit { background-color: #1e2435; color: #e2e8f0; font-family: monospace; }"
        )

        if metadata:
            # Filter empty items and format JSON with indentation
            filtered = _filter_empty_metadata(metadata)
            if filtered:
                formatted = json.dumps(filtered, indent=2)
                meta_text.setPlainText(formatted)
            else:
                meta_text.setPlainText("No metadata available")
        else:
            meta_text.setPlainText("No metadata available")

        layout.addWidget(meta_text)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ==============================================================================
# Tensor Browser Widget (Pure Qt)
# ==============================================================================

# Bottom message pane. One label shows all transient feedback and errors; its
# lifecycle is driven by a level:
#   "info"  — a one-shot outcome ("added 3 sources"). Self-clears after
#             _MESSAGE_AUTO_CLEAR_MS so a stale line does not linger (the d&d
#             summary was the motivating case).
#   "busy"  — an ongoing state ("Connecting…", "Indexing…"). Sticky: it reflects
#             a condition, so the flow that started it clears it when the
#             condition resolves, not a timer.
#   "error" — sticky until replaced or explicitly cleared (never times out).
_MESSAGE_AUTO_CLEAR_MS = 6000


def _callout_qss(text_hex: str, accent_hex: str, bg_rgba: str) -> str:
    """A left-accented callout stylesheet, so the pane reads as one component
    whose color alone signals severity (blue = status, red = error)."""
    return (
        "QLabel {"
        f" color: {text_hex};"
        " font-weight: bold;"
        f" background-color: {bg_rgba};"
        f" border-left: 3px solid {accent_hex};"
        " border-radius: 2px;"
        " padding: 4px 8px;"
        " }"
    )


# level -> stylesheet. "busy" shares the blue status look with "info".
_MESSAGE_STYLES = {
    "info": _callout_qss("#93c5fd", "#60a5fa", "rgba(96, 165, 250, 40)"),
    "busy": _callout_qss("#93c5fd", "#60a5fa", "rgba(96, 165, 250, 40)"),
    "error": _callout_qss("#fca5a5", "#ef4444", "rgba(239, 68, 68, 40)"),
}


class TensorBrowserWidget(QWidget):
    """Widget to browse and load tensors from a TensorFlight server."""

    # Emitted (via the connection's on_sources_changed hook) when the background
    # source watcher re-lists the catalog from its daemon thread. A Qt signal —
    # not a direct call or QTimer — because the watcher fires off the Qt main
    # thread; the queued connection marshals the tree rebuild back onto it.
    _sources_changed = Signal(object)

    # Emitted (with the connect generation) from the background connect worker
    # when an auto_connect attempt finishes. A Qt signal — not a direct call —
    # because the worker runs off the Qt main thread; the queued connection
    # marshals the tree render back onto it. See :meth:`_start_connect`.
    _connect_done = Signal(int)

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        connection: TensorConnection | None = None,
        compute_scheduler: str | None = None,
    ):
        super().__init__()
        self._viewer = viewer
        # The data layer is owned by a TensorConnection service that this
        # widget consumes. When constructed standalone (no MCP), build our own.
        self._conn = connection or TensorConnection()
        # When set (MCP context), pin loaded layers' slice reads to a
        # single-process scheduler so the serial viewer shares the main-process
        # chunk cache instead of scattering across the cluster (issue #8). None
        # in the standalone napari plugin -> arrays passed through unchanged.
        self._compute_scheduler = compute_scheduler
        self._selected_source_id: str | None = None
        self._selected_tensor_id: str | None = None
        self._expanded_folders: Set[str] = set()
        # On the first unfiltered render, expand every top-level node so the
        # user lands on the first level of leaves instead of a wall of
        # collapsed roots. Flipped once, then ordinary expand-state tracking
        # (``_expanded_folders``) takes over so later rebuilds respect the
        # user's manual collapses.
        self._initial_expand_done: bool = False

        # Connect runs the shared, non-blocking auto_connect policy on a worker
        # thread (see :meth:`_start_connect`) so the viewer stays responsive and
        # nothing blocks the kernel's Qt loop. ``_connecting`` is True while a
        # worker is in flight (the source watcher skips re-rendering then, to
        # avoid fighting the connect that is about to repaint). ``_connect_gen``
        # is a supersession token: each new connect bumps it so a stale worker's
        # result is dropped when the user retargets a different server (the old
        # local server, if one was launched, is left running).
        self._connecting: bool = False
        self._connect_gen: int = 0

        # Set up widget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # In-flight resolve workers, owned by thread lifetime: a worker is held
        # here from start() until its `finished` fires (then discarded +
        # deleteLater'd). A set, not a single slot, so overlapping resolves can't
        # clobber each other's only ref and get the QThread GC'd / destroyed while
        # still running -- important once a non-modal progress/cancel lets two run.
        self._resolve_workers: set = set()
        # In-flight hydrate-ahead warms, keyed by source_id -> _WarmState. This map
        # is *UI state* -- it answers "is this source warming?" for the context
        # menu / dedup and carries the row progress fraction so a rebuild can
        # re-apply the inline bar. It is popped on warmed/cancelled/failed (which
        # all fire *before* `finished`) so the menu flips back to "Hydrate"
        # promptly; it is therefore NOT the GC owner of the QThread.
        self._warms: Dict[str, _WarmState] = {}
        # GC ownership of warm QThreads, separate from `_warms` and held all the
        # way to `finished` (like `_resolve_workers`). `_warms` drops the worker
        # too early to anchor it through the warmed->finished window, so a strong
        # Python ref must live here or a backend that doesn't keep the wrapper
        # alive could GC/destroy the QThread while it is still running.
        self._warm_retain: set = set()
        # In-flight drag-drop add worker (at most one at a time) plus GC retention
        # to the ``finished`` signal, mirroring the resolve/warm ownership rule.
        self._add_worker: _AddSourceWorker | None = None
        self._add_retain: set = set()
        self._setup_ui()

        # Self-heal the tree when the background source watcher re-lists a
        # catalog that was cached mid-index (issue #44). The watcher runs on a
        # daemon thread, so it reaches the GUI through a queued signal. In the
        # MCP context the kernel bootstrap also starts the watch on this same
        # shared connection; start_source_watch is idempotent, so the duplicate
        # call here (covering the standalone plugin) is harmless.
        self._sources_changed.connect(self._on_sources_changed)
        self._conn.on_sources_changed = self._sources_changed.emit
        self._conn.start_source_watch()

        # Render a background auto_connect's outcome on the Qt main thread.
        self._connect_done.connect(self._on_connect_done)

        # Auto-connect on next event loop tick
        QTimer.singleShot(0, self._auto_connect)

    # The client and source catalog live on the connection service; expose them
    # read-only so the widget's internal read sites stay unchanged.
    @property
    def _client(self):
        return self._conn.client

    @property
    def _sources(self):
        return self._conn.sources

    @property
    def _use_server_query(self):
        return self._conn.use_server_query

    def _setup_ui(self):
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Accept file/folder drops onto the widget (see dragEnterEvent/dropEvent).
        # The affordance and its enablement are surfaced by the drop-hint row
        # below -- a refused drag never reaches dropEvent, so the *reason* must
        # live in this always-visible label, not in a drop-time message.
        self.setAcceptDrops(True)

        # Compact connection summary row: a single clickable line showing the
        # server + state ("server_url — connected") with a trailing disclosure
        # caret. Clicking it toggles the advanced connection controls
        # (URL/token/Connect/Refresh). Those controls are touched once at setup;
        # day-to-day the user only needs to see *that* they are connected, so
        # they are collapsed by default. The caret + pointing-hand cursor are the
        # affordance that the line is expandable (biopb/biopb-mcp).
        self._advanced_expanded = False
        self._status_summary = QLabel()
        self._status_summary.setWordWrap(True)
        self._status_summary.setCursor(Qt.PointingHandCursor)
        self._status_summary.setToolTip("Show/hide connection settings")
        # A QLabel has no clicked signal; route its click straight to the toggle.
        self._status_summary.mousePressEvent = lambda _event: self._toggle_advanced()
        layout.addWidget(self._status_summary)

        # Advanced connection panel — hidden until the summary line is clicked.
        # Holds the server URL, token, and Connect/Refresh controls.
        self._advanced_panel = QWidget()
        adv_layout = QVBoxLayout(self._advanced_panel)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(4)

        # Server URL input (label and input on same row)
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server:"))
        self._server_input = QLineEdit()
        self._server_input.setText(self._conn.url)
        self._server_input.setPlaceholderText("Flight server URL")
        server_layout.addWidget(self._server_input)
        adv_layout.addLayout(server_layout)

        # Token input (label, input, and toggle on same row)
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Token:"))
        self._token_input = QLineEdit()
        if self._conn.token:
            self._token_input.setText(self._conn.token)
        self._token_input.setPlaceholderText("optional")
        self._token_input.setEchoMode(QLineEdit.Password)
        self._show_token_btn = QPushButton("Show")
        self._show_token_btn.setFixedWidth(50)
        self._show_token_btn.clicked.connect(self._toggle_token_visibility)
        token_layout.addWidget(self._token_input)
        token_layout.addWidget(self._show_token_btn)
        adv_layout.addLayout(token_layout)

        # Connect and Refresh buttons
        btn_layout = QHBoxLayout()
        self._connect_button = QPushButton("Connect")
        self._connect_button.clicked.connect(self._on_connect_clicked)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh)
        self._refresh_button.setEnabled(False)
        btn_layout.addWidget(self._connect_button)
        btn_layout.addWidget(self._refresh_button)
        adv_layout.addLayout(btn_layout)

        self._advanced_panel.setVisible(False)
        layout.addWidget(self._advanced_panel)
        self._update_status_summary()

        # Filter input (label and input on same row)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Search sources...")
        self._filter_input.textChanged.connect(self._on_filter_text_changed)
        filter_layout.addWidget(self._filter_input)
        layout.addLayout(filter_layout)

        # Debounce timer for filter
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._apply_filter)

        # Tree widget - give it most of the space
        self._tree_widget = QTreeWidget()
        self._tree_widget.setHeaderHidden(True)
        self._tree_widget.setExpandsOnDoubleClick(False)
        self._tree_widget.setIndentation(12)
        self._tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        self._tree_widget.itemClicked.connect(self._on_tree_item_clicked)
        self._tree_widget.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self._tree_widget.setStyleSheet("QTreeWidget { min-height: 300px; }")
        # Paints the inline hydrate-ahead progress fill behind a source row
        # (biopb/biopb#202) -- replaces the old floating progress dialog.
        self._tree_widget.setItemDelegate(_WarmProgressDelegate(self._tree_widget))
        layout.addWidget(self._tree_widget, stretch=1)

        # Drag-drop affordance row: an always-visible hint reflecting whether a
        # drop is possible right now (connected + localhost), plus a Cancel button
        # shown only while an add is in flight. The hint is the ONLY place a
        # refused drop's reason can be shown (dragEnterEvent -> ignore never
        # reaches dropEvent), and it doubles as the non-modal progress line.
        drop_layout = QHBoxLayout()
        self._drop_hint_label = QLabel()
        self._drop_hint_label.setWordWrap(True)
        self._drop_hint_label.setStyleSheet("color: #888; font-size: 11px;")
        drop_layout.addWidget(self._drop_hint_label, stretch=1)
        self._add_cancel_btn = QPushButton("Cancel")
        self._add_cancel_btn.setFixedWidth(60)
        self._add_cancel_btn.setVisible(False)
        self._add_cancel_btn.clicked.connect(self._cancel_add)
        drop_layout.addWidget(self._add_cancel_btn)
        layout.addLayout(drop_layout)
        self._update_drop_hint()

        # Metadata display
        self._metadata_label = QLabel()
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setStyleSheet("color: #888; font-size: 11px;")
        self._metadata_label.setVisible(False)
        layout.addWidget(self._metadata_label)

        # Bottom message pane: a single callout that carries both transient
        # status/progress ("Indexing…", "added 3 sources") and inline errors,
        # colored by level and self-clearing per the rules on _MESSAGE_STYLES /
        # _MESSAGE_AUTO_CLEAR_MS above. It is styled as a left-accented callout
        # so it stays visually distinct from the grey metadata pane above.
        # _show_status / _show_error / _clear_status / _clear_error route here.
        self._message_level: str | None = None
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_message)
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setVisible(False)
        layout.addWidget(self._message_label)

    def _on_connect_clicked(self, *args):
        """Connect button handler: retarget to the typed URL/token, connect."""
        self._conn.url = self._server_input.text().strip()
        self._conn.token = self._token_input.text().strip() or None
        self._start_connect()

    def _auto_connect(self):
        """Connect on startup using the resolved URL/token (no user prompt)."""
        self._start_connect()

    def _start_connect(self):
        """Run the shared auto-connect policy on a worker thread.

        Delegates to :meth:`TensorConnection.auto_connect` — the same
        non-blocking policy the headless kernel uses (try the URL, wait through a
        ``STARTING`` data-folder scan, and auto-start a local biopb server as a
        last resort when the URL is local and the CLI is installed) — run off the
        Qt main thread. Two reasons it must not run inline: the viewer stays
        responsive while the server binds/scans, and, in the MCP context, the
        widget lives in the kernel whose Qt loop ``start_kernel`` waits on, so a
        blocking connect (or a modal prompt — now gone) here would wedge the
        kernel. Completion is marshaled back via ``_connect_done``; a generation
        token drops the result of any connect the user has since superseded.
        """
        self._clear_error()
        self._clear_status()
        self._tree_widget.clear()
        self._metadata_label.setVisible(False)
        self._selected_source_id = None
        self._selected_tensor_id = None

        self._connect_gen += 1
        gen = self._connect_gen
        self._connecting = True
        self._connect_button.setEnabled(False)
        self._update_status_summary()
        self._show_status(f"Connecting to {self._conn.url}…", sticky=True)

        def _worker():
            # auto_connect is best-effort: it swallows its own failures and
            # records last_status/last_message, so we only signal completion and
            # let _on_connect_done read the outcome off the connection. The
            # except is belt-and-suspenders — always signal, never let the
            # worker thread die with an unhandled exception.
            try:
                self._conn.auto_connect()
            except Exception:
                logger.exception("Connect worker failed")
            finally:
                self._connect_done.emit(gen)

        threading.Thread(target=_worker, name="tbw-connect", daemon=True).start()

    def _on_connect_done(self, gen: int):
        """Render the outcome of a background ``auto_connect`` (main thread).

        Queued from ``_connect_done``. A stale generation (the user retargeted a
        different server while this one was still connecting) is dropped; the
        superseding connect owns the UI.
        """
        if gen != self._connect_gen:
            return
        self._connecting = False
        self._connect_button.setEnabled(True)
        self._clear_status()
        self._update_status_summary()
        self._update_drop_hint()

        if not self._conn.is_connected:
            # auto_connect recorded the friendly reason (down / still starting).
            self._show_error(
                self._conn.last_message
                or f"Cannot reach tensor server at {self._conn.url} — is it running?"
            )
            self._tree_widget.clear()
            self._refresh_button.setEnabled(False)
            return

        sources = self._conn.sources
        if not sources:
            # While the server is still indexing, keep Refresh enabled (more
            # sources are coming, and the watcher re-lists as they appear); a
            # genuinely empty server leaves it disabled, as before.
            indexing = self._show_empty_state()
            self._refresh_button.setEnabled(indexing)
            return

        if self._use_server_query:
            self._filter_input.setPlaceholderText("Search (SQL filter)")
            logger.info(
                "Large catalog (%d sources), server-side SQL filter enabled",
                len(sources),
            )
        else:
            self._filter_input.setPlaceholderText("Search sources...")

        self._build_and_display_tree()
        self._refresh_button.setEnabled(True)

    def _toggle_advanced(self):
        """Show/hide the full connection controls behind the summary line."""
        self._advanced_expanded = not self._advanced_expanded
        self._advanced_panel.setVisible(self._advanced_expanded)
        self._update_status_summary()

    def _update_status_summary(self):
        """Refresh the compact connection summary line.

        Renders ``<url> — <state> <caret>`` with a leading state glyph and a
        trailing disclosure caret, mirroring the live connection state
        (connecting / connected / disconnected) so the user can see they are
        connected without expanding the advanced panel. The caret signals that
        the line is clickable to reveal the connection settings.
        """
        url = self._conn.url or "(no server)"
        if self._connecting:
            glyph, color, state = "◌", "#888", "connecting…"
        elif self._conn.is_connected:
            glyph, color, state = "●", "#4ade80", "connected"
        else:
            glyph, color, state = "○", "#f87171", "disconnected"
        caret = "▾" if self._advanced_expanded else "▸"
        self._status_summary.setText(
            f"<span style='color:{color}'>{glyph}</span> "
            f"<b>{url}</b> — <span style='color:{color}'>{state}</span> "
            f"<span style='color:#888'>{caret}</span>"
        )

    def _toggle_token_visibility(self):
        """Toggle token field visibility between password and normal mode."""
        if self._token_input.echoMode() == QLineEdit.Password:
            self._token_input.setEchoMode(QLineEdit.Normal)
            self._show_token_btn.setText("Hide")
        else:
            self._token_input.setEchoMode(QLineEdit.Password)
            self._show_token_btn.setText("Show")

    def _show_empty_state(self) -> bool:
        """Render the no-sources state, distinguishing indexing from empty.

        With progressive discovery the server reports ``SERVING`` while its
        data-folder scan is still running, so an empty catalog right after
        connect is often "not done indexing yet," not "nothing here." When the
        last-observed health says a full scan is in progress, show a transient
        grey status instead of an error -- the background source watcher re-lists
        the tree automatically as sources are found. Returns True in that case,
        False when the catalog is genuinely empty (an error is shown).
        """
        if self._conn.scan_in_progress():
            self._clear_error()
            self._show_status(
                f"Indexing data folder… "
                f"({self._conn.scan_source_count()} sources so far). "
                "The list updates automatically as sources are found.",
                sticky=True,
            )
            return True
        self._clear_status()
        self._show_error("No sources found on server")
        return False

    def _show_message(self, msg: str, *, level: str, sticky: bool):
        """Show *msg* in the bottom pane at *level* (see _MESSAGE_STYLES).

        Non-sticky messages self-clear after _MESSAGE_AUTO_CLEAR_MS; sticky ones
        persist until replaced or explicitly cleared. Each call restarts (or
        stops) the single-shot timer, so the visible message always owns it.
        """
        self._message_level = level
        self._message_label.setStyleSheet(_MESSAGE_STYLES[level])
        self._message_label.setText(msg)
        self._message_label.setVisible(True)
        self._message_timer.stop()
        if not sticky:
            self._message_timer.start(_MESSAGE_AUTO_CLEAR_MS)

    def _clear_message(self):
        """Clear the bottom pane regardless of level and cancel any timer."""
        self._message_timer.stop()
        self._message_level = None
        self._message_label.setVisible(False)
        self._message_label.setText("")

    def _show_error(self, msg: str):
        """Display an inline error (red, sticky until replaced/cleared)."""
        self._show_message(msg, level="error", sticky=True)

    def _clear_error(self):
        """Clear the pane only if it is currently showing an error.

        Callers sprinkle this before a new action to wipe a stale error; it must
        not knock out a busy/info status set by a concurrent flow (e.g. the
        background "Indexing…" line), so it is scoped to the error level.
        """
        if self._message_level == "error":
            self._clear_message()

    def _report_failure(self, title: str, message: str):
        """Modally report a failed *user-initiated* action (resolve/hydrate/load).

        These actions are explicit, consenting gestures the user actively
        triggered and watched (a modal progress dialog, or a busy cursor during
        load), so their failure deserves an acknowledged modal box rather than
        the easily-missed, transient inline error pane (which is wiped by the
        next refresh/selection). Background errors (connect, refresh, list) stay
        on the inline pane -- issue #206.
        """
        QMessageBox.critical(self, title, message or "Unknown error")

    def _show_status(self, msg: str, *, sticky: bool = False):
        """Display a transient status/progress message (blue callout).

        Pass ``sticky=True`` for an *ongoing* state ("Connecting…", "Indexing…")
        whose owning flow clears it when the state resolves; leave it False for a
        one-shot outcome ("added 3 sources") that should self-clear.
        """
        self._show_message(msg, level="busy" if sticky else "info", sticky=sticky)

    def _clear_status(self):
        """Clear the pane only if it is currently showing a status (not an error)."""
        if self._message_level in ("info", "busy"):
            self._clear_message()

    # ------------------------------------------------------------------
    # Drag-and-drop: add a dropped local file/dir as a source and serve it.
    # ------------------------------------------------------------------

    def _can_accept_drop(self) -> tuple[bool, str]:
        """Whether a drop is possible right now, and the reason to display.

        The reason is the ambient affordance text -- a refused drag never reaches
        ``dropEvent``, so this string (shown in the drop-hint label) is the only
        place the user learns *why* a drop is unavailable. Drops are enabled only
        against a connected, **localhost** server: a dropped path is a client-side
        filesystem path, meaningful to the server only when they share a disk.
        """
        if self._connecting or not self._conn.is_connected:
            return False, "Not connected — connect to add data by drag-drop"
        if not self._conn.is_localhost():
            return False, "Connected to a remote server — drag-drop unavailable"
        if self._add_worker is not None:
            return False, "Adding data…"
        return True, "Drop image files or folders here to add them"

    def _update_drop_hint(self):
        """Refresh the ambient drop-hint text from the current connection state."""
        if self._add_worker is not None:
            return  # a live add owns the label (progress line)
        _, reason = self._can_accept_drop()
        self._drop_hint_label.setText(reason)

    @staticmethod
    def _local_paths_from_mime(mime) -> List[str]:
        """The single local path carried by a drag, wrapped in a list, or [].

        A drop is accepted only when it is *exactly one* local file/folder. A
        multi-item drag is refused (returns ``[]``, so the cursor shows "no
        drop") rather than partially accepted — the add pipeline handles one
        path per drop, and one folder still discovers many datasets in a single
        call, so the common case is unaffected. A non-file URL (e.g. a web link)
        is likewise rejected.
        """
        if not mime.hasUrls():
            return []
        urls = mime.urls()
        if len(urls) != 1:
            return []  # exactly one item per drop; multi-select is refused
        url = urls[0]
        if not url.isLocalFile():
            return []
        return [url.toLocalFile()]

    def dragEnterEvent(self, event):
        """Accept a drag only if it is all-local files onto a localhost server."""
        ok, _ = self._can_accept_drop()
        if ok and self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        """Register the single dropped local path as a source on the server."""
        ok, _ = self._can_accept_drop()
        paths = self._local_paths_from_mime(event.mimeData())
        if not ok or not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        path = paths[0]
        if not self._confirm_large_drop(path):
            return  # user declined scanning an oversized folder; nothing sent
        if not self._confirm_cloud_drop(path):
            return  # user declined adding from a cloud-synced folder
        self._start_add(path)

    def _confirm_large_drop(self, path: str) -> bool:
        """Ask before scanning a big folder; return True to proceed.

        Only a directory over the entry threshold prompts — a file or a small
        folder proceeds silently. The count runs *locally*: the drop UI is
        enabled only against a localhost server (``_can_accept_drop``), so the
        client shares the server's filesystem and can size the tree cheaply
        (short-circuiting at the threshold) before any scan is sent. This is a
        coarse footgun-stopper for dropping a home/root folder by mistake, kept
        client-side so the user stays in control instead of the server hard-
        rejecting; the walk itself still happens server-side once confirmed.
        """
        if not _dir_exceeds_entry_threshold(path):
            return True
        name = os.path.basename(path.rstrip("/")) or path
        resp = QMessageBox.question(
            self,
            "Add large folder?",
            f"“{name}” contains many files. Scan it and add all datasets "
            "found under it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return resp == QMessageBox.Yes

    def _confirm_cloud_drop(self, path: str) -> bool:
        """Warn before adding a source from a cloud-synced (OneDrive) folder.

        OneDrive "Files On-Demand" can evict a file's bytes to the cloud, leaving
        an offline placeholder: indexed fine while resident, but a later read
        recalls it -- slow, or failing outright when offline. The server keeps such
        a source registered across rescans (its walk skips OneDrive; the reconcile
        preserves already-registered claims there), so this is a heads-up about
        read behavior, not a block -- the default action is to proceed. Returns
        True to add, False to cancel.
        """
        warning = _cloud_drop_warning(path)
        if warning is None:
            return True
        resp = QMessageBox.question(
            self,
            "Add source from a cloud folder?",
            warning,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return resp == QMessageBox.Yes

    def _start_add(self, path: str):
        """Spawn the off-GUI-thread add worker for *path* (non-modal)."""
        if self._add_worker is not None:
            return  # one add at a time
        self._clear_error()
        worker = _AddSourceWorker(self._conn, path)
        self._add_worker = worker
        self._add_retain.add(worker)
        worker.progress.connect(self._on_add_progress)
        worker.done.connect(self._on_add_done)
        worker.failed.connect(self._on_add_failed)
        worker.finished.connect(lambda w=worker: self._add_retain.discard(w))
        self._add_cancel_btn.setVisible(True)
        label = os.path.basename(path.rstrip("/")) or path
        self._drop_hint_label.setText(f"Adding {label}…")
        worker.start()

    def _cancel_add(self):
        """Cancel the in-flight add (keeps sources already registered)."""
        if self._add_worker is not None:
            self._add_worker.request_cancel()
            self._drop_hint_label.setText("Cancelling…")

    def _on_add_progress(self, progress):
        """Relay per-source add progress to the drop-hint line (count-up)."""
        count = progress.added_count
        path = progress.current_path or ""
        msg = f"Adding… {count} source{'' if count == 1 else 's'} registered"
        if os.path.isabs(path):
            # A real filesystem path being scanned -> show its basename.
            msg += f" (scanning {os.path.basename(path.rstrip('/'))})"
        elif path:
            # A status sentence (e.g. the catalog-lock wait heartbeat) -> show it
            # verbatim, not run through the "scanning {basename}" label.
            msg += f" ({path})"
        self._drop_hint_label.setText(msg)

    def _on_add_failed(self, msg: str):
        """Whole-request add failure (e.g. path not found on the server)."""
        self._add_worker = None
        self._add_cancel_btn.setVisible(False)
        self._update_drop_hint()
        self._report_failure("Add data failed", msg)

    def _on_add_done(self, payload):
        """Terminal add tally: refresh, summarize, report failures."""
        added, already, failed = payload
        self._add_worker = None
        self._add_cancel_btn.setVisible(False)

        # Prompt sources appear immediately; the background watcher would also
        # catch up, but an explicit refresh is prompt.
        if added or already:
            try:
                self._refresh()
            except Exception:
                logger.exception("refresh after add_source failed")

        parts = []
        if added:
            parts.append(f"added {len(added)}")
        if already:
            parts.append(f"{len(already)} already present")
        if failed:
            parts.append(f"{len(failed)} failed")
        self._show_status("Add data: " + (", ".join(parts) if parts else "nothing"))
        self._update_drop_hint()

        if failed:
            detail = "\n".join(
                f"• {os.path.basename(p.rstrip('/')) or p}: {reason}"
                for p, reason in failed
            )
            self._report_failure("Some items were not added", detail)

    def _refresh(self):
        """Refresh the source list from server."""
        self._clear_error()

        if not self._conn.is_connected:
            self._show_error("Not connected")
            return

        try:
            sources = self._conn.refresh()

            if not sources:
                self._show_empty_state()
                self._tree_widget.clear()
                return

            self._build_and_display_tree()

            if self._use_server_query:
                self._filter_input.setPlaceholderText("Search (SQL filter)")
            else:
                self._filter_input.setPlaceholderText("Search sources...")

        except Exception:
            # A failed re-list on a previously-"connected" server almost always
            # means the server is gone. is_connected doesn't self-revalidate, so
            # without this the status line would stay a stale "connected"; drop
            # the client to make the indicator honest and steer the user to
            # reconnect. Stopgap until a live health signal exists (#319).
            self._conn.mark_disconnected("Lost connection to server")
            self._show_error("Refresh failed — lost connection to server")
            self._refresh_button.setEnabled(False)
            self._update_status_summary()
            logger.exception("Failed to refresh source list")

    def _on_sources_changed(self, sources):
        """Rebuild the tree after the background watcher re-lists (issue #44).

        Runs on the Qt main thread (queued from ``_sources_changed``). The
        connection has already swapped in the fresh catalog, so we just re-render
        — through ``_apply_filter`` so any active search text is preserved — and
        only while connected and not mid-(re)connect, to avoid fighting a
        concurrent connect that is about to repaint anyway.
        """
        if not self._conn.is_connected or self._connecting:
            return
        self._clear_error()
        self._apply_filter()

    def closeEvent(self, event):
        """Detach our source-watch hook, but leave the watcher running.

        The ``TensorConnection`` is shared with the kernel/agent (which reads
        ``sources`` live and starts its own watch on the same connection), so we
        must not stop the watcher when only this widget closes — that would kill
        the agent's self-healing. We just drop our callback so the daemon thread
        stops emitting into this soon-to-be-destroyed widget (issue #44).
        """
        if self._conn.on_sources_changed == self._sources_changed.emit:
            self._conn.on_sources_changed = None
        super().closeEvent(event)

    def _build_and_display_tree(self, filtered_ids: Set[str] | None = None):
        """Build tree from sources and display in widget."""
        self._tree_widget.clear()

        if not self._sources:
            return

        # Build tree
        root = _build_tree(self._sources)

        # Apply filter if provided
        display_tree = root
        if filtered_ids:
            new_expanded = set(self._expanded_folders)
            filtered = _filter_tree(root, filtered_ids, new_expanded)
            if filtered:
                display_tree = filtered
                self._expanded_folders = new_expanded
            else:
                # No matches
                return

        # Populate tree widget
        for child in display_tree.children:
            self._add_tree_node(self._tree_widget, child)

        # First unfiltered render: seed every top-level node as expanded so the
        # first level of leaves is visible up front. Persisted via the normal
        # expand-state set so it survives rebuilds and the user can collapse it.
        if not self._initial_expand_done and not filtered_ids:
            for child in display_tree.children:
                self._expanded_folders.add(child.node_id)
            self._initial_expand_done = True

        # Restore expanded state
        self._restore_expanded_state()
        # Restore the highlighted item: the tree is cleared and rebuilt on every
        # filter/refresh, which drops Qt's current-item even though we still track
        # the logical selection (issue #191).
        self._restore_selection()
        # Re-apply inline hydrate-ahead progress bars: clear() above dropped the
        # per-item _WARM_ROLE, so an in-flight warm's bar must be painted back onto
        # the freshly built row (biopb/biopb#202).
        self._reapply_warm_indicators()

    def _add_tree_node(self, parent, node: _TreeNode):
        """Add a tree node to the widget."""
        item = QTreeWidgetItem(parent)
        item.setData(0, Qt.ItemDataRole.UserRole, node.node_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, node.node_type)

        if node.node_type == "folder":
            item.setText(0, node.name)
            for child in node.children:
                self._add_tree_node(item, child)
        else:
            # Source node
            src = node.source
            assert src is not None
            display_name = node.name
            if len(src.tensors) == 1 and src.tensors[0]:
                # Show shape badge for single tensor
                shape_str = _format_shape(src.tensors[0].shape)
                display_name = f"{node.name}  [{shape_str}]"
            elif len(src.tensors) > 1:
                # Show tensor count
                display_name = f"{node.name}  [{len(src.tensors)} tensors]"

            # Residency indicator: flag non-resident (remote/dehydrated) sources
            # with a leading cloud glyph and greyed text; resident sources stay
            # plain. Both known states get an explanatory tooltip; an unknown
            # state (old server) is left unmarked.
            residency = _residency_state(src)
            if residency == "remote":
                display_name = f"{_RESIDENCY_GLYPH} {display_name}"
                item.setForeground(0, QColor("#888"))
                item.setToolTip(0, _REMOTE_TOOLTIP)
            elif residency == "resident":
                item.setToolTip(0, _RESIDENT_TOOLTIP)

            item.setText(0, display_name)

            # Add nested tensor items for multi-tensor sources
            if len(src.tensors) > 1:
                for tensor in src.tensors:
                    tensor_item = QTreeWidgetItem(item)
                    tensor_item.setData(0, Qt.ItemDataRole.UserRole, tensor.array_id)
                    tensor_item.setData(0, Qt.ItemDataRole.UserRole + 1, "tensor")
                    tensor_item.setData(0, Qt.ItemDataRole.UserRole + 2, src.source_id)
                    tensor_name = _tensor_short_name(tensor.array_id)
                    shape_str = _format_shape(tensor.shape)
                    tensor_item.setText(0, f"{tensor_name}  [{shape_str}]")

    def _restore_expanded_state(self):
        """Restore expanded state for folders."""

        def restore_recursive(item: QTreeWidgetItem):
            item_id = item.data(0, Qt.ItemDataRole.UserRole)
            if item_id in self._expanded_folders:
                item.setExpanded(True)
            for i in range(item.childCount()):
                restore_recursive(item.child(i))

        for i in range(self._tree_widget.topLevelItemCount()):
            restore_recursive(self._tree_widget.topLevelItem(i))

    def _restore_selection(self):
        """Re-highlight the tracked source/tensor in a freshly rebuilt tree.

        Walk the new items, find the one matching ``_selected_source_id`` (and the
        tensor field when one is selected), make it the current item, and scroll it
        into view so a refresh -- notably after resolving a cloud source -- doesn't
        lose the user's place (issue #191). Prefer the tensor child when a field is
        selected, else fall back to the source node.
        """
        if self._selected_source_id is None:
            return

        target_source: QTreeWidgetItem | None = None
        target_tensor: QTreeWidgetItem | None = None

        def find_recursive(item: QTreeWidgetItem):
            nonlocal target_source, target_tensor
            node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if node_type == "tensor":
                if (
                    self._selected_tensor_id is not None
                    and item.data(0, Qt.ItemDataRole.UserRole + 2)
                    == self._selected_source_id
                    and item.data(0, Qt.ItemDataRole.UserRole)
                    == self._selected_tensor_id
                ):
                    target_tensor = item
            elif node_type == "source":
                if item.data(0, Qt.ItemDataRole.UserRole) == self._selected_source_id:
                    target_source = item
            for i in range(item.childCount()):
                find_recursive(item.child(i))

        for i in range(self._tree_widget.topLevelItemCount()):
            find_recursive(self._tree_widget.topLevelItem(i))

        target = target_tensor or target_source
        if target is not None:
            self._tree_widget.setCurrentItem(target)
            self._tree_widget.scrollToItem(target)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int):
        """Handle tree item click."""
        self._clear_error()

        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if node_type == "folder":
            # Toggle expansion on click
            item_id = item.data(0, Qt.ItemDataRole.UserRole)
            expanded = not item.isExpanded()
            item.setExpanded(expanded)
            if expanded:
                self._expanded_folders.add(item_id)
            else:
                self._expanded_folders.discard(item_id)
            self._metadata_label.setVisible(False)
            return

        # Determine selection
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            self._selected_source_id = source_id
            self._selected_tensor_id = tensor_id
        else:
            # Source item clicked
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            self._selected_source_id = source_id
            src = self._sources.get(source_id)
            if src and len(src.tensors) == 1 and src.tensors[0]:
                self._selected_tensor_id = src.tensors[0].array_id
            else:
                self._selected_tensor_id = None

            # Multi-tensor source: toggle its field list on click, like a folder
            if item.childCount() > 0:
                expanded = not item.isExpanded()
                item.setExpanded(expanded)
                if expanded:
                    self._expanded_folders.add(source_id)
                else:
                    self._expanded_folders.discard(source_id)

        self._update_metadata_display()

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int):
        """Handle tree item double-click - add tensor to viewer."""
        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        # Skip folders
        if node_type == "folder":
            return

        # Determine selection and add to viewer
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            self._selected_source_id = source_id
            self._selected_tensor_id = tensor_id
        else:
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            src = self._sources.get(source_id)
            if src and _is_unresolved(src):
                # Cloud / unresolved source: double-click triggers an explicit,
                # consented resolve (downloads the file), not a viewer add.
                self._resolve_source(source_id)
                return
            if src and len(src.tensors) == 1 and src.tensors[0]:
                self._selected_source_id = source_id
                self._selected_tensor_id = src.tensors[0].array_id
            else:
                # Multi-tensor source - don't add on double-click
                return

        self._add_to_viewer()

    def _show_context_menu(self, pos):
        """Show context menu for tree items."""
        item = self._tree_widget.itemAt(pos)
        if not item:
            return

        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        # Skip folders
        if node_type == "folder":
            return

        # Determine selection for menu actions
        if node_type == "tensor":
            tensor_id = item.data(0, Qt.ItemDataRole.UserRole)
            source_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
            is_multi_tensor_source = False
            is_unresolved_source = False
        else:
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            src = self._sources.get(source_id)
            is_unresolved_source = src is not None and _is_unresolved(src)
            if src and len(src.tensors) == 1 and src.tensors[0]:
                tensor_id = src.tensors[0].array_id
                is_multi_tensor_source = False
            else:
                # Multi-tensor or unresolved source
                tensor_id = None
                is_multi_tensor_source = src is not None and len(src.tensors) > 1

        menu = QMenu(self)

        # Primary action: Resolve (unresolved/cloud), "View all" (multi-tensor),
        # or "View" (single tensor).
        if is_unresolved_source:
            resolve_action = menu.addAction("Resolve (downloads file)…")
            resolve_action.triggered.connect(lambda: self._resolve_source(source_id))
        elif is_multi_tensor_source:
            view_action = menu.addAction("View all")
            view_action.triggered.connect(lambda: self._view_all_tensors(source_id))
        else:
            view_action = menu.addAction("View")
            if tensor_id:
                view_action.triggered.connect(
                    lambda: self._view_tensor(source_id, tensor_id)
                )
            else:
                view_action.setEnabled(False)

        # Hydrate-ahead: while a warm is in flight on this source, offer to cancel
        # it (the inline row bar is the only other affordance); otherwise offer to
        # (re)start it on any resolved multi-file source. Harmless to re-run
        # (idempotent); a single-file source is excluded (warm is a no-op there).
        menu_src = self._sources.get(source_id)
        if source_id in self._warms:
            cancel_action = menu.addAction("Cancel hydration")
            cancel_action.triggered.connect(lambda: self._cancel_warm(source_id))
        elif menu_src is not None and _is_multifile_source(menu_src):
            warm_action = menu.addAction("Hydrate all files…")
            warm_action.triggered.connect(lambda: self._warm_source(source_id))

        # Metadata action
        meta_action = menu.addAction("Metadata")
        meta_action.triggered.connect(
            lambda: self._show_metadata_dialog(source_id, tensor_id)
        )

        menu.exec_(self._tree_widget.mapToGlobal(pos))

    def _resolve_source(self, source_id: str):
        """Warn, then resolve an unresolved (cloud) source off the GUI thread.

        Resolving downloads the source's whole file, so we (1) take explicit
        consent via a modal warning, then (2) run the blocking resolve in a worker
        thread behind a modal progress dialog — the user is blocked from other
        actions but the UI stays painted — and (3) on success repopulate the tree
        from the now-resolved field list. The repopulate is necessary because
        resolution does not change the server ``source_count``, so the background
        watcher won't pick it up (issue #44); we refresh explicitly here.
        """
        src = self._sources.get(source_id)
        if not src or not self._conn.is_connected:
            return

        parts = _get_path_parts(src.source_url)
        name = parts[-1] if parts else source_id

        confirm = QMessageBox.warning(
            self,
            "Resolve cloud source",
            f"Resolving “{name}” downloads the entire file from remote "
            f"storage.\n\nThis may take several minutes, use local disk space, "
            f"and will not work offline. Continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if confirm != QMessageBox.Ok:
            return

        self._clear_error()

        # Modal, indeterminate (no byte-level progress available), with a working
        # Cancel button. The label is refreshed from server heartbeats with
        # elapsed time + target size so the user can judge whether to wait. We
        # manage close ourselves (autoClose/autoReset off) so that hitting Cancel
        # shows a "Cancelling…" state and the dialog stays up until the worker
        # confirms the stop — which also blocks a second resolve in the meantime.
        progress = QProgressDialog(f"Resolving “{name}”…", "Cancel", 0, 0, self)
        progress.setWindowTitle("Resolving")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        worker = _ResolveWorker(self._conn, source_id)
        # Own the worker by its thread lifetime (held until `finished`), so a
        # later/overlapping resolve can't drop its only ref and have the QThread
        # destroyed mid-run.
        self._resolve_workers.add(worker)

        def _on_progress(p):
            elapsed = int(p.elapsed_seconds)
            label = f"Resolving “{p.target_name or name}”… {elapsed}s"
            if p.target_bytes:
                label += f" ({_human_bytes(p.target_bytes)})"
            progress.setLabelText(label)

        def _on_resolved(descriptor):
            progress.close()
            # Pin the just-resolved source as the logical selection so the rebuild
            # below re-highlights it and scrolls it into view -- otherwise the user
            # loses track of it in the refreshed list (issue #191). Set here (not
            # only on click) because double-click/context-menu resolve can fire
            # without a prior single-click selection. Drop any tensor sub-selection
            # so we land on the source node itself.
            self._selected_source_id = source_id
            self._selected_tensor_id = None
            # The connection snapshot was refreshed by resolve_source(); re-render
            # through _apply_filter so any active search text is preserved and the
            # resolved source now shows its shape badge / field children.
            self._apply_filter()
            # A multi-file source's member data files are still dehydrated -- they
            # recall lazily (and slowly) onto the read path. The user just asked
            # for this source explicitly, so start hydrating ahead in the
            # background now without a second confirmation (biopb/biopb#202).
            # _warm_source shows progress as an inline bar on the source's row
            # (no dialog); a source whose files resolve already recalled (e.g. a
            # TIFF sequence) warms to a near-instant no-op and the bar barely
            # appears. Cancel / re-trigger is available via the context menu.
            if descriptor is not None and _is_multifile_source(descriptor):
                self._warm_source(source_id)

        def _on_failed(message):
            progress.close()
            self._report_failure("Resolve failed", message)

        def _on_cancelled():
            # User-initiated stop; the server recall continues + caches, so a
            # later resolve coalesces. Close quietly, no error banner.
            progress.close()

        def _on_cancel_clicked():
            # Cooperative request; keep the dialog up showing "Cancelling…" until
            # the worker confirms (within one heartbeat) via the cancelled signal.
            progress.setLabelText(f"Cancelling “{name}”…")
            worker.request_cancel()

        worker.progress.connect(_on_progress)
        worker.resolved.connect(_on_resolved)
        worker.failed.connect(_on_failed)
        worker.cancelled.connect(_on_cancelled)
        progress.canceled.connect(_on_cancel_clicked)
        worker.finished.connect(lambda: self._resolve_workers.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        # Modal: blocks here (event loop still runs, dialog stays painted) until a
        # slot above calls progress.close(). The worker's queued signal is
        # delivered inside this exec_, so a fast finish can't deadlock.
        progress.exec_()

    def _find_source_item(self, source_id: str) -> QTreeWidgetItem | None:
        """Return the ``"source"`` tree item for ``source_id`` (or None).

        Walks the freshly built tree the same way :meth:`_restore_selection`
        does; used to paint/clear a row's inline hydrate-ahead progress bar.
        """
        found: QTreeWidgetItem | None = None

        def walk(item: QTreeWidgetItem):
            nonlocal found
            if found is not None:
                return
            if (
                item.data(0, Qt.ItemDataRole.UserRole + 1) == "source"
                and item.data(0, Qt.ItemDataRole.UserRole) == source_id
            ):
                found = item
                return
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self._tree_widget.topLevelItemCount()):
            walk(self._tree_widget.topLevelItem(i))
        return found

    def _set_warm_indicator(self, source_id: str, fraction: float | None):
        """Set (or clear) the inline hydrate-ahead progress bar for a source.

        Records ``fraction`` on the live :class:`_WarmState` (so a tree rebuild
        can re-apply it) and writes it onto the source's tree item at
        :data:`_WARM_ROLE`, which repaints just that row via the delegate.
        ``fraction is None`` removes the bar. Safe if the row isn't currently
        materialized (e.g. filtered out) -- the state still carries the value.
        """
        state = self._warms.get(source_id)
        if state is not None:
            state.fraction = fraction
        item = self._find_source_item(source_id)
        if item is not None:
            item.setData(0, _WARM_ROLE, fraction)

    def _reapply_warm_indicators(self):
        """Repaint every in-flight warm's bar after a tree clear/rebuild."""
        for source_id, state in self._warms.items():
            item = self._find_source_item(source_id)
            if item is not None:
                item.setData(0, _WARM_ROLE, state.fraction)

    def _cancel_warm(self, source_id: str):
        """Cancel an in-flight hydrate-ahead warm (from the context menu).

        Cooperative: the worker closes the stream and the server stops the
        recall; the ``cancelled`` signal then clears the inline bar.
        """
        state = self._warms.get(source_id)
        if state is not None:
            state.worker.request_cancel()

    def _warm_source(self, source_id: str):
        """Hydrate-ahead a resolved multi-file source's member files.

        Runs the blocking warm in a background worker (the user keeps
        browsing/viewing while the server recalls files) and surfaces progress as
        an *inline bar painted across the source's tree row* rather than a
        floating dialog (biopb/biopb#202). Cancel is offered from the context
        menu (:meth:`_cancel_warm`). Idempotent: re-triggering while a warm is
        already in flight is a no-op, and re-triggering after a cancel just
        finishes the remainder; several sources can warm at once.

        A source whose member files were already recalled by resolve (e.g. a TIFF
        sequence, whose construction opens every file) warms to a near-instant
        server no-op -- the bar flicks on and off too fast to notice -- while a
        slow cloud chunk recall (zarr / ome-zarr) fills the row as it progresses.
        """
        src = self._sources.get(source_id)
        if not src or not self._conn.is_connected:
            return
        if source_id in self._warms:  # already hydrating -- don't double-start
            return

        self._clear_error()

        worker = _WarmWorker(self._conn, source_id)
        self._warms[source_id] = _WarmState(worker=worker)  # UI state
        self._warm_retain.add(worker)  # GC owner, held until `finished`
        # Indeterminate until the first server count arrives.
        self._set_warm_indicator(source_id, _WARM_INDETERMINATE)

        def _on_progress(p):
            if p.bytes_total:
                fraction = p.bytes_done / p.bytes_total
            elif p.files_total:
                fraction = p.files_done / p.files_total
            else:
                fraction = _WARM_INDETERMINATE
            self._set_warm_indicator(source_id, fraction)

        def _finish():
            self._warms.pop(source_id, None)
            self._set_warm_indicator(source_id, None)  # remove the bar

        def _on_failed(message):
            # Hydrate is a background, unintrusive operation (inline row bar, no
            # modal progress), so its failure stays on the inline error pane to
            # match -- unlike the explicit, modal resolve/load paths which report
            # via _report_failure (biopb/biopb#206).
            _finish()
            self._show_error(f"Hydrate failed: {message}")

        worker.progress.connect(_on_progress)
        worker.warmed.connect(lambda _done: _finish())
        worker.cancelled.connect(_finish)
        worker.failed.connect(_on_failed)
        # Release the GC ref and schedule C++ deletion only once the thread has
        # actually finished -- _finish() above drops the earlier `_warms` entry,
        # so `_warm_retain` is what keeps the QThread alive in between.
        worker.finished.connect(lambda: self._warm_retain.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _view_tensor(self, source_id: str, tensor_id: str):
        """Add single tensor to viewer."""
        self._selected_source_id = source_id
        self._selected_tensor_id = tensor_id
        self._add_to_viewer()

    def _view_all_tensors(self, source_id: str):
        """Add all tensors from a source to viewer."""
        src = self._sources.get(source_id)
        if not src or not self._client:
            return

        url_parts = _get_path_parts(src.source_url)
        stem = url_parts[-1] if url_parts else source_id

        # Show busy cursor during loading
        QApplication.setOverrideCursor(Qt.BusyCursor)

        try:
            for tensor in src.tensors:
                try:
                    tensor_name = _tensor_short_name(tensor.array_id)
                    layer_name = f"{stem}/{tensor_name}"
                    # Shared build-pyramid -> wrap -> OME scale -> add_image
                    # pipeline (also used by the MCP add_tensor).
                    add_tensor_layer(
                        self._viewer,
                        self._client,
                        source_id,
                        tensor.array_id,
                        tensor,
                        name=layer_name,
                        source_desc=src,
                        compute_scheduler=self._compute_scheduler,
                    )
                    logger.info(
                        "Added tensor layer '%s' from source '%s'",
                        layer_name,
                        source_id,
                    )
                except Exception:
                    logger.exception("Failed to load tensor %s", tensor.array_id)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_metadata_dialog(self, source_id: str, tensor_id: str | None):
        """Show metadata dialog for source/tensor."""
        src = self._sources.get(source_id)
        if not src:
            return

        # Fetch metadata from server
        metadata = None
        if self._client:
            try:
                metadata = self._client.get_source_metadata(source_id)
            except Exception:
                logger.warning("Failed to fetch metadata for %s", source_id)

        dialog = MetadataDialog(self, src, tensor_id, metadata)
        dialog.exec_()

    def _update_metadata_display(self):
        """Update metadata display for selected tensor."""
        if not self._selected_tensor_id or not self._selected_source_id:
            self._metadata_label.setVisible(False)
            return

        src = self._sources.get(self._selected_source_id)
        if not src:
            self._metadata_label.setVisible(False)
            return

        # Find tensor descriptor
        tensor_desc = next(
            (t for t in src.tensors if t.array_id == self._selected_tensor_id),
            None,
        )
        if not tensor_desc:
            self._metadata_label.setVisible(False)
            return

        shape_str = _format_shape(tensor_desc.shape)
        dims_str = (
            ", ".join(tensor_desc.dim_labels) if tensor_desc.dim_labels else "N/A"
        )
        chunks_str = _format_shape(tensor_desc.chunk_shape)
        meta_text = (
            f"Source: {self._selected_source_id}\n"
            f"Tensor: {self._selected_tensor_id}\n"
            f"Shape: {shape_str}\n"
            f"Dtype: {tensor_desc.dtype}\n"
            f"Dims: {dims_str}\n"
            f"Chunks: {chunks_str}"
        )
        self._metadata_label.setText(meta_text)
        self._metadata_label.setVisible(True)

    def _on_filter_text_changed(self, _text: str):
        """Handle filter text change with debounce."""
        self._filter_timer.start(300)  # 300ms debounce

    def _apply_filter(self):
        """Apply the current filter to the tree."""
        query = self._filter_input.text().strip().lower()

        if not query:
            # Clear filter
            self._build_and_display_tree()
            return

        if self._use_server_query and self._client:
            # Server-side SQL query for large catalogs
            self._apply_server_filter(query)
        else:
            # Client-side filter
            self._apply_client_filter(query)

    def _apply_server_filter(self, query: str):
        """Apply server-side SQL filter for large catalogs."""
        try:
            # Escape SQL special characters
            escaped = query.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
            sql = (
                f"SELECT source_id FROM sources WHERE "
                f"LOWER(source_id) LIKE '%{escaped}%' OR "
                f"LOWER(source_url) LIKE '%{escaped}%' OR "
                f"LOWER(source_type) LIKE '%{escaped}%'"
            )
            table = self._client.query_sources(sql)
            ids = {row["source_id"].as_py() for row in table.to_batches()}
            self._build_and_display_tree(filtered_ids=ids)
        except Exception:
            logger.exception("Server filter failed")
            # Fall back to client-side filter
            self._apply_client_filter(query)

    def _apply_client_filter(self, query: str):
        """Apply client-side filter."""
        matching_ids: Set[str] = set()
        for src in self._sources.values():
            hay = f"{src.source_id} {src.source_url} {src.source_type}".lower()
            if query in hay:
                matching_ids.add(src.source_id)

        self._build_and_display_tree(filtered_ids=matching_ids)

    def _add_to_viewer(self):
        """Add selected tensor as dask array to viewer."""
        self._clear_error()

        if self._client is None:
            self._show_error("Not connected to server")
            return

        if not self._selected_source_id or not self._selected_tensor_id:
            self._show_error("No tensor selected")
            return

        src = self._sources.get(self._selected_source_id)
        if not src:
            self._show_error("Source not found")
            return

        # Find tensor descriptor
        tensor_desc = next(
            (t for t in src.tensors if t.array_id == self._selected_tensor_id),
            None,
        )
        if not tensor_desc:
            self._show_error("Tensor descriptor not found")
            return

        try:
            # Show busy cursor during loading
            QApplication.setOverrideCursor(Qt.BusyCursor)

            # Build layer name: source_url.stem[/tensor_short_name]
            url_parts = _get_path_parts(src.source_url)
            stem = url_parts[-1] if url_parts else self._selected_source_id

            if len(src.tensors) == 1:
                layer_name = stem
            else:
                tensor_name = _tensor_short_name(self._selected_tensor_id)
                layer_name = f"{stem}/{tensor_name}"

            # Shared build-pyramid -> wrap -> OME scale -> add_image pipeline
            # (also used by the MCP add_tensor).
            add_tensor_layer(
                self._viewer,
                self._client,
                self._selected_source_id,
                self._selected_tensor_id,
                tensor_desc,
                name=layer_name,
                source_desc=src,
                compute_scheduler=self._compute_scheduler,
            )
            logger.info(
                "Added tensor layer '%s' from source '%s'",
                layer_name,
                self._selected_source_id,
            )

        except Exception:
            self._report_failure("Load failed", "Failed to load tensor")
            logger.exception(
                "Failed to get tensor %s from %s",
                self._selected_tensor_id,
                self._selected_source_id,
            )
        finally:
            QApplication.restoreOverrideCursor()
