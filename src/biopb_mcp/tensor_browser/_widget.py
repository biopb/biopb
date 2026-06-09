"""Tensor browser widget for napari.

Provides a tree-based UI to browse biopb.tensor datastore catalog and add
selected tensors as dask arrays to the napari viewer. Supports authentication
tokens and search filtering.

Uses pure Qt for complex UI (tree widget, custom layouts).
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Set
from urllib.parse import urlparse

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .._connection import ServerStarting, TensorConnection
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
    """Extract path parts from source_url."""
    if not url:
        return []
    try:
        parsed = urlparse(url)
        return [p for p in parsed.path.split("/") if p]
    except Exception:
        return [p for p in url.split("/") if p]


def _format_shape(shape: List[int]) -> str:
    """Format shape as compact string."""
    return "×".join(str(s) for s in shape)


def _tensor_short_name(array_id: str) -> str:
    """Get short name for tensor from its array_id."""
    parts = [p for p in array_id.split("/") if p]
    return parts[-1] if parts else array_id


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
                    len(child.children) == 1
                    and child.children[0].node_type == "folder"
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
        url_display = (
            "/" + "/".join(url_parts) if url_parts else source.source_id
        )
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


class TensorBrowserWidget(QWidget):
    """Widget to browse and load tensors from a TensorFlight server."""

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

        # Non-blocking connect poller state (issue #12). The connect runs on the
        # Qt main thread, so a STARTING (booting) server is polled via a
        # self-rescheduling single-shot timer with capped exponential backoff
        # rather than a blocking wait that would freeze the viewer.
        self._connect_url: str = ""
        self._connect_token: str | None = None
        self._offer_autostart_on_fail: bool = False
        self._connect_backoff_ms: int = 500
        # Monotonic deadline until which a *connection failure* is tolerated as
        # "still booting" (set only while waiting on a server we just launched);
        # None means a down server fails fast.
        self._connect_boot_deadline: float | None = None
        # Supersession token: each new connect bumps it so any in-flight poll
        # chain (e.g. an autostart boot wait) stops when the user retargets a
        # different server. The old local server is left running.
        self._connect_gen: int = 0

        # Set up widget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._setup_ui()

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

        # Server URL input (label and input on same row)
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server:"))
        self._server_input = QLineEdit()
        self._server_input.setText(self._conn.url)
        self._server_input.setPlaceholderText("Flight server URL")
        server_layout.addWidget(self._server_input)
        layout.addLayout(server_layout)

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
        layout.addLayout(token_layout)

        # Connect and Refresh buttons
        btn_layout = QHBoxLayout()
        self._connect_button = QPushButton("Connect")
        self._connect_button.clicked.connect(self._on_connect_clicked)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh)
        self._refresh_button.setEnabled(False)
        btn_layout.addWidget(self._connect_button)
        btn_layout.addWidget(self._refresh_button)
        layout.addLayout(btn_layout)

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
        self._tree_widget.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self._tree_widget.itemClicked.connect(self._on_tree_item_clicked)
        self._tree_widget.itemDoubleClicked.connect(
            self._on_tree_item_double_clicked
        )
        self._tree_widget.setStyleSheet("QTreeWidget { min-height: 300px; }")
        layout.addWidget(self._tree_widget, stretch=1)

        # Metadata display
        self._metadata_label = QLabel()
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setStyleSheet("color: #888; font-size: 11px;")
        self._metadata_label.setVisible(False)
        layout.addWidget(self._metadata_label)

        # Status/progress display (e.g. "server starting — scanning…"); grey,
        # distinct from the red error label below.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #888;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Error display
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #d32f2f;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

    def _on_connect_clicked(self, *args):
        """Connect button handler: connect without offering autostart on fail."""
        self._offer_autostart_on_fail = False
        self._connect()

    def _auto_connect(self):
        """Attempt to connect using resolved URL/token.

        Connect runs as a non-blocking poller (see :meth:`_connect`); if it
        ends up unreachable, the terminal "down" branch offers to start a local
        biopb server as a last resort (see ``_maybe_offer_start_server``).
        """
        self._offer_autostart_on_fail = True
        self._connect()

    def _maybe_offer_start_server(self):
        """Offer to start a local biopb server when startup connect failed.

        Last-resort fallback: if the configured URL is local and the ``biopb``
        CLI is installed, ask the user whether to launch a server for them and,
        if so, start it and reconnect. Skipped in headless/offscreen sessions
        (CI, tests) where there is no one to answer the prompt.
        """
        if QApplication.platformName() == "offscreen":
            return
        if not self._conn.can_autostart_server():
            return

        reply = QMessageBox.question(
            self,
            "Start biopb server?",
            f"Could not connect to {self._conn.url}.\n\n"
            "The 'biopb' command-line tool is installed. Start a local "
            "tensor server now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Launch the daemon (quick — it detaches immediately), then hand off to
        # the non-blocking poller so the viewer stays responsive while the
        # server binds and scans its data folder. The user may type a different
        # server URL and click Connect during the wait; that supersedes this
        # poll (the local server is left running). See issue #12.
        self._clear_error()
        timeout = self._conn.server_start_timeout()
        try:
            self._conn.launch_local_server(startup_timeout=timeout)
        except Exception as exc:
            # Launching a server we were asked to start failing is unexpected:
            # show the underlying cause (e.g. port already in use) inline and
            # also let it propagate to napari's notification_manager. See
            # docs/troubleshooting.md for remedies.
            self._show_error(f"Failed to start local biopb server: {exc}")
            raise

        self._server_input.setText(self._conn.url)
        self._connect_url = self._conn.url
        self._connect_token = self._conn.token
        # Tolerate connection-refused as "still booting" until this deadline.
        self._connect_boot_deadline = time.monotonic() + timeout
        # Don't re-offer autostart from inside the boot poll.
        self._offer_autostart_on_fail = False
        self._show_status("Starting local biopb server…")
        self._begin_poll()

    def _toggle_token_visibility(self):
        """Toggle token field visibility between password and normal mode."""
        if self._token_input.echoMode() == QLineEdit.Password:
            self._token_input.setEchoMode(QLineEdit.Normal)
            self._show_token_btn.setText("Hide")
        else:
            self._token_input.setEchoMode(QLineEdit.Password)
            self._show_token_btn.setText("Show")

    def _show_error(self, msg: str):
        """Display error message."""
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    def _clear_error(self):
        """Clear error message."""
        self._error_label.setVisible(False)
        self._error_label.setText("")

    def _show_status(self, msg: str):
        """Display a transient status/progress message (grey, non-error)."""
        self._status_label.setText(msg)
        self._status_label.setVisible(True)

    def _clear_status(self):
        """Clear the status/progress message."""
        self._status_label.setVisible(False)
        self._status_label.setText("")

    def _connect(self):
        """Begin connecting to the server in the input field (non-blocking).

        Wired (via :meth:`_on_connect_clicked`) to the Connect button and called
        by :meth:`_auto_connect`. Reads the URL/token from the fields, drops out
        of any boot wait (a down server fails fast here), and starts a fresh
        poll chain — superseding any in-flight one, so the user can retarget a
        different server mid-wait (issue #12).
        """
        self._clear_error()
        self._clear_status()
        self._tree_widget.clear()
        self._metadata_label.setVisible(False)
        self._selected_source_id = None
        self._selected_tensor_id = None

        self._connect_url = self._server_input.text().strip()
        self._connect_token = self._token_input.text().strip() or None
        self._connect_boot_deadline = None
        self._begin_poll()

    def _begin_poll(self):
        """(Re)start the poll chain, superseding any in-flight one."""
        self._connect_gen += 1
        # Reset backoff so a fresh connect/retry starts snappy.
        self._connect_backoff_ms = 500
        self._connect_tick(self._connect_gen)

    def _rearm(self, gen: int):
        """Schedule the next poll tick with capped exponential backoff."""
        QTimer.singleShot(
            self._connect_backoff_ms, lambda: self._connect_tick(gen)
        )
        self._connect_backoff_ms = min(self._connect_backoff_ms * 2, 5000)

    def _connect_tick(self, gen: int):
        """One connect attempt; re-arms itself while the server is starting."""
        if gen != self._connect_gen:
            # Superseded by a newer connect (e.g. the user retargeted a
            # different server). Stop this stale chain.
            return
        try:
            # Connection service owns the client/sources and persists the URL.
            sources = self._conn.connect(
                self._connect_url, self._connect_token
            )
        except ServerStarting:
            # Server is up but still scanning its data folder: keep waiting
            # with progress feedback and capped exponential backoff.
            self._show_status(self._conn.last_message)
            self._rearm(gen)
            return
        except Exception as exc:
            # Connection failed. While waiting on a server we just launched,
            # tolerate this (the daemon may not have bound its port yet) until
            # the boot deadline, then give up. Otherwise fail fast.
            if self._connect_boot_deadline is not None:
                if time.monotonic() < self._connect_boot_deadline:
                    self._show_status("Starting local biopb server…")
                    self._rearm(gen)
                    return
                # Boot timed out: this is unexpected (we launched it
                # ourselves), so surface the underlying cause and let napari's
                # notification_manager show it.
                self._connect_boot_deadline = None
                self._clear_status()
                self._show_error(
                    f"Local biopb server did not become ready in time: {exc}"
                )
                self._tree_widget.clear()
                self._refresh_button.setEnabled(False)
                raise
            # An ordinary unreachable server is an expected condition, not an
            # error to surface to napari: show a friendly inline hint instead.
            self._clear_status()
            self._show_error(
                f"Cannot reach tensor server at {self._connect_url} — "
                "is it running?"
            )
            logger.info("Tensor server unreachable at %s", self._connect_url)
            self._tree_widget.clear()
            self._refresh_button.setEnabled(False)
            if self._offer_autostart_on_fail:
                self._maybe_offer_start_server()
            return

        # Connected.
        self._connect_boot_deadline = None
        self._clear_status()
        if self._use_server_query:
            self._filter_input.setPlaceholderText("Search (SQL filter)")
        else:
            self._filter_input.setPlaceholderText("Search sources...")

        if not sources:
            self._show_error("No sources found on server")
            self._refresh_button.setEnabled(False)
            return

        self._build_and_display_tree()
        self._refresh_button.setEnabled(True)

        if self._use_server_query:
            logger.info(
                "Large catalog (%d sources), server-side SQL filter enabled",
                len(sources),
            )

    def _refresh(self):
        """Refresh the source list from server."""
        self._clear_error()

        if not self._conn.is_connected:
            self._show_error("Not connected")
            return

        try:
            sources = self._conn.refresh()

            if not sources:
                self._show_error("No sources found on server")
                self._tree_widget.clear()
                return

            self._build_and_display_tree()

            if self._use_server_query:
                self._filter_input.setPlaceholderText("Search (SQL filter)")
            else:
                self._filter_input.setPlaceholderText("Search sources...")

        except Exception:
            self._show_error("Refresh failed")
            logger.exception("Failed to refresh source list")

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

        # Restore expanded state
        self._restore_expanded_state()

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
            item.setText(0, display_name)

            # Add nested tensor items for multi-tensor sources
            if len(src.tensors) > 1:
                for tensor in src.tensors:
                    tensor_item = QTreeWidgetItem(item)
                    tensor_item.setData(
                        0, Qt.ItemDataRole.UserRole, tensor.array_id
                    )
                    tensor_item.setData(
                        0, Qt.ItemDataRole.UserRole + 1, "tensor"
                    )
                    tensor_item.setData(
                        0, Qt.ItemDataRole.UserRole + 2, src.source_id
                    )
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

        self._update_metadata_display()

    def _on_tree_item_double_clicked(
        self, item: QTreeWidgetItem, _column: int
    ):
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
        else:
            source_id = item.data(0, Qt.ItemDataRole.UserRole)
            src = self._sources.get(source_id)
            if src and len(src.tensors) == 1 and src.tensors[0]:
                tensor_id = src.tensors[0].array_id
                is_multi_tensor_source = False
            else:
                # Multi-tensor source
                tensor_id = None
                is_multi_tensor_source = (
                    src is not None and len(src.tensors) > 1
                )

        menu = QMenu(self)

        # View action - single tensor or "View all" for multi-tensor source
        if is_multi_tensor_source:
            view_action = menu.addAction("View all")
            view_action.triggered.connect(
                lambda: self._view_all_tensors(source_id)
            )
        else:
            view_action = menu.addAction("View")
            if tensor_id:
                view_action.triggered.connect(
                    lambda: self._view_tensor(source_id, tensor_id)
                )
            else:
                view_action.setEnabled(False)

        # Metadata action
        meta_action = menu.addAction("Metadata")
        meta_action.triggered.connect(
            lambda: self._show_metadata_dialog(source_id, tensor_id)
        )

        menu.exec_(self._tree_widget.mapToGlobal(pos))

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
                    logger.exception(
                        "Failed to load tensor %s", tensor.array_id
                    )
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
            ", ".join(tensor_desc.dim_labels)
            if tensor_desc.dim_labels
            else "N/A"
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
            escaped = (
                query.replace("'", "''")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
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
            self._show_error("Failed to load tensor")
            logger.exception(
                "Failed to get tensor %s from %s",
                self._selected_tensor_id,
                self._selected_source_id,
            )
        finally:
            QApplication.restoreOverrideCursor()
