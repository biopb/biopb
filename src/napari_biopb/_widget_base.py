import logging
import threading
from typing import TYPE_CHECKING

import grpc
from magicgui.widgets import (
    ComboBox,
    Container,
    Label,
    ProgressBar,
    create_widget,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QSizePolicy

from ._config import load_config, save_config

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


def _format_error_message(exc: Exception) -> str:
    """Format exception into a succinct error message for widget display.

    For gRPC errors, extracts status code name and first line of details.
    For other errors, returns the exception type and message (truncated).

    Args:
        exc: The exception to format

    Returns:
        Succinct error message string with HTML formatting for status part
    """
    if isinstance(exc, grpc.RpcError):
        # gRPC errors have code() and details()
        status_name = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
        details = exc.details() if hasattr(exc, "details") else str(exc)
        # Take first line, strip actual/literal newlines, truncate to ~100 chars
        first_line = details.split("\n")[0].replace("\\n", " ").strip()
        if len(first_line) > 100:
            first_line = first_line[:100] + "..."
        return (
            f'<span style="color: #d32f2f; font-weight: bold;">{status_name}</span>: '
            f"{first_line}"
        )
    else:
        # Non-gRPC error: show type and truncated message
        exc_type = type(exc).__name__
        msg = str(exc).split("\n")[0].replace("\\n", " ").strip()
        if len(msg) > 100:
            msg = msg[:100] + "..."
        return (
            f'<span style="color: #d32f2f; font-weight: bold;">{exc_type}</span>: '
            f"{msg}"
        )


def _make_full_width(widget: Label) -> None:
    """Configure a Label widget to span full width by hiding its label column.

    When widgets are added to a Container with labels=True, each widget is wrapped
    in a _LabeledWidget with a horizontal layout containing label + value. This
    function hides the label portion so the value spans the full width.

    Args:
        widget: A Label widget that has been added to a Container
    """
    lw_ref = getattr(widget, "_labeled_widget_ref", None)
    if lw_ref is None:
        return
    lw = lw_ref()
    if lw is None:
        return

    # Hide and shrink the label widget
    lw._label_widget.native.setFixedWidth(0)
    lw._label_widget.native.hide()

    # Set stretch factors: 0 for label, 1 for value
    layout = lw.native.layout()
    layout.setStretch(0, 0)
    layout.setStretch(1, 1)


class _PersistentComboBox(ComboBox):
    """ComboBox that preserves dynamically set choices during napari state resets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choices_initialized = True

    def reset_choices(self):
        # Skip reset after initial setup to preserve dynamically populated values
        if not getattr(self, "_choices_initialized", False):
            super().reset_choices()


class _WidgetBase(Container):
    """Base class for biopb napari widgets."""

    def run(self):
        """Run the processing workflow."""
        raise NotImplementedError("Subclass must implement 'run' method.")

    def _snapshot(self) -> dict:
        """Capture current widget settings as a dict."""
        return {w.label: w.value for w in self._elements}

    def _cleanup(self):
        """Reset widget state after processing completes."""
        self._stop_all_progress()
        self._progress_bar.visible = False
        self._progress_bar.label = (
            "Running..."  # Reset label from "Canceling..."
        )
        self._run_button.visible = True
        self._run_button.enabled = True
        self._cancel_button.visible = False

        # Disconnect the cancel callback to prevent accumulation
        if self._cancel_callback is not None:
            try:
                self._cancel_button.clicked.disconnect(self._cancel_callback)
            except TypeError:
                pass  # Already disconnected
            self._cancel_callback = None

    def _error(self, exc: Exception):
        """Display error message in widget, then reraise for napari notification."""
        self._cleanup()
        # Show succinct error message with word wrap
        self._error_label.value = _format_error_message(exc)
        self._error_label.visible = True
        self._error_label.native.adjustSize()
        logger.error("Processing failed: %s", exc, exc_info=True)
        # Reraise so napari's notification_manager shows the error
        raise exc

    def _cancel(self, worker):
        """Cancel the running worker - non-blocking.

        Args:
            worker: The thread worker to cancel
        """
        self._stop_all_progress()
        self._abort_event.set()

        # Directly cancel the active gRPC futures (if any) for immediate abort
        active_futures = self._active_future_container.get("active")
        if active_futures is not None:
            # Handle both single future and set of futures
            if isinstance(active_futures, set):
                for f in active_futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass  # Future may already be done
            else:
                # Single future (for object detection which is still sequential)
                try:
                    active_futures.cancel()
                except Exception:
                    pass
            self._active_future_container["active"] = None

        worker.quit()
        self._cancel_button.enabled = False
        self._progress_bar.label = "Canceling..."
        # Worker finishes asynchronously → finished signal → _cleanup()

    def _prepare(self):
        """Prepare widget state before processing starts."""
        self._abort_event.clear()
        self._active_future_container = {}  # Reset container for new run
        self._calls_completed = 0
        self._fake_subprogress = 0
        self._pending_calls = 0  # Track calls submitted but not yet completed
        self._total_calls = 0  # Total number of calls expected

        # Clear previous error
        self._error_label.visible = False
        self._error_label.value = ""

        self._progress_bar.visible = True
        self._progress_bar.value = 0

        self._run_button.enabled = False
        self._run_button.visible = False

        self._cancel_button.enabled = True
        self._cancel_button.visible = True

        self.out_layer = None

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._cancel_callback = None
        self._abort_event = threading.Event()
        self._active_future_container: dict = (
            {}
        )  # Container for active gRPC future

        # Make container expand to fill available width in dock widget
        self.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Load persisted config
        self._config = load_config()

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self._is3d = create_widget(
            label="3D", annotation=bool, value=self._config.get("3D", False)
        )

        self._server = create_widget(
            value=self._config["server"]["url"],
            label="Server",
            annotation=str,
        )

        self._progress_bar = ProgressBar(
            label="Running...", value=0, step=1, visible=False
        )

        self._cancel_button = create_widget(
            label="Cancel", widget_type="Button"
        )
        self._cancel_button.visible = False

        self._run_button = create_widget(label="Run", widget_type="Button")
        self._run_button.clicked.connect(self.run)

        self._error_label = Label(value="", label="")
        self._error_label.visible = False
        # Enable rich text for colored status part
        self._error_label.native.setTextFormat(Qt.TextFormat.RichText)
        # Ignore sizeHint horizontally so widget doesn't expand to fit text
        self._error_label.native.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )

        self._elements = [
            self._image_layer_combo,
            self._is3d,
            self._server,
        ]

        # Timer for fake progress during long gRPC calls
        # Parent to native widget for proper Qt lifecycle management
        self._progress_timer = QTimer(self.native)
        self._progress_timer.timeout.connect(self._tick_fake_progress)
        self._calls_completed = 0  # Real progress count
        self._fake_subprogress = 0  # Legacy: kept for compatibility
        self._pending_calls = 0  # Track calls submitted but not yet completed
        self._total_calls = 0  # Total number of calls expected

    def _tick_fake_progress(self):
        """Fake progress between completions - bump current progress slightly.

        With concurrent calls, we show activity by inching toward the completion-based
        target. Progress is capped at max-1 to leave room for final completion.
        """
        if self._pending_calls > 0 and self._total_calls > 0:
            current = self._progress_bar.value
            target = int(
                self._calls_completed
                / self._total_calls
                * self._progress_bar.max
            )
            # Fake progress: inch toward target + small offset, capped at max-1
            fake_target = min(target + 5, self._progress_bar.max - 1)
            self._progress_bar.value = min(current + 1, fake_target)
            logger.debug(
                "Progress tick: current=%d, target=%d, fake_target=%d, value=%d",
                current,
                target,
                fake_target,
                self._progress_bar.value,
            )

    def _on_call_starting(self):
        """Mark start of a gRPC call - increment pending counter and start timer.

        For concurrent calls, we track how many calls are pending (submitted but
        not yet completed) and only start the timer once.
        """
        logger.debug(
            "Progress: call starting, pending=%d", self._pending_calls
        )
        self._pending_calls += 1
        self._total_calls = max(
            self._total_calls, self._pending_calls + self._calls_completed
        )
        # Start timer if not already running and we have pending calls
        if self._pending_calls > 0 and not self._progress_timer.isActive():
            self._progress_timer.start(500)  # 500ms interval

    def _on_call_completed(self):
        """Mark completion of a gRPC call - decrement pending, update progress.

        Progress is calculated based on completed/total ratio with guard to ensure
        value <= max. Timer stops when no pending calls remain.
        """
        logger.debug(
            "Progress: call completed, pending=%d, completed=%d, total=%d",
            self._pending_calls,
            self._calls_completed,
            self._total_calls,
        )
        self._pending_calls -= 1
        self._calls_completed += 1
        # Update progress bar directly with guard to ensure value <= max
        if self._total_calls > 0:
            progress = int(
                self._calls_completed
                / self._total_calls
                * self._progress_bar.max
            )
            self._progress_bar.value = min(progress, self._progress_bar.max)
        # Stop timer when no pending calls
        if self._pending_calls <= 0:
            self._progress_timer.stop()
            # Set to max on completion (guard against wrong estimation)
            self._progress_bar.value = self._progress_bar.max

    def _stop_all_progress(self):
        """Stop progress timer on cancel/finish."""
        self._progress_timer.stop()
        self._calls_completed = 0
        self._fake_subprogress = 0
        self._pending_calls = 0
        self._total_calls = 0

    def _save_config(self):
        """Save current widget settings to config file."""
        settings = self._snapshot()
        self._config["server"]["url"] = settings["Server"]
        self._config["3D"] = settings["3D"]
        save_config(self._config)
