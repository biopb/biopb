import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    Container,
    FloatSpinBox,
    Label,
    LineEdit,
    SpinBox,
)
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QSizePolicy

from ._chunking import (
    ResultBuilder,
    _get_axis_mapping,
    _get_iter_spec,
    _validate_data_shape,
)
from ._widget_base import _make_full_width, _PersistentComboBox, _WidgetBase

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class ImageProcessingWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ProcessImage endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        logger.debug(
            "Initialized ImageProcessingWidget with server: %s",
            self._server.value,
        )

        # Op selector for choosing which operation to run
        self._op_selector = _PersistentComboBox(
            choices=["<no ops>"],  # Placeholder until populated from server
            label="Op",
            visible=False,  # Hidden until valid schema is loaded
        )

        # Description display for the selected op (multiline with word wrap)
        self._op_description = Label(value="", label="")
        self._op_description.visible = False
        # Configure for multiline with word wrap
        self._op_description.native.setWordWrap(True)
        self._op_description.native.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self._op_description.native.setMaximumHeight(250)

        # Container for dynamically generated kwargs widgets
        self._kwargs_container = Container(label="Kwargs", visible=False)

        # Status label for ops connection status
        self._ops_status = Label(value="Ops: Not connected", label="")

        # Store schemas for each op (populated from GetOpNames response)
        self._op_schemas = {}

        # Debounce timer for server URL changes (1.5 seconds)
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._fetch_ops)

        # Connect server changes to debounce timer
        self._server.changed.connect(lambda: self._debounce_timer.start(1500))

        # Connect op selector changes to rebuild kwargs widgets
        self._op_selector.changed.connect(self._on_op_changed)

        self.extend(
            self._elements
            + [
                self._ops_status,
                self._op_selector,
                self._op_description,
                self._kwargs_container,
                self._progress_bar,
                self._cancel_button,
                self._run_button,
                self._error_label,
            ]
        )

        # Configure labels to span full width (hide the label column)
        _make_full_width(self._error_label)
        _make_full_width(self._ops_status)
        _make_full_width(self._op_description)

        # Fetch ops during initialization
        self._fetch_ops()

    def _snapshot(self) -> dict:
        """Capture current widget settings including op and kwargs."""
        settings = super()._snapshot()
        settings["Op"] = self._op_selector.value
        for w in self._kwargs_container:
            value = w.value
            if isinstance(w, SpinBox):
                value = int(value)
            # Use original parameter key (stored in _param_key) for kwargs
            key = getattr(w, "_param_key", w.label)
            settings[key] = value
        return settings

    def _fetch_ops(self):
        """Fetch available operations from server asynchronously."""
        from ._grpc import _get_label_filter, get_op_names

        self._ops_status.value = "Ops: Fetching..."
        self._op_selector.visible = False
        self._kwargs_container.visible = False
        settings = self._snapshot()

        # Extract label filter from server URL
        label_filter = _get_label_filter(settings["Server"])

        def _fetch():
            try:
                ops = get_op_names(settings)
                return ops, label_filter
            except Exception as e:
                logger.debug("Failed to fetch ops: %s", e)
                return None, label_filter

        def _on_result(result_tuple):
            result, label_filter = result_tuple
            if result:
                self._op_schemas = dict(result.op_schemas)

                # Filter ops by label if specified
                # Head match requires min 3 chars; shorter filters use exact match only
                op_list = list(result.names)
                if label_filter:
                    if len(label_filter) >= 3:
                        filtered_ops = [
                            op
                            for op in op_list
                            if any(
                                label.startswith(label_filter)
                                for label in self._op_schemas[op].labels
                            )
                        ]
                    else:
                        # Short filter: exact match only (e.g., "op" matches label "op")
                        filtered_ops = [
                            op
                            for op in op_list
                            if label_filter in self._op_schemas[op].labels
                        ]
                    op_list = filtered_ops

                op_list = sorted(op_list)
                self._op_selector.choices = op_list
                if op_list:
                    self._op_selector.value = op_list[0]

                if len(op_list) == 1:
                    # Single op: hide selector, show op name in status
                    self._ops_status.value = f"Ops: Connected ({op_list[0]})"
                    self._op_selector.visible = False
                elif op_list:
                    # Multiple ops: show selector with count in status
                    count_str = f"{len(op_list)} ops"
                    if label_filter:
                        count_str += f", filtered by '{label_filter}'"
                    self._ops_status.value = f"Ops: Connected ({count_str})"
                    self._op_selector.visible = True
                else:
                    # No ops match the filter
                    if label_filter:
                        self._ops_status.value = (
                            f"Ops: No ops with label '{label_filter}'"
                        )
                    else:
                        self._ops_status.value = "Ops: No ops available"

        self._ops_status.value = "Ops: No schema available"
        self._op_selector.choices = ["<no ops>"]
        self._op_schemas = {}
        self._op_selector.visible = False
        self._kwargs_container.visible = False

        thread_worker(
            _fetch,
            start_thread=True,
            connect={"returned": _on_result},
        )()

    def _on_op_changed(self):
        """Rebuild kwargs widgets when op selection changes."""
        op_name = self._op_selector.value
        self._clear_kwargs_container()

        if not op_name or op_name not in self._op_schemas:
            self._kwargs_container.visible = False
            self._op_description.visible = False
            return

        schema = self._op_schemas[op_name]

        # Show description if available
        if schema.description:
            self._op_description.value = schema.description
            # Force Qt to recalculate height for wrapped text
            self._op_description.native.adjustSize()
            self._op_description.visible = True
        else:
            self._op_description.visible = False

        has_kwargs = bool(dict(schema.default_kwargs))
        self._kwargs_container.visible = has_kwargs

        if has_kwargs:
            default_kwargs = dict(schema.default_kwargs)
            logger.debug(
                "Building kwargs widgets for op: %s with kwargs: %s",
                op_name,
                default_kwargs,
            )
            for key, value in default_kwargs.items():
                widget = self._create_widget_for_value(key, value)
                if widget is not None:
                    self._kwargs_container.append(widget)

    def _clear_kwargs_container(self):
        """Clear all widgets from kwargs container."""
        while len(self._kwargs_container) > 0:
            widget = self._kwargs_container[0]
            self._kwargs_container.remove(widget)

    def _create_widget_for_value(self, key: str, value):
        """Create appropriate widget for a kwargs value type.

        Args:
            key: Parameter name (used as widget label)
            value: Default value (determines widget type)

        Returns:
            magicgui widget or None for unsupported types (list_value)
        """
        # struct_pb2.Value has different fields for different types
        if isinstance(value, bool):
            widget = CheckBox(value=value, label=key)
        elif isinstance(value, int):
            widget = SpinBox(value=value, label=key, step=1)
        elif isinstance(value, float):
            # Heuristic: check if float has fractional part
            if value == int(value):
                # No fractional part → treat as integer
                widget = SpinBox(value=int(value), label=key, step=1)
            else:
                widget = FloatSpinBox(value=value, label=key)
        elif isinstance(value, str):
            widget = LineEdit(value=value, label=key)
        elif isinstance(value, list):
            # TODO: Skip list values for now - complex UI
            logger.debug("Skipping list kwargs: %s", key)
            return None
        else:
            logger.debug("Unknown kwargs type for %s: %s", key, type(value))
            return None

        # Store original parameter key for use in _snapshot
        widget._param_key = key
        return widget

    def run(self):
        from ._grpc import CALL_START, grpc_process_image

        settings = self._snapshot()
        image_layer = settings["Image"]
        op_name = settings["Op"]

        # Get axis order from raw layer data (respects napari heuristics)
        axis_order = _get_axis_mapping(image_layer, settings["3D"])

        # Get hint from op schema
        hint = None
        if op_name in self._op_schemas:
            hint = self._op_schemas[op_name].input_shape_hint

        # Handle multiscale: use highest resolution level
        image_data = image_layer.data
        if image_layer.multiscale:
            image_data = image_data[0]

        # Compute iteration spec
        iter_spec = _get_iter_spec(axis_order, hint)

        # Validate
        _validate_data_shape(image_data, axis_order, hint)

        # Prepare result builder and progress
        result_builder = ResultBuilder(iter_spec, image_data.shape)
        n_iterations = (
            1
            if not iter_spec.iter_dims
            else int(
                np.prod(
                    [
                        image_data.shape[axis_order.index(ax)]
                        for ax in iter_spec.iter_dims
                    ]
                )
            )
        )

        self._prepare()
        self._progress_bar.max = n_iterations * 10

        # Initialize annotation collection
        self._current_annotations = []

        def _update(value):
            if value == CALL_START:
                # gRPC call starting - begin fake progress
                self._on_call_starting()
                return

            result_chunk, position, annotation_data = value

            # Mark call completion
            self._on_call_completed()

            # Collect annotation data if present, annotated with position
            if not annotation_data.empty:
                # Add position columns to annotation
                position_annotation = _add_position_to_annotation(
                    annotation_data, position, iter_spec.axis_order
                )
                self._current_annotations.append(position_annotation)

            # Skip image handling if no image data in response
            if result_chunk is None:
                return

            result_builder.add_result(position, result_chunk)

            if result_builder.buffer is not None:
                # Use raw 5D buffer for preview during streaming
                output_data = result_builder.buffer
                if self.out_layer is None:
                    name = image_layer.name + "_processed"
                    if name in self._viewer.layers:
                        self._viewer.layers.remove(name)
                    self._viewer.add_image(output_data, name=name)
                    self.out_layer = self._viewer.layers[name]
                    self.out_layer.reset_contrast_limits()
                else:
                    self.out_layer.data = output_data
                    self.out_layer.refresh()

        def _on_success():
            """Callback on successful completion - cleanup and save config."""
            if result_builder.buffer is not None:
                # Transform 5D buffer to final shape and recreate layer
                final_data, is_rgb = _prepare_for_viewer(result_builder.buffer)
                name = image_layer.name + "_processed"

                # Remove existing layer if present
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)

                # Add new layer with correct RGB setting
                self._viewer.add_image(final_data, name=name, rgb=is_rgb)
                self.out_layer = self._viewer.layers[name]
                self.out_layer.reset_contrast_limits()

            # Handle annotations: merge and display table
            if self._current_annotations:
                merged_annotation = _merge_annotations(self._current_annotations)
                if not merged_annotation.empty:
                    # Use output layer if exists, else input layer
                    target_layer = self.out_layer if self.out_layer else image_layer
                    target_layer.metadata["annotation"] = merged_annotation
                    _show_annotation_table(
                        target_layer, self._viewer, merged_annotation
                    )

            self._cleanup()
            self._save_config()

        # Use dask_optimized_slicing for large images
        with image_layer.dask_optimized_slicing():
            worker = grpc_process_image(
                image_data,
                settings,
                iter_spec,
                abort_event=self._abort_event,
                future_container=self._active_future_container,
            )

        self._cancel_callback = lambda: self._cancel(worker)
        self._cancel_button.clicked.connect(self._cancel_callback)
        worker.yielded.connect(_update)
        worker.finished.connect(_on_success)
        worker.errored.connect(self._error)

        worker.start()


def _prepare_for_viewer(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Prepare 5D TZCYX data for napari viewer.

    Checks if data is RGB (uint8 with 3 or 4 channels) and moves C to last dim.
    Otherwise squeezes singleton dims.

    Args:
        data: 5D array in TZCYX order

    Returns:
        Tuple of (array suitable for napari viewer, is_rgb flag)
    """
    # data is TZCYX (5D)
    is_rgb = data.dtype == np.uint8 and data.shape[2] in (
        3,
        4,
    )  # C dimension at index 2 in TZCYX

    if is_rgb:
        # Move C from index 2 to last: TZCYX -> TZYXC
        data = np.moveaxis(data, 2, -1)
        # Selectively squeeze singleton dims (except last which is C)
        squeeze_axes = tuple(i for i, s in enumerate(data.shape[:-1]) if s == 1)
        if squeeze_axes:
            data = np.squeeze(data, axis=squeeze_axes)
    else:
        # Non-RGB: squeeze singleton dims
        data = np.squeeze(data)

    return data, is_rgb


def _add_position_to_annotation(
    annotation: pd.DataFrame, position: dict, axis_order: str
) -> pd.DataFrame:
    """Add position columns to annotation DataFrame.

    Args:
        annotation: DataFrame of annotation data
        position: Dict mapping numeric dim indices to index values
        axis_order: Axis order string (e.g., "TZYXC", "TZCYX")

    Returns:
        DataFrame with added position columns (T, Z if applicable)
    """
    if annotation.empty:
        return annotation

    result = annotation.copy()
    # Add position columns based on axis_order
    for dim_idx, dim_value in position.items():
        dim_name = axis_order[dim_idx]
        if dim_name in ("T", "Z"):
            result[dim_name] = dim_value

    return result


def _merge_annotations(annotations: list) -> pd.DataFrame:
    """Merge multiple annotation DataFrames into one.

    Args:
        annotations: List of DataFrames from multiple chunks

    Returns:
        Concatenated DataFrame with all rows
    """
    if not annotations:
        return pd.DataFrame()

    return pd.concat(annotations, ignore_index=True)


def _show_annotation_table(
    layer: "napari.layers.Layer",
    viewer: "napari.Viewer",
    annotation: pd.DataFrame,
):
    """Display annotation as a table widget in napari viewer.

    Args:
        layer: Layer to associate with the table
        viewer: napari viewer instance
        annotation: DataFrame of annotation data
    """
    try:
        from napari_skimage_regionprops import TableWidget

        table_widget = TableWidget(layer, viewer)
        # Convert DataFrame to dict for TableWidget
        table_widget.set_content(annotation.to_dict("list"))
        viewer.window.add_dock_widget(
            table_widget,
            area="right",
            name="Annotation: " + layer.name,
        )
    except ImportError:
        logger.warning(
            "napari-skimage-regionprops not installed, "
            "annotation stored in layer.metadata but table not displayed"
        )
