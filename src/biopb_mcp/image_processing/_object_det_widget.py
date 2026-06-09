import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from magicgui.widgets import ComboBox, create_widget

from .._config import CONFIG, get_grid_params
from ._widget_base import _make_full_width, _WidgetBase

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class ObjectDetectionWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ObjectDetection endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        self._threshold = create_widget(
            value=CONFIG.get("widget.detection.min_score"),
            label="Min Score",
            annotation=float,
            widget_type="FloatSlider",
            options={"min": 0, "max": 1},
        )

        self._use_advanced = create_widget(
            value=False,
            label="Advanced",
            annotation=bool,
        )
        self._use_advanced.changed.connect(self._activte_advanced_inputs)

        self._size_hint = create_widget(
            value=CONFIG.get("widget.detection.size_hint"),
            label="Size Hint",
            annotation=float,
            widget_type="FloatSlider",
            options={"min": 10, "max": 200, "visible": False},
        )

        self._nms = ComboBox(
            value=CONFIG.get("widget.detection.nms"),
            choices=["Off", "Iou-0.2", "Iou-0.4", "Iou-0.6", "Iou-0.8"],
            label="NMS",
            visible=False,
        )

        self._aspect_ratio = create_widget(
            value=CONFIG.get("widget.detection.z_aspect_ratio"),
            label="Z Aspect Ratio",
            options={"visible": False},
        )

        self._hidden_elements = [
            self._size_hint,
            self._nms,
            self._aspect_ratio,
        ]

        self._elements += [
            self._threshold,
            self._use_advanced,
        ]
        self._elements += self._hidden_elements

        # append into/extend the container with your widgets
        self.extend(
            self._elements
            + [
                self._progress_bar,
                self._cancel_button,
                self._run_button,
                self._error_label,
            ]
        )

        # Configure error label to span full width
        _make_full_width(self._error_label)

    def _activte_advanced_inputs(self):
        """Toggle visibility of advanced settings widgets."""
        for ctl in self._hidden_elements:
            ctl.visible = self._use_advanced.value

    def _get_data(self, image_layer, is_3d: bool):
        """Extract image data from layer, handling multiscale and RGB.

        Args:
            image_layer: napari Image layer
            is_3d: whether to treat as 3D data

        Returns:
            reshaped image array (batch, z/y, y, x, channel)
        """
        image_data = image_layer.data
        if image_layer.multiscale:
            image_data = image_data[0]
        if image_layer.rgb:
            img_dim = image_data.shape[-4:] if is_3d else image_data.shape[-3:]
            image_data = image_data.reshape((-1,) + img_dim)
        else:
            img_dim = image_data.shape[-3:] if is_3d else image_data.shape[-2:]
            image_data = image_data.reshape((-1,) + img_dim + (1,))

        return image_data

    def _get_grid_positions(
        self, image, settings: dict
    ) -> List[Tuple[slice, ...]]:
        """Compute grid positions for tiled processing.

        Args:
            image: input image array
            settings: widget settings dict

        Returns:
            list of slice tuples defining patch positions
        """
        is_3d = settings["3D"]
        grid_size, stride = get_grid_params(is_3d, CONFIG.as_dict())

        pos_pars = (
            image.shape[:-1],
            grid_size,
            stride,
        )

        grid_start = [
            slice(0, max(d - (gs - ss), 1), ss) for d, gs, ss in zip(*pos_pars)
        ]
        grid_start = np.moveaxis(np.mgrid[grid_start], 0, -1)
        grid_start = grid_start.reshape(-1, image.ndim - 1)

        grids = []
        for x in grid_start:
            slc = (slice(x[i], x[i] + gs) for i, gs in enumerate(pos_pars[1]))
            grids.append(tuple(slc))

        return grids

    def _save_config(self):
        """Save current widget settings via the config singleton.

        Targeted ``CONFIG.set`` writes touch only this widget's leaves; sibling
        keys (e.g. mcp.services.process_image_servers) are preserved. Batched
        with ``persist=False`` and flushed once via ``CONFIG.save()``.
        """
        settings = self._snapshot()
        CONFIG.set("widget.server_url", settings["Server"], persist=False)
        CONFIG.set("widget.is_3d", settings["3D"], persist=False)
        CONFIG.set(
            "widget.detection.min_score", settings["Min Score"], persist=False
        )
        CONFIG.set(
            "widget.detection.size_hint", settings["Size Hint"], persist=False
        )
        CONFIG.set("widget.detection.nms", settings["NMS"], persist=False)
        CONFIG.set(
            "widget.detection.z_aspect_ratio",
            settings["Z Aspect Ratio"],
            persist=False,
        )
        CONFIG.save()

    def run(self):
        from ._grpc import CALL_START, grpc_object_detection

        self.n_results = 0

        settings = self._snapshot()
        image_layer = settings["Image"]

        def _update(value):
            if value == CALL_START:
                # gRPC call starting - begin fake progress
                self._on_call_starting()
            elif value is None:  # patch prediction completed
                self._on_call_completed()
            else:  # full image prediction result (label array)
                self._on_call_completed()
                if self.out_layer is None:
                    name = self._image_layer_combo.value.name + "_label"
                    _output = np.zeros(image_data.shape[:-1], dtype=int)
                    if name in self._viewer.layers:
                        self._viewer.layers[name].data = _output
                    else:
                        self._viewer.add_labels(_output, name=name)

                    self.out_layer = self._viewer.layers[name]

                self.out_layer.data[self.n_results, ...] = value
                self.n_results += 1
                self.out_layer.refresh()

        def _on_success():
            """Callback on successful completion - cleanup and save config."""
            self._cleanup()
            self._save_config()

        with image_layer.dask_optimized_slicing():
            image_data = self._get_data(image_layer, settings["3D"])

            grid_positions = self._get_grid_positions(image_data[0], settings)

            self._progress_bar.max = len(image_data) * len(grid_positions) * 10

            self._prepare()

            worker = grpc_object_detection(
                image_data,
                settings,
                grid_positions,
                abort_event=self._abort_event,
                future_container=self._active_future_container,
            )

            self._cancel_callback = lambda: self._cancel(worker)
            self._cancel_button.clicked.connect(self._cancel_callback)
            worker.yielded.connect(_update)
            worker.finished.connect(_on_success)
            worker.errored.connect(self._error)

            worker.start()
