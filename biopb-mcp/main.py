"""Entry point for bundled biopb-mcp application."""

import napari
from biopb_mcp.tensor_browser import TensorBrowserWidget

viewer = napari.Viewer()

# Add Tensor Browser widget (requires viewer argument)
viewer.window.add_dock_widget(TensorBrowserWidget(viewer))

napari.run()
