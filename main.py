"""Entry point for bundled napari-biopb application."""

import napari
from napari_biopb import TensorBrowserWidget

viewer = napari.Viewer()

# Add Tensor Browser widget (requires viewer argument)
viewer.window.add_dock_widget(TensorBrowserWidget(viewer))

napari.run()
