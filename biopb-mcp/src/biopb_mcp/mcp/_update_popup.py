"""Qt popup for the update **nagger** (issue #87) — window-only.

A reminder, not a self-applying updater: it tells the user a newer release is
available and shows the install/upgrade command to run (with a one-click copy);
it never installs anything itself (see ``_update_apply`` for why). Opt-out is
per-version ("Skip vX.Y.Z") or total ("Stop checking").

Shown only when a napari window exists (the bootstrap fires the check solely on
the GUI branch). **Non-blocking**: built and ``.show()``n on the Qt main thread
rather than ``.exec()``'d, so it never spins a nested event loop inside
``run_on_main`` — which would tie up the background check thread up to its 300s
timeout. Button clicks invoke ``on_choice(action)`` via Qt's signal delivery, on
the main thread.
"""

import logging

logger = logging.getLogger(__name__)

# Keep shown dialogs alive: a QMessageBox with no surviving Python reference can
# be garbage-collected out from under the event loop. Parenting to the napari
# window mostly covers this; this set is belt-and-suspenders, cleared on close.
_live_dialogs = set()


def show_update_popup(info, on_choice, viewer=None):
    """Build and show the reminder dialog on the Qt main thread; return the box.

    ``on_choice`` is called with one of ``skip`` / ``disable`` / ``later`` when
    the user clicks (the "Copy command" button is handled inline here). ``viewer``
    (when given) parents the box to the napari main window so it centers on it and
    shares its lifetime.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QMessageBox

    from ._update_apply import upgrade_command

    parent = None
    if viewer is not None:
        try:
            parent = viewer.window._qt_window
        except Exception:
            parent = None

    cmd = upgrade_command()

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle("biopb update available")
    box.setText(f"A newer biopb release is available: {info.current} → {info.latest}.")
    box.setInformativeText(
        "To update, run the biopb install/upgrade script in a terminal:\n\n"
        f"    {cmd}\n\n"
        "It reinstalls biopb and briefly restarts it (closing this napari "
        "session). biopb does not update itself."
    )

    # `&&` renders a literal `&` (a single `&` would be a Qt mnemonic). Qt orders
    # the buttons by role, not insertion order.
    copy_btn = box.addButton("Copy command", QMessageBox.ButtonRole.AcceptRole)
    skip_btn = box.addButton(f"Skip {info.latest}", QMessageBox.ButtonRole.ActionRole)
    disable_btn = box.addButton("Stop checking", QMessageBox.ButtonRole.DestructiveRole)
    dismiss_btn = box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(copy_btn)
    box.setEscapeButton(dismiss_btn)

    def _dispatch(btn):
        if btn is copy_btn:
            # Copy the upgrade command to the clipboard; no config change.
            try:
                QApplication.clipboard().setText(cmd)
                logger.info("update upgrade command copied to clipboard")
            except Exception:
                logger.exception("failed to copy upgrade command")
            return
        action = {skip_btn: "skip", disable_btn: "disable"}.get(btn, "later")
        try:
            on_choice(action)
        except Exception:
            logger.exception("update popup choice handler failed")

    box.buttonClicked.connect(_dispatch)

    _live_dialogs.add(box)
    box.finished.connect(lambda _=None: _live_dialogs.discard(box))

    # WindowModal: blocks input to the napari window but not the whole app, and
    # (unlike exec) does not spin a nested event loop, so show() returns at once.
    box.setWindowModality(Qt.WindowModality.WindowModal)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.show()
    box.raise_()
    box.activateWindow()
    return box
