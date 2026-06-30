from __future__ import annotations

import sys
import warnings

from PySide6.QtWidgets import QApplication


def main() -> None:
    app = QApplication(sys.argv)

    # soundcard (Windows/MediaFoundation) can emit frequent RuntimeWarning messages like
    # "data discontinuity in recording" when the app can't keep up in real time.
    # Keep the console usable by showing it at most once.
    warnings.filterwarnings(
        "once",
        message="data discontinuity in recording",
        category=RuntimeWarning,
        module=r"soundcard\..*",
    )

    # Delay importing Windows COM/MediaFoundation users until Qt is initialized.
    # This avoids OleInitialize() failures (0x80010106) caused by prior COM init.
    from transcripto.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())
