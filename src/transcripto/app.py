from __future__ import annotations

import sys
import warnings
import faulthandler
from pathlib import Path

from PySide6.QtWidgets import QApplication


def main() -> None:
    # Capture hard crashes (e.g., native library aborts) to a log file.
    try:
        crash_log = Path.cwd() / "neoscry_crash.log"
        with crash_log.open("a", encoding="utf-8") as f:
            faulthandler.enable(file=f, all_threads=True)
    except Exception:
        pass

    app = QApplication(sys.argv)

    # soundcard (Windows/MediaFoundation) can emit frequent warnings like
    # "data discontinuity in recording" when the app can't keep up in real time.
    # Hide them to keep the console usable.
    try:
        from soundcard import SoundcardRuntimeWarning

        warnings.filterwarnings(
            "ignore",
            message="data discontinuity in recording",
            category=SoundcardRuntimeWarning,
            module=r"soundcard\..*",
        )
    except Exception:
        warnings.filterwarnings(
            "ignore",
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
