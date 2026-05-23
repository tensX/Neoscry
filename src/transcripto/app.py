from __future__ import annotations

import logging
import os
import sys
import warnings
import faulthandler
from pathlib import Path

from PySide6.QtWidgets import QApplication


def _crash_log_path() -> Path:
    """Resolve a writable per-user location for the faulthandler log.

    Path.cwd() can be read-only (Program Files-style launches) or change between
    runs, which makes the crash log useless. Prefer the platform appdata dir,
    fall back to the user home, and only then fall back to cwd.
    """
    candidates: list[Path] = []
    if os.name == "nt":
        appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Neoscry")
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        if xdg:
            candidates.append(Path(xdg) / "Neoscry")
        candidates.append(Path.home() / ".local" / "state" / "Neoscry")
    candidates.append(Path.home() / ".neoscry")
    candidates.append(Path.cwd())

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            p = base / "neoscry_crash.log"
            # Touch to confirm writability before handing off to faulthandler.
            with p.open("a", encoding="utf-8"):
                pass
            return p
        except Exception:
            continue
    return Path.cwd() / "neoscry_crash.log"


def main() -> None:
    # Capture hard crashes (e.g., native library aborts) to a log file.
    try:
        crash_log = _crash_log_path()
        # Keep the file handle open for the lifetime of the process so
        # faulthandler can still write it after our cleanup runs.
        crash_fp = crash_log.open("a", encoding="utf-8")
        faulthandler.enable(file=crash_fp, all_threads=True)
    except Exception:
        logging.getLogger(__name__).warning("faulthandler setup failed", exc_info=True)

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
