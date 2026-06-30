from __future__ import annotations


def format_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"
