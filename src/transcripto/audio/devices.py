from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import soundcard as sc


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str
    kind: str  # "mic" | "speaker"

    def label(self) -> str:
        return self.name


def list_microphones() -> list[AudioDevice]:
    out: list[AudioDevice] = []
    for m in sc.all_microphones(include_loopback=False):
        out.append(AudioDevice(id=str(m.id), name=str(m.name), kind="mic"))
    return out


def list_loopback_microphones() -> list[AudioDevice]:
    """Loopback capture endpoints, exposed as microphones."""
    out: list[AudioDevice] = []
    for m in sc.all_microphones(include_loopback=True):
        if bool(getattr(m, "isloopback", False)):
            out.append(AudioDevice(id=str(m.id), name=f"{m.name} [loopback]", kind="mic"))
    return out


def list_speakers() -> list[AudioDevice]:
    out: list[AudioDevice] = []
    for s in sc.all_speakers():
        out.append(AudioDevice(id=str(s.id), name=str(s.name), kind="speaker"))
    return out


def get_default_microphone_id() -> str | None:
    try:
        m = sc.default_microphone()
        return str(m.id) if m else None
    except Exception:
        return None


def get_default_speaker_id() -> str | None:
    try:
        s = sc.default_speaker()
        return str(s.id) if s else None
    except Exception:
        return None


def get_default_loopback_microphone_id() -> str | None:
    """Best-effort default for loopback capture."""
    try:
        s = sc.default_speaker()
        if not s:
            return None
        m = sc.get_microphone(s.id, include_loopback=True)
        return str(m.id) if m else None
    except Exception:
        return None


def find_device_by_id(devices: Iterable[AudioDevice], device_id: str | None) -> AudioDevice | None:
    if not device_id:
        return None
    for d in devices:
        if d.id == device_id:
            return d
    return None
