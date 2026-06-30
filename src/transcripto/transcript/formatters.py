from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from transcripto.asr.transcriber import Segment
from transcripto.utils.timefmt import format_hhmmss


@dataclass(frozen=True)
class TranscriptMeta:
    created_at: str
    language: str
    model: str
    samplerate: int
    mic_device: str
    output_device: str
    duration_s: float


def to_pretty_txt(meta: TranscriptMeta, segments: list[Segment]) -> str:
    lines: list[str] = []
    lines.append("Транскрипт")
    lines.append(f"Дата: {meta.created_at}")
    lines.append(f"Язык: {meta.language}")
    lines.append(f"Модель: {meta.model}")
    lines.append(f"Длительность: {format_hhmmss(meta.duration_s)}")
    lines.append(f"Микрофон: {meta.mic_device}")
    lines.append(f"Выход/loopback: {meta.output_device}")
    lines.append("")

    def speaker_label(speaker: str) -> str:
        return "Я" if speaker == "me" else "Собеседник"

    for s in segments:
        t0 = format_hhmmss(s.start)
        t1 = format_hhmmss(s.end)
        lines.append(f"[{t0} - {t1}] {speaker_label(s.speaker)}: {s.text}")
    lines.append("")
    return "\n".join(lines)


def to_json(meta: TranscriptMeta, segments: list[Segment]) -> str:
    payload: dict[str, Any] = {
        "meta": asdict(meta),
        "segments": [asdict(s) for s in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
