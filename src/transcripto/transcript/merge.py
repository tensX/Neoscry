from __future__ import annotations

from dataclasses import asdict

from transcripto.asr.transcriber import Segment


def merge_and_sort(segments: list[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: (s.start, s.end))


def merge_consecutive_same_speaker(segments: list[Segment], max_gap_s: float = 0.75) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = []
    cur = segments[0]
    for s in segments[1:]:
        if s.speaker == cur.speaker and (s.start - cur.end) <= max_gap_s:
            text = (cur.text.rstrip() + " " + s.text.lstrip()).strip()
            cur = Segment(start=cur.start, end=max(cur.end, s.end), text=text, speaker=cur.speaker)
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)
    return merged


def segments_to_dicts(segments: list[Segment]) -> list[dict]:
    return [asdict(s) for s in segments]
