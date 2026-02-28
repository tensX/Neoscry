from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from transcripto.asr.transcriber import WhisperTranscriber
from transcripto.transcript.formatters import TranscriptMeta, now_iso_local, to_json, to_pretty_txt
from transcripto.transcript.merge import merge_and_sort, merge_consecutive_same_speaker


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="neoscry-transcribe")

    p.add_argument("--mic", default="", help="Path to mic.wav (optional)")
    p.add_argument("--loop", default="", help="Path to loopback.wav (optional)")

    p.add_argument("--model", default="large-v3")
    p.add_argument("--lang", default="ru")
    p.add_argument("--ui-lang", default="ru")
    p.add_argument("--device", default="auto")
    p.add_argument("--compute", default="auto")

    p.add_argument("--mic-label", default="")
    p.add_argument("--out-label", default="")

    p.add_argument("--out-txt", required=True)
    p.add_argument("--out-json", required=True)

    return p.parse_args(argv)


def _non_empty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 4096
    except Exception:
        return False


def _infer_media_info(paths: list[Path]) -> tuple[int, float]:
    sr = 48000
    dur = 0.0
    for p in paths:
        if not _non_empty(p):
            continue
        try:
            info = sf.info(str(p))
            if getattr(info, "samplerate", None):
                sr = int(info.samplerate)
            if getattr(info, "frames", None) and sr > 0:
                dur = max(dur, float(info.frames) / float(sr))
        except Exception:
            continue
    return sr, dur


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    mic = Path(args.mic) if args.mic else Path()
    loop = Path(args.loop) if args.loop else Path()
    out_txt = Path(args.out_txt)
    out_json = Path(args.out_json)

    mic_ok = bool(args.mic) and _non_empty(mic)
    loop_ok = bool(args.loop) and _non_empty(loop)
    if not mic_ok and not loop_ok:
        raise SystemExit("No non-empty input audio files provided")

    sr, dur = _infer_media_info([mic, loop])

    tr = WhisperTranscriber(
        model_name=str(args.model),
        language=str(args.lang).strip() or "ru",
        ui_lang=str(args.ui_lang).strip() or "ru",
        device=str(args.device),
        compute_type=str(args.compute),
    )

    mic_segments = tr.transcribe_file(str(mic), speaker="me") if mic_ok else []
    other_segments = tr.transcribe_file(str(loop), speaker="other") if loop_ok else []

    merged = merge_and_sort(mic_segments + other_segments)
    merged = merge_consecutive_same_speaker(merged)

    meta = TranscriptMeta(
        created_at=now_iso_local(),
        language=str(args.lang).strip() or "ru",
        model=str(args.model),
        samplerate=int(sr),
        mic_device=str(args.mic_label),
        output_device=str(args.out_label),
        duration_s=float(dur),
    )

    txt = to_pretty_txt(meta, merged, ui_lang=str(args.ui_lang))
    js = to_json(meta, merged)

    out_txt.write_text(txt, encoding="utf-8")
    out_json.write_text(js, encoding="utf-8")

    # Also print a tiny JSON summary for callers.
    print(json.dumps({"ok": True, "segments": len(merged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
