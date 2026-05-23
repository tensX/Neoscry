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
    p.add_argument("--lang", default="auto")
    p.add_argument("--ui-lang", default="ru")
    p.add_argument("--device", default="auto")
    p.add_argument("--compute", default="auto")

    p.add_argument("--mic-label", default="")
    p.add_argument("--out-label", default="")
    p.add_argument("--session-label", default="")
    p.add_argument("--swap-speakers", action="store_true")

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


def _classify_error(msg: str) -> str:
    s = msg.lower()
    if "cublas" in s and ".dll" in s:
        return "cuda_missing"
    if "cudart" in s and ".dll" in s:
        return "cuda_missing"
    if "libcublas" in s and (".so" in s or "cannot open shared object" in s):
        return "cuda_missing"
    if "libcudart" in s and (".so" in s or "cannot open shared object" in s):
        return "cuda_missing"
    return "unknown"


def _emit(payload: dict) -> None:
    # All status output goes through this JSON-per-line envelope so the GUI
    # can dispatch on type without parsing free-form text.
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        # Last-resort: drop non-ASCII so we never die in the error reporter.
        try:
            safe = json.dumps(payload, ensure_ascii=True) + "\n"
            sys.stdout.write(safe)
            sys.stdout.flush()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    # Stdout on Windows defaults to cp1252, which cannot encode Cyrillic.
    # Forcing UTF-8 keeps both the JSON envelope and any localized error
    # messages printable instead of crashing the error reporter itself.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    ui_lang = str(args.ui_lang).strip().lower() or "ru"

    try:
        mic = Path(args.mic) if args.mic else Path()
        loop = Path(args.loop) if args.loop else Path()
        out_txt = Path(args.out_txt)
        out_json = Path(args.out_json)

        mic_ok = bool(args.mic) and _non_empty(mic)
        loop_ok = bool(args.loop) and _non_empty(loop)
        if not mic_ok and not loop_ok:
            raise RuntimeError(
                "No non-empty input audio files provided" if ui_lang == "en" else "Не переданы непустые аудиофайлы"
            )

        sr, dur = _infer_media_info([mic, loop])

        model_name = str(args.model).strip() or "large-v3"
        tr = WhisperTranscriber(
            model_name=model_name,
            language=str(args.lang).strip() or "auto",
            ui_lang=ui_lang,
            device=str(args.device),
            compute_type=str(args.compute),
        )

        mic_speaker = "other" if bool(args.swap_speakers) else "me"
        loop_speaker = "me" if bool(args.swap_speakers) else "other"
        mic_segments = tr.transcribe_file(str(mic), speaker=mic_speaker) if mic_ok else []
        other_segments = tr.transcribe_file(str(loop), speaker=loop_speaker) if loop_ok else []

        merged = merge_and_sort(mic_segments + other_segments)
        merged = merge_consecutive_same_speaker(merged)

        meta = TranscriptMeta(
            created_at=now_iso_local(),
            session_label=str(args.session_label).strip(),
            language=str(args.lang).strip() or "auto",
            model=model_name,
            samplerate=int(sr),
            mic_device=str(args.mic_label),
            output_device=str(args.out_label),
            duration_s=float(dur),
        )

        txt = to_pretty_txt(meta, merged, ui_lang=ui_lang)
        js = to_json(meta, merged)

        out_txt.write_text(txt, encoding="utf-8")
        out_json.write_text(js, encoding="utf-8")

        _emit({"ok": True, "segments": len(merged)})
    except Exception as e:
        raw = str(e)
        code = _classify_error(raw)
        if code == "cuda_missing":
            if ui_lang == "en":
                message = (
                    "GPU transcription failed: CUDA runtime libraries were not found (cublas).\n"
                    "Install NVIDIA CUDA Runtime 12.x or switch Device to CPU."
                )
            else:
                message = (
                    "Не удалось распознать на GPU: не найдены CUDA-библиотеки (cublas).\n"
                    "Установите NVIDIA CUDA Runtime 12.x или переключите устройство на CPU."
                )
        else:
            message = raw

        _emit({"ok": False, "error": code, "message": message, "detail": raw})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
