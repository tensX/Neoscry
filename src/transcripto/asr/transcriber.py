from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str  # "me" | "other"


def _resample_to_16000(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float32)
    if sr == 16000:
        return x, sr

    try:
        from scipy.signal import resample_poly

        y = resample_poly(x, 16000, sr).astype(np.float32, copy=False)
        return y, 16000
    except Exception:
        # Fallback: linear interpolation (worse quality, but avoids hard crash).
        if x.size == 0:
            return x, 16000
        duration = x.size / float(sr)
        n = max(1, int(round(duration * 16000)))
        t_old = np.linspace(0.0, duration, num=x.size, endpoint=False)
        t_new = np.linspace(0.0, duration, num=n, endpoint=False)
        y = np.interp(t_new, t_old, x).astype(np.float32, copy=False)
        return y, 16000


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "ru",
        ui_lang: str = "ru",
        beam_size: int = 5,
        vad_filter: bool = True,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        compute_type: str = "auto",  # "auto" | "float16" | "int8" | ...
    ) -> None:
        self._model_name = model_name
        self._language = language
        self._ui_lang = str(ui_lang).strip().lower() or "ru"
        self._beam_size = int(beam_size)
        self._vad_filter = bool(vad_filter)
        self._device = str(device)
        self._compute_type = str(compute_type)

        self._model = None
        self.runtime_device: str | None = None
        self.runtime_compute_type: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        def _pick_compute(device: str) -> str:
            if self._compute_type != "auto":
                return self._compute_type
            return "float16" if device == "cuda" else "int8"

        pref = self._device.lower().strip()
        if pref not in ("auto", "cuda", "cpu"):
            pref = "auto"

        if pref in ("auto", "cuda"):
            try:
                ct = _pick_compute("cuda")
                self._model = WhisperModel(self._model_name, device="cuda", compute_type=ct)
                self.runtime_device = "cuda"
                self.runtime_compute_type = ct
                return self._model
            except Exception as e:
                if pref == "cuda":
                    if self._ui_lang == "en":
                        raise RuntimeError(
                            "CUDA selected, but initialization failed. "
                            "Most likely a CPU-only build of ctranslate2 is installed. "
                            f"Original error: {e}"
                        )
                    raise RuntimeError(
                        "CUDA выбран, но инициализация не удалась. "
                        "Скорее всего установлен CPU-only ctranslate2. "
                        f"Оригинальная ошибка: {e}"
                    )

        ct = _pick_compute("cpu")
        self._model = WhisperModel(self._model_name, device="cpu", compute_type=ct)
        self.runtime_device = "cpu"
        self.runtime_compute_type = ct
        return self._model

    def transcribe(self, audio: np.ndarray, samplerate: int, speaker: str) -> list[Segment]:
        x, sr = _resample_to_16000(audio, samplerate)
        if x.size == 0:
            return []

        model = self._load()
        segments, _info = model.transcribe(
            x,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )

        out: list[Segment] = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(Segment(start=float(s.start), end=float(s.end), text=text, speaker=speaker))
        return out

    def transcribe_file(self, path: str, speaker: str) -> list[Segment]:
        model = self._load()
        segments, _info = model.transcribe(
            path,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )
        out: list[Segment] = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(Segment(start=float(s.start), end=float(s.end), text=text, speaker=speaker))
        return out
