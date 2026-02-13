from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


@dataclass(frozen=True)
class DualRecording:
    samplerate: int
    mic_path: str
    loopback_path: str
    started_at_unix: float
    duration_s: float
    errors: dict[str, str] = field(default_factory=dict)


class DualRecorder:
    def __init__(
        self,
        mic_device_id: str,
        output_speaker_id: str,
        samplerate: int = 48000,
        blocksize: int = 2048,
        record_mic: bool = True,
        record_output: bool = True,
        loopback_source: str = "speaker",  # "speaker" | "mic"
        loopback_mic_id: str = "",
        loopback_mic_include_loopback: bool = True,
        live_window_s: float = 30.0,
    ) -> None:
        self._mic_device_id = mic_device_id
        self._output_speaker_id = output_speaker_id
        self._samplerate = int(samplerate)
        self._blocksize = int(blocksize)
        self._record_mic = bool(record_mic)
        self._record_output = bool(record_output)
        self._loopback_source = str(loopback_source)
        self._loopback_mic_id = str(loopback_mic_id)
        self._loopback_mic_include_loopback = bool(loopback_mic_include_loopback)

        self._live_window_s = float(live_window_s)

        self._lock = threading.Lock()
        self._mic_ring: deque[np.ndarray] = deque()
        self._loop_ring: deque[np.ndarray] = deque()
        self._mic_ring_samples_live = 0
        self._loop_ring_samples_live = 0
        self._mic_samples = 0
        self._loop_samples = 0

        self._mic_file: sf.SoundFile | None = None
        self._loop_file: sf.SoundFile | None = None
        self._mic_path: str | None = None
        self._loop_path: str | None = None

        self._stop = threading.Event()
        # Full audio is streamed directly to disk; keep only a small ring buffer for live mode.
        self._threads: list[threading.Thread] = []
        self._started_at_unix: float | None = None
        self._started_at_monotonic: float | None = None
        self._errors: dict[str, str] = {}

    @property
    def samplerate(self) -> int:
        return self._samplerate

    def start(self, mic_path: str, loopback_path: str) -> None:
        if self._threads:
            raise RuntimeError("Recording already started")

        if not self._record_mic and not self._record_output:
            raise RuntimeError("No streams selected")

        self._mic_path = str(mic_path) if self._record_mic else ""
        self._loop_path = str(loopback_path) if self._record_output else ""

        if self._record_mic:
            Path(self._mic_path).parent.mkdir(parents=True, exist_ok=True)
            self._mic_file = sf.SoundFile(self._mic_path, mode="w", samplerate=self._samplerate, channels=1, subtype="PCM_16")
        if self._record_output:
            Path(self._loop_path).parent.mkdir(parents=True, exist_ok=True)
            self._loop_file = sf.SoundFile(self._loop_path, mode="w", samplerate=self._samplerate, channels=1, subtype="PCM_16")

        mic = None
        loop = None
        if self._record_mic:
            mic = _get_microphone_by_id(self._mic_device_id, include_loopback=False)
        if self._record_output:
            if self._loopback_source == "mic":
                if not self._loopback_mic_id:
                    raise RuntimeError("Loopback microphone is not selected")
                loop = _get_microphone_by_id(
                    self._loopback_mic_id,
                    include_loopback=self._loopback_mic_include_loopback,
                )
            else:
                # Default: derive loopback endpoint from selected speaker id.
                loop = _get_microphone_by_id(self._output_speaker_id, include_loopback=True)

        self._stop.clear()
        with self._lock:
            self._mic_ring.clear()
            self._loop_ring.clear()
            self._mic_ring_samples_live = 0
            self._loop_ring_samples_live = 0
            self._mic_samples = 0
            self._loop_samples = 0
        self._errors = {}
        self._started_at_unix = time.time()
        self._started_at_monotonic = time.monotonic()

        threads: list[threading.Thread] = []
        if self._record_mic:
            threads.append(
                threading.Thread(
                    target=self._record_loop,
                    args=("mic", mic, None),
                    daemon=True,
                )
            )
        if self._record_output:
            threads.append(
                threading.Thread(
                    target=self._record_loop,
                    args=("loop", loop, None),
                    daemon=True,
                )
            )

        self._threads = threads
        for t in self._threads:
            t.start()

    def stop(self) -> DualRecording:
        if not self._threads:
            raise RuntimeError("Recording not started")

        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []

        try:
            if self._mic_file is not None:
                self._mic_file.flush()
                self._mic_file.close()
            if self._loop_file is not None:
                self._loop_file.flush()
                self._loop_file.close()
        finally:
            self._mic_file = None
            self._loop_file = None

        mic_path = str(self._mic_path or "")
        loop_path = str(self._loop_path or "")

        started_at_unix = float(self._started_at_unix or time.time())
        started_at_monotonic = float(self._started_at_monotonic or time.monotonic())
        duration_s = max(0.0, time.monotonic() - started_at_monotonic)

        return DualRecording(
            samplerate=self._samplerate,
            mic_path=mic_path,
            loopback_path=loop_path,
            started_at_unix=started_at_unix,
            duration_s=duration_s,
            errors=dict(self._errors),
        )

    def get_recent_window(self, kind: str, window_s: float | None = None) -> tuple[np.ndarray, int, int]:
        """Return (audio_mono, samplerate, window_start_sample_abs)."""
        if window_s is None:
            window_s = self._live_window_s
        need = max(0, int(round(float(window_s) * self._samplerate)))
        if need == 0:
            return np.zeros((0,), dtype=np.float32), self._samplerate, 0

        with self._lock:
            if kind == "mic":
                ring = list(self._mic_ring)
                total = self._mic_samples
            else:
                ring = list(self._loop_ring)
                total = self._loop_samples

        if not ring:
            return np.zeros((0,), dtype=np.float32), self._samplerate, max(0, total - need)

        chunks: list[np.ndarray] = []
        got = 0
        for c in reversed(ring):
            if got >= need:
                break
            chunks.append(c)
            got += int(c.shape[0])
        chunks.reverse()
        x = np.concatenate(chunks, axis=0)
        if x.ndim == 2:
            x = x[:, 0]
        if x.shape[0] > need:
            x = x[-need:]
        start_sample = max(0, total - x.shape[0])
        return np.asarray(x, dtype=np.float32), self._samplerate, start_sample

    def get_total_samples(self, kind: str) -> int:
        with self._lock:
            return self._mic_samples if kind == "mic" else self._loop_samples

    def _record_loop(self, kind: str, mic, _unused) -> None:  # noqa: ANN001
        try:
            if kind == "mic":
                assert mic is not None
                with mic.recorder(samplerate=self._samplerate, channels=1) as rec:
                    while not self._stop.is_set():
                        x = rec.record(numframes=self._blocksize)
                        fx = np.asarray(x, dtype=np.float32, order="C").copy()
                        if self._mic_file is not None:
                            self._mic_file.write(fx)
                        with self._lock:
                            self._push_live_chunk("mic", fx)
                            self._mic_samples += int(fx.shape[0])
            else:
                assert mic is not None
                # Loopback capture is exposed as a Microphone with isloopback=True.
                with mic.recorder(samplerate=self._samplerate, channels=1) as rec:
                    while not self._stop.is_set():
                        x = rec.record(numframes=self._blocksize)
                        fx = np.asarray(x, dtype=np.float32, order="C").copy()
                        if self._loop_file is not None:
                            self._loop_file.write(fx)
                        with self._lock:
                            self._push_live_chunk("loop", fx)
                            self._loop_samples += int(fx.shape[0])
        except Exception as e:
            # Keep the other stream running if possible.
            self._errors[kind] = f"{e}"
            return

    def _push_live_chunk(self, kind: str, fx: np.ndarray) -> None:
        """Append to live ring buffer and trim efficiently.

        Trimming used to recompute the ring length via sum(...) on every chunk,
        which can be surprisingly expensive and cause missed audio deadlines on
        Windows (soundcard then warns about discontinuities).
        """
        max_samples = int(round(self._live_window_s * self._samplerate * 1.5))
        n = int(fx.shape[0])

        if kind == "mic":
            ring = self._mic_ring
            ring.append(fx)
            self._mic_ring_samples_live += n
            while ring and self._mic_ring_samples_live > max_samples:
                old = ring.popleft()
                self._mic_ring_samples_live -= int(old.shape[0])
        else:
            ring = self._loop_ring
            ring.append(fx)
            self._loop_ring_samples_live += n
            while ring and self._loop_ring_samples_live > max_samples:
                old = ring.popleft()
                self._loop_ring_samples_live -= int(old.shape[0])


def _get_microphone_by_id(device_id: str, include_loopback: bool):
    m = sc.get_microphone(device_id, include_loopback=include_loopback)
    if not m:
        raise RuntimeError("Microphone not found")
    return m


def _concat_mono(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    x = np.concatenate(chunks, axis=0)
    if x.ndim == 2:
        x = x[:, 0]
    return np.asarray(x, dtype=np.float32)
