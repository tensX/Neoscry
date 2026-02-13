from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import time

from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QWidget,
)

from transcripto.asr.transcriber import Segment, WhisperTranscriber
from transcripto.audio.devices import (
    find_device_by_id,
    get_default_loopback_microphone_id,
    get_default_microphone_id,
    get_default_speaker_id,
    list_loopback_microphones,
    list_microphones,
    list_speakers,
)
from transcripto.audio.recorder import DualRecorder
from transcripto.transcript.formatters import TranscriptMeta, now_iso_local, to_json, to_pretty_txt
from transcripto.transcript.merge import merge_and_sort, merge_consecutive_same_speaker


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    mic_wav: Path
    loop_wav: Path
    transcript_txt: Path
    transcript_json: Path


class TranscribeWorker(QThread):
    progress = Signal(int)
    done = Signal(list, str, str)  # segments, txt, json
    failed = Signal(str)

    def __init__(
        self,
        rec_samplerate: int,
        mic_path: str,
        loop_path: str,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        mic_label: str,
        out_label: str,
        duration_s: float,
    ) -> None:
        super().__init__()
        self._rec_samplerate = rec_samplerate
        self._mic_path = mic_path
        self._loop_path = loop_path
        self._model_name = model_name
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._mic_label = mic_label
        self._out_label = out_label
        self._duration_s = duration_s

    def run(self) -> None:
        try:
            self.progress.emit(5)
            tr = WhisperTranscriber(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                compute_type=self._compute_type,
            )
            self.progress.emit(10)

            mic_segments = tr.transcribe_file(self._mic_path, speaker="me") if self._mic_path else []
            self.progress.emit(55)
            other_segments = tr.transcribe_file(self._loop_path, speaker="other") if self._loop_path else []
            self.progress.emit(85)

            merged: list[Segment] = merge_and_sort(mic_segments + other_segments)
            merged = merge_consecutive_same_speaker(merged)

            meta = TranscriptMeta(
                created_at=now_iso_local(),
                language=self._language,
                model=self._model_name,
                samplerate=int(self._rec_samplerate),
                mic_device=self._mic_label,
                output_device=self._out_label,
                duration_s=float(self._duration_s),
            )
            txt = to_pretty_txt(meta, merged)
            js = to_json(meta, merged)
            self.progress.emit(100)
            self.done.emit(merged, txt, js)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neoscry")

        self._settings = QSettings("transcripto", "transcripto")
        self._loading_settings = False

        self._inputs = list_microphones()
        self._outputs = list_speakers()

        self._recorder: DualRecorder | None = None
        self._last_txt: str | None = None
        self._last_json: str | None = None
        self._last_session: SessionPaths | None = None
        self._worker: TranscribeWorker | None = None
        self._live_enabled = False
        self._live_segments: list[Segment] = []
        self._live_worker: LiveWorker | None = None

        self._build_ui()
        self._apply_settings_to_ui()
        self._populate_devices()
        self._wire_persistence()

        # Ensure app starts idle.
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QGridLayout(root)

        devices_box = QGroupBox("Источники")
        dlay = QGridLayout(devices_box)

        self.mic_combo = QComboBox()
        self.out_combo = QComboBox()
        self.loop_mode = QComboBox()
        self.loop_mode.addItems(
            [
                "Устройство записи (любое)",
                "Loopback (устройство записи)",
                "Loopback от устройства воспроизведения",
            ]
        )
        self.chk_mic = QCheckBox("Записывать микрофон")
        self.chk_out = QCheckBox("Записывать выход")
        self.chk_mic.setChecked(True)
        self.chk_out.setChecked(True)

        dlay.addWidget(self.chk_mic, 0, 0)
        dlay.addWidget(self.mic_combo, 0, 1)
        dlay.addWidget(self.chk_out, 1, 0)
        dlay.addWidget(self.out_combo, 1, 1)
        dlay.addWidget(QLabel("Источник собеседника"), 2, 0)
        dlay.addWidget(self.loop_mode, 2, 1)

        model_box = QGroupBox("Распознавание")
        mlay = QGridLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "medium", "small"])
        self.lang_edit = QLineEdit("ru")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "float16", "int8"])
        mlay.addWidget(QLabel("Модель"), 0, 0)
        mlay.addWidget(self.model_combo, 0, 1)
        mlay.addWidget(QLabel("Язык"), 1, 0)
        mlay.addWidget(self.lang_edit, 1, 1)
        mlay.addWidget(QLabel("Устройство"), 2, 0)
        mlay.addWidget(self.device_combo, 2, 1)
        mlay.addWidget(QLabel("Compute"), 3, 0)
        mlay.addWidget(self.compute_combo, 3, 1)

        actions_box = QGroupBox("Действия")
        alay = QHBoxLayout(actions_box)
        self.btn_start = QPushButton("Старт записи")
        self.btn_stop = QPushButton("Стоп + транскрибировать")
        self.btn_export_txt = QPushButton("Экспорт TXT")
        self.btn_export_json = QPushButton("Экспорт JSON")
        self.btn_stop.setEnabled(False)
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)
        alay.addWidget(self.btn_start)
        alay.addWidget(self.btn_stop)
        alay.addWidget(self.btn_export_txt)
        alay.addWidget(self.btn_export_json)

        self.chk_live = QCheckBox("Лайв транскрипция")
        self.chk_live.setChecked(False)
        alay.addWidget(self.chk_live)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.transcript_view = QPlainTextEdit()
        self.transcript_view.setReadOnly(True)

        layout.addWidget(devices_box, 0, 0)
        layout.addWidget(model_box, 1, 0)
        layout.addWidget(actions_box, 2, 0)
        layout.addWidget(self.progress, 3, 0)
        layout.addWidget(self.transcript_view, 4, 0)

        self.setCentralWidget(root)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_export_txt.clicked.connect(self._on_export_txt)
        self.btn_export_json.clicked.connect(self._on_export_json)

        self.chk_mic.toggled.connect(self.mic_combo.setEnabled)
        self.chk_out.toggled.connect(self.out_combo.setEnabled)
        self.loop_mode.currentIndexChanged.connect(self._on_loop_mode_changed)
        self.chk_live.toggled.connect(lambda v: self._set("ui/live", bool(v)))

    def _wire_persistence(self) -> None:
        self.chk_mic.toggled.connect(lambda v: self._set("ui/record_mic", bool(v)))
        self.chk_out.toggled.connect(lambda v: self._set("ui/record_out", bool(v)))
        self.loop_mode.currentIndexChanged.connect(lambda i: self._set("ui/loop_mode", int(i)))
        self.model_combo.currentTextChanged.connect(lambda t: self._set("asr/model", str(t)))
        self.lang_edit.textChanged.connect(lambda t: self._set("asr/lang", str(t).strip()))
        self.device_combo.currentTextChanged.connect(lambda t: self._set("asr/device", str(t)))
        self.compute_combo.currentTextChanged.connect(lambda t: self._set("asr/compute", str(t)))
        self.mic_combo.currentIndexChanged.connect(self._save_device_selection)
        self.out_combo.currentIndexChanged.connect(self._save_device_selection)

    def _apply_settings_to_ui(self) -> None:
        self._loading_settings = True
        try:
            self.chk_mic.setChecked(bool(self._settings.value("ui/record_mic", True, bool)))
            self.chk_out.setChecked(bool(self._settings.value("ui/record_out", True, bool)))

            raw_loop_mode = self._settings.value("ui/loop_mode", 0)
            try:
                loop_mode = int(str(raw_loop_mode))
            except Exception:
                loop_mode = 0
            loop_mode = max(0, min(loop_mode, self.loop_mode.count() - 1))
            self.loop_mode.setCurrentIndex(loop_mode)

            model = str(self._settings.value("asr/model", "large-v3"))
            i = self.model_combo.findText(model)
            if i >= 0:
                self.model_combo.setCurrentIndex(i)

            lang = str(self._settings.value("asr/lang", "ru")).strip() or "ru"
            self.lang_edit.setText(lang)

            dev = str(self._settings.value("asr/device", "auto"))
            i = self.device_combo.findText(dev)
            if i >= 0:
                self.device_combo.setCurrentIndex(i)

            comp = str(self._settings.value("asr/compute", "auto"))
            i = self.compute_combo.findText(comp)
            if i >= 0:
                self.compute_combo.setCurrentIndex(i)

            self.mic_combo.setEnabled(self.chk_mic.isChecked())
            self.out_combo.setEnabled(self.chk_out.isChecked())

            self.chk_live.setChecked(bool(self._settings.value("ui/live", False, bool)))
        finally:
            self._loading_settings = False

    def _show_slot_error(self, title: str) -> None:
        QMessageBox.critical(self, title, traceback.format_exc())

    def _set(self, key: str, value) -> None:  # noqa: ANN001
        if self._loading_settings:
            return
        self._settings.setValue(key, value)

    def _save_device_selection(self) -> None:
        if self._loading_settings:
            return

        mic_id = self.mic_combo.currentData()
        if mic_id is not None:
            self._settings.setValue("devices/mic_id", str(mic_id))

        out_id = self.out_combo.currentData()
        if out_id is None:
            return

        idx = self.loop_mode.currentIndex()
        if idx == 0:
            self._settings.setValue("devices/other_input_id", str(out_id))
        elif idx == 1:
            self._settings.setValue("devices/loopback_input_id", str(out_id))
        else:
            self._settings.setValue("devices/speaker_id", str(out_id))

    def _populate_devices(self) -> None:
        self.mic_combo.clear()
        for d in self._inputs:
            self.mic_combo.addItem(d.label(), userData=d.id)

        saved_mic_id = str(self._settings.value("devices/mic_id", ""))
        if saved_mic_id:
            self._set_combo_to_device(self.mic_combo, saved_mic_id)
        else:
            default_mic = find_device_by_id(self._inputs, get_default_microphone_id())
            if default_mic:
                self._set_combo_to_device(self.mic_combo, default_mic.id)

        self._reload_output_devices()

    @staticmethod
    def _set_combo_to_device(combo: QComboBox, device_id: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == device_id:
                combo.setCurrentIndex(i)
                return

    def _on_start(self) -> None:
        try:
            self.transcript_view.setPlainText("Нажата кнопка 'Старт записи'...")

            if not self.chk_mic.isChecked() and not self.chk_out.isChecked():
                QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один поток записи")
                return
            if self.chk_mic.isChecked() and self.mic_combo.currentIndex() < 0:
                QMessageBox.warning(self, "Ошибка", "Выберите микрофон")
                return
            if self.chk_out.isChecked() and self.out_combo.currentIndex() < 0:
                QMessageBox.warning(self, "Ошибка", "Выберите выход")
                return

            mic_id = str(self.mic_combo.currentData()) if self.chk_mic.isChecked() else ""
            loop_id = str(self.out_combo.currentData()) if self.chk_out.isChecked() else ""

            self.progress.setValue(0)
            self._last_txt = None
            self._last_json = None
            self._last_session = None
            self.btn_export_txt.setEnabled(False)
            self.btn_export_json.setEnabled(False)

            self._recorder = DualRecorder(
                mic_device_id=mic_id,
                output_speaker_id=loop_id,
                record_mic=self.chk_mic.isChecked(),
                record_output=self.chk_out.isChecked(),
                loopback_source=self._loopback_source_mode(),
                loopback_mic_id=loop_id,
                loopback_mic_include_loopback=(self.loop_mode.currentIndex() == 1),
            )
            session = self._create_session_paths()
            self._last_session = session
            session.root.mkdir(parents=True, exist_ok=True)
            self._recorder.start(str(session.mic_wav), str(session.loop_wav))

            self._start_live_if_enabled()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.transcript_view.setPlainText(
                "Идет запись...\n"
                "Если включен лайв, черновик будет обновляться.\n"
                "Нажмите 'Стоп + транскрибировать' для финальной точности.\n"
                f"Сессия: {session.root}"
            )
        except Exception:
            self._recorder = None
            self._show_slot_error("Не удалось начать запись")

    def _loopback_source_mode(self) -> str:
        # 0/1 = record from an input device, 2 = speaker-derived loopback
        return "speaker" if self.loop_mode.currentIndex() == 2 else "mic"

    def _on_loop_mode_changed(self) -> None:
        self._reload_output_devices()

    def _reload_output_devices(self) -> None:
        # Depending on mode, show either loopback microphones (input devices)
        # or speakers (playback devices).
        self.out_combo.clear()
        idx = self.loop_mode.currentIndex()
        saved_id = ""
        if idx == 0:
            devs = list_microphones()
            default_id = get_default_microphone_id()
            saved_id = str(self._settings.value("devices/other_input_id", ""))
        elif idx == 1:
            devs = list_loopback_microphones()
            default_id = get_default_loopback_microphone_id()
            saved_id = str(self._settings.value("devices/loopback_input_id", ""))
        else:
            devs = list_speakers()
            default_id = get_default_speaker_id()
            saved_id = str(self._settings.value("devices/speaker_id", ""))

        # Keep for later lookups (optional, but consistent).
        self._outputs = devs

        for d in devs:
            self.out_combo.addItem(d.label(), userData=d.id)

        if saved_id:
            self._set_combo_to_device(self.out_combo, saved_id)
        else:
            default_dev = find_device_by_id(devs, default_id)
            if default_dev:
                self._set_combo_to_device(self.out_combo, default_dev.id)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        # Persist last known selections.
        self._save_device_selection()
        super().closeEvent(event)

    def _on_stop(self) -> None:
        if not self._recorder:
            return

        self.btn_stop.setEnabled(False)
        try:
            rec = self._recorder.stop()
        except Exception as e:
            self._recorder = None
            QMessageBox.critical(self, "Ошибка записи", str(e))
            self.btn_start.setEnabled(True)
            return
        finally:
            self._recorder = None

        self._stop_live()

        # Verify recorded files exist and are non-empty.
        if not self._last_session:
            QMessageBox.warning(self, "Пустая запись", "Не найдены файлы сессии")
            self.btn_start.setEnabled(True)
            return

        def _non_empty(p: Path) -> bool:
            try:
                return p.exists() and p.stat().st_size > 4096
            except Exception:
                return False

        has_mic = _non_empty(self._last_session.mic_wav)
        has_out = _non_empty(self._last_session.loop_wav)

        if not has_mic and not has_out:
            QMessageBox.warning(
                self,
                "Пустая запись",
                "Не удалось записать звук ни с микрофона, ни с выхода. "
                "Проверьте выбранные устройства и права доступа к микрофону.",
            )
            self.btn_start.setEnabled(True)
            return

        if getattr(rec, "errors", None):
            msg = "\n".join([f"{k}: {v}" for k, v in rec.errors.items()])
            QMessageBox.information(self, "Захват аудио: предупреждение", f"Есть ошибки в одном из потоков:\n{msg}")

        self.transcript_view.setPlainText(
            "Запись остановлена. Идет распознавание...\n"
            f"Сессия: {self._last_session.root if self._last_session else ''}\n"
        )

        mic_label = str(self.mic_combo.currentText())
        out_label = str(self.out_combo.currentText())
        model_name = str(self.model_combo.currentText())
        language = (self.lang_edit.text() or "ru").strip()
        device = str(self.device_combo.currentText())
        compute = str(self.compute_combo.currentText())

        mic_path = rec.mic_path if has_mic else ""
        out_path = rec.loopback_path if has_out else ""
        self._worker = TranscribeWorker(
            rec_samplerate=rec.samplerate,
            mic_path=mic_path,
            loop_path=out_path,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute,
            mic_label=mic_label,
            out_label=out_label,
            duration_s=rec.duration_s,
        )
        self._worker.progress.connect(self.progress.setValue)
        self._worker.done.connect(self._on_transcribed)
        self._worker.failed.connect(self._on_transcribe_failed)
        self._worker.start()

    def _start_live_if_enabled(self) -> None:
        self._live_enabled = bool(self.chk_live.isChecked())
        self._live_segments = []
        if not self._live_enabled or not self._recorder:
            return

        self._live_worker = LiveWorker(
            recorder=self._recorder,
            model_name=str(self.model_combo.currentText()),
            language=(self.lang_edit.text() or "ru").strip(),
            device=str(self.device_combo.currentText()),
            compute_type=str(self.compute_combo.currentText()),
            enable_mic=bool(self.chk_mic.isChecked()),
            enable_out=bool(self.chk_out.isChecked()),
            mic_label=str(self.mic_combo.currentText()),
            out_label=str(self.out_combo.currentText()),
            interval_s=3.0,
            window_s=30.0,
        )
        self._live_worker.update.connect(self._on_live_update)
        self._live_worker.start()

    def _stop_live(self) -> None:
        if self._live_worker is not None:
            self._live_worker.requestInterruption()
            self._live_worker.wait(2000)
            self._live_worker = None

    def _on_live_update(self, txt: str) -> None:
        # Live draft; final pass happens after stop.
        self.transcript_view.setPlainText(txt)

    def _on_transcribed(self, segments: list, txt: str, js: str) -> None:  # noqa: ANN001
        self._last_txt = txt
        self._last_json = js
        self.transcript_view.setPlainText(txt)
        self.btn_start.setEnabled(True)
        self.btn_export_txt.setEnabled(True)
        self.btn_export_json.setEnabled(True)

        if self._last_session:
            try:
                self._last_session.transcript_txt.write_text(txt, encoding="utf-8")
                self._last_session.transcript_json.write_text(js, encoding="utf-8")
            except Exception:
                pass

    def _on_transcribe_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка распознавания", message)
        self.btn_start.setEnabled(True)
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)

    def _on_export_txt(self) -> None:
        if not self._last_txt:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить TXT", "transcript.txt", "Text (*.txt)")
        if not path:
            return
        Path(path).write_text(self._last_txt, encoding="utf-8")

    def _on_export_json(self) -> None:
        if not self._last_json:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить JSON", "transcript.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(self._last_json, encoding="utf-8")

    def _create_session_paths(self) -> SessionPaths:
        root = Path(os.getcwd()) / "sessions"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_root = root / ts
        return SessionPaths(
            root=session_root,
            mic_wav=session_root / "mic.wav",
            loop_wav=session_root / "loopback.wav",
            transcript_txt=session_root / "transcript.txt",
            transcript_json=session_root / "transcript.json",
        )


class LiveWorker(QThread):
    update = Signal(str)

    def __init__(
        self,
        recorder: DualRecorder,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        enable_mic: bool,
        enable_out: bool,
        mic_label: str,
        out_label: str,
        interval_s: float,
        window_s: float,
    ) -> None:
        super().__init__()
        self._recorder = recorder
        self._model_name = model_name
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._enable_mic = enable_mic
        self._enable_out = enable_out
        self._mic_label = mic_label
        self._out_label = out_label
        self._interval_s = float(interval_s)
        self._window_s = float(window_s)

    def run(self) -> None:
        try:
            tr = WhisperTranscriber(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                compute_type=self._compute_type,
            )
            while not self.isInterruptionRequested():
                all_segments: list[Segment] = []
                if self._enable_mic:
                    x, sr, start_sample = self._recorder.get_recent_window("mic", window_s=self._window_s)
                    segs = tr.transcribe(x, sr, speaker="me")
                    for s in segs:
                        off = start_sample / sr
                        all_segments.append(Segment(start=s.start + off, end=s.end + off, text=s.text, speaker=s.speaker))
                if self._enable_out:
                    x, sr, start_sample = self._recorder.get_recent_window("loop", window_s=self._window_s)
                    segs = tr.transcribe(x, sr, speaker="other")
                    for s in segs:
                        off = start_sample / sr
                        all_segments.append(Segment(start=s.start + off, end=s.end + off, text=s.text, speaker=s.speaker))

                merged = merge_consecutive_same_speaker(merge_and_sort(all_segments))
                sr0 = int(getattr(self._recorder, "samplerate", 48000))
                meta = TranscriptMeta(
                    created_at=now_iso_local(),
                    language=self._language,
                    model=self._model_name,
                    samplerate=sr0,
                    mic_device=self._mic_label,
                    output_device=self._out_label,
                    duration_s=float(max(
                        self._recorder.get_total_samples("mic"),
                        self._recorder.get_total_samples("loop"),
                    ) / float(sr0)),
                )
                txt = to_pretty_txt(meta, merged)
                txt += "\n(Лайв черновик: финальная версия будет точнее после стопа)\n"
                self.update.emit(txt)

                # Sleep in small steps so stop is responsive.
                end = time.monotonic() + self._interval_s
                while time.monotonic() < end:
                    if self.isInterruptionRequested():
                        return
                    time.sleep(0.05)
        except Exception as e:
            self.update.emit(f"Лайв транскрипция остановлена из-за ошибки:\n{e}\n")
            return
