from __future__ import annotations

import os
import sys
import traceback
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
import time

from PySide6.QtCore import QObject, QProcess, QSettings, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
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
    meta_json: Path
    label: str
    slug: str
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
        parent: QObject | None,
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
        ui_lang: str = "ru",
    ) -> None:
        super().__init__(parent)
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
        self._ui_lang = str(ui_lang).strip().lower() or "ru"

    def run(self) -> None:
        try:
            self.progress.emit(5)
            tr = WhisperTranscriber(
                model_name=self._model_name,
                language=self._language,
                ui_lang=self._ui_lang,
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
                session_label="",
                language=self._language,
                model=self._model_name,
                samplerate=int(self._rec_samplerate),
                mic_device=self._mic_label,
                output_device=self._out_label,
                duration_s=float(self._duration_s),
            )
            txt = to_pretty_txt(meta, merged, ui_lang=self._ui_lang)
            js = to_json(meta, merged)
            self.progress.emit(100)
            self.done.emit(merged, txt, js)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neoscry")

        self._start_delay_s = 4

        self._current_session_label = ""

        self._settings = QSettings("Neoscry", "Neoscry")
        if not self._settings.allKeys():
            # Best-effort migration from the old app name.
            old = QSettings("transcripto", "transcripto")
            for k in old.allKeys():
                self._settings.setValue(k, old.value(k))
            self._settings.sync()
        self._loading_settings = False

        self._ui_lang = "ru"

        self._inputs = list_microphones()
        self._outputs = list_speakers()

        self._recorder: DualRecorder | None = None
        self._last_txt: str | None = None
        self._last_json: str | None = None
        self._last_session: SessionPaths | None = None
        self._worker: TranscribeWorker | None = None
        self._proc: QProcess | None = None
        self._live_enabled = False
        self._live_segments: list[Segment] = []
        self._live_worker: LiveWorker | None = None
        self._test_worker: AudioTestWorker | None = None

        self._rec_timer: QTimer | None = None
        self._rec_elapsed_s: int = 0

        self._test_ui_timer: QTimer | None = None
        self._test_ui_left_s: int = 0
        self._test_ui_phase: str = ""
        self._test_ui_button: QPushButton | None = None

        self._start_timer: QTimer | None = None
        self._start_countdown_s: int = 0
        self._pending_start_params: dict[str, object] | None = None

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

        self.devices_box = QGroupBox()
        dlay = QGridLayout(self.devices_box)

        self.mic_combo = QComboBox()
        self.out_combo = QComboBox()
        self.loop_mode = QComboBox()
        self.chk_mic = QCheckBox()
        self.chk_out = QCheckBox()
        self.chk_mic.setChecked(True)
        self.chk_out.setChecked(True)

        dlay.addWidget(self.chk_mic, 0, 0)
        dlay.addWidget(self.mic_combo, 0, 1)
        dlay.addWidget(self.chk_out, 1, 0)
        dlay.addWidget(self.out_combo, 1, 1)
        self.lbl_other_source = QLabel()
        dlay.addWidget(self.lbl_other_source, 2, 0)
        dlay.addWidget(self.loop_mode, 2, 1)

        self.btn_test_mic = QPushButton()
        self.btn_test_out = QPushButton()
        test_row = QWidget()
        test_lay = QHBoxLayout(test_row)
        test_lay.setContentsMargins(0, 0, 0, 0)
        test_lay.addWidget(self.btn_test_mic)
        test_lay.addWidget(self.btn_test_out)
        self.lbl_test = QLabel()
        dlay.addWidget(self.lbl_test, 3, 0)
        dlay.addWidget(test_row, 3, 1)

        self.model_box = QGroupBox()
        mlay = QGridLayout(self.model_box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "medium", "small"])
        self.lang_edit = QLineEdit("ru")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "float16", "int8"])
        self.lbl_model = QLabel()
        self.lbl_asr_lang = QLabel()
        self.lbl_device = QLabel()
        self.lbl_compute = QLabel()
        mlay.addWidget(self.lbl_model, 0, 0)
        mlay.addWidget(self.model_combo, 0, 1)
        mlay.addWidget(self.lbl_asr_lang, 1, 0)
        mlay.addWidget(self.lang_edit, 1, 1)
        mlay.addWidget(self.lbl_device, 2, 0)
        mlay.addWidget(self.device_combo, 2, 1)
        mlay.addWidget(self.lbl_compute, 3, 0)
        mlay.addWidget(self.compute_combo, 3, 1)

        self.actions_box = QGroupBox()
        alay = QHBoxLayout(self.actions_box)
        self.btn_start = QPushButton()
        self.btn_stop = QPushButton()
        self.btn_export_txt = QPushButton()
        self.btn_export_json = QPushButton()
        self.btn_stop.setEnabled(False)
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)
        alay.addWidget(self.btn_start)
        alay.addWidget(self.btn_stop)
        alay.addWidget(self.btn_export_txt)
        alay.addWidget(self.btn_export_json)

        self.btn_settings = QToolButton()
        self.btn_settings.setText("⚙")
        self.btn_settings.setToolTip("Settings")
        self.btn_settings.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        alay.addWidget(self.btn_settings)

        self.settings_menu = QMenu(self)

        self.act_transcribe_session = QAction(self)
        self.settings_menu.addAction(self.act_transcribe_session)

        self.act_asr_settings = QAction(self)
        self.settings_menu.addAction(self.act_asr_settings)

        self.act_check_gpu = QAction(self)
        self.settings_menu.addAction(self.act_check_gpu)

        self.act_install_cuda = QAction(self)
        self.settings_menu.addAction(self.act_install_cuda)

        self.settings_menu.addSeparator()

        self.act_live = QAction(self)
        self.act_live.setCheckable(True)
        self.settings_menu.addAction(self.act_live)

        self.act_on_top = QAction(self)
        self.act_on_top.setCheckable(True)
        self.settings_menu.addAction(self.act_on_top)

        self.settings_menu.addSeparator()
        self.lang_menu = QMenu(self)
        self.settings_menu.addMenu(self.lang_menu)
        self.lang_group = QActionGroup(self)
        self.lang_group.setExclusive(True)
        self.act_lang_ru = QAction(self)
        self.act_lang_ru.setCheckable(True)
        self.act_lang_en = QAction(self)
        self.act_lang_en.setCheckable(True)
        self.lang_group.addAction(self.act_lang_ru)
        self.lang_group.addAction(self.act_lang_en)
        self.lang_menu.addAction(self.act_lang_ru)
        self.lang_menu.addAction(self.act_lang_en)

        self.btn_settings.setMenu(self.settings_menu)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.transcript_view = QPlainTextEdit()
        self.transcript_view.setReadOnly(True)

        layout.addWidget(self.devices_box, 0, 0)
        layout.addWidget(self.actions_box, 1, 0)
        layout.addWidget(self.progress, 2, 0)
        layout.addWidget(self.transcript_view, 3, 0)

        self.setCentralWidget(root)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_export_txt.clicked.connect(self._on_export_txt)
        self.btn_export_json.clicked.connect(self._on_export_json)
        self.btn_test_mic.clicked.connect(self._on_test_mic)
        self.btn_test_out.clicked.connect(self._on_test_out)

        self.chk_mic.toggled.connect(self.mic_combo.setEnabled)
        self.chk_out.toggled.connect(self.out_combo.setEnabled)
        self.chk_mic.toggled.connect(lambda _v: self._update_test_buttons())
        self.chk_out.toggled.connect(lambda _v: self._update_test_buttons())
        self.mic_combo.currentIndexChanged.connect(lambda _i: self._update_test_buttons())
        self.out_combo.currentIndexChanged.connect(lambda _i: self._update_test_buttons())
        self.loop_mode.currentIndexChanged.connect(self._on_loop_mode_changed)

        self.act_live.toggled.connect(self._on_live_toggled)
        self.act_on_top.toggled.connect(self._on_always_on_top)
        self.act_on_top.toggled.connect(lambda v: self._set("ui/always_on_top", bool(v)))
        self.act_lang_ru.triggered.connect(lambda: self._set_ui_language("ru"))
        self.act_lang_en.triggered.connect(lambda: self._set_ui_language("en"))

        self.act_transcribe_session.triggered.connect(self._on_transcribe_existing_session)
        self.act_asr_settings.triggered.connect(self._on_open_asr_settings)
        self.act_check_gpu.triggered.connect(self._on_check_gpu)
        self.act_install_cuda.triggered.connect(self._on_install_cuda)

        self._set_start_button_state("idle")

        # Apply initial texts (language is loaded from settings later).
        self._apply_ui_texts("ru")

        # ASR settings dialog (model/language/device/compute).
        self.asr_dialog = QDialog(self)
        self.asr_dialog.setModal(False)
        v = QVBoxLayout(self.asr_dialog)
        v.addWidget(self.model_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.asr_dialog.close)
        v.addWidget(buttons)

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

            ui_lang = str(self._settings.value("ui/lang", "ru")).strip().lower()
            if ui_lang not in ("ru", "en"):
                ui_lang = "ru"
            self._ui_lang = ui_lang

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

            self.act_live.setChecked(bool(self._settings.value("ui/live", False, bool)))
            self.act_on_top.setChecked(bool(self._settings.value("ui/always_on_top", False, bool)))

            self.act_lang_ru.setChecked(self._ui_lang == "ru")
            self.act_lang_en.setChecked(self._ui_lang == "en")
        finally:
            self._loading_settings = False

        self._apply_ui_texts(self._ui_lang)

        self._update_test_buttons()

    def _show_slot_error(self, title: str) -> None:
        QMessageBox.critical(self, title, traceback.format_exc())

    def _set(self, key: str, value) -> None:  # noqa: ANN001
        if self._loading_settings:
            return
        self._settings.setValue(key, value)

    def _tr(self, ru: str, en: str) -> str:
        return en if getattr(self, "_ui_lang", "ru") == "en" else ru

    def _set_ui_language(self, lang: str) -> None:
        lang = str(lang).strip().lower()
        if lang not in ("ru", "en"):
            lang = "ru"

        self._ui_lang = lang
        self._set("ui/lang", lang)

        # Keep menu state consistent.
        self.act_lang_ru.setChecked(lang == "ru")
        self.act_lang_en.setChecked(lang == "en")

        self._apply_ui_texts(lang)

    def _apply_ui_texts(self, lang: str) -> None:
        lang = str(lang).strip().lower()
        if lang not in ("ru", "en"):
            lang = "ru"
        self._ui_lang = lang

        self.devices_box.setTitle(self._tr("Источники", "Sources"))
        self.model_box.setTitle(self._tr("Распознавание", "Transcription"))
        self.actions_box.setTitle(self._tr("Действия", "Actions"))

        self.chk_mic.setText(self._tr("Записывать микрофон", "Record microphone"))
        self.chk_out.setText(self._tr("Записывать выход", "Record output"))
        self.lbl_other_source.setText(self._tr("Источник собеседника", "Other side source"))
        self.lbl_test.setText(self._tr("Проверка", "Test"))

        self.lbl_model.setText(self._tr("Модель", "Model"))
        self.lbl_asr_lang.setText(self._tr("Язык", "Language"))
        self.lbl_device.setText(self._tr("Устройство", "Device"))
        self.lbl_compute.setText("Compute")

        # Loopback source modes (keep current index).
        idx = self.loop_mode.currentIndex()
        self.loop_mode.blockSignals(True)
        try:
            self.loop_mode.clear()
            if lang == "en":
                items = [
                    "Input device (any)",
                    "Loopback (input device)",
                    "Loopback from playback device",
                ]
            else:
                items = [
                    "Устройство записи (любое)",
                    "Loopback (устройство записи)",
                    "Loopback от устройства воспроизведения",
                ]
            self.loop_mode.addItems(items)
            if idx >= 0:
                self.loop_mode.setCurrentIndex(max(0, min(idx, self.loop_mode.count() - 1)))
        finally:
            self.loop_mode.blockSignals(False)

        # Buttons
        self.btn_stop.setText(self._tr("Стоп + транскрибировать", "Stop + transcribe"))
        self.btn_export_txt.setText(self._tr("Экспорт TXT", "Export TXT"))
        self.btn_export_json.setText(self._tr("Экспорт JSON", "Export JSON"))

        # Test button base labels are handled in _set_test_button_visual("idle"...)
        self._set_test_button_visual(self.btn_test_mic, "idle", 0)
        self._set_test_button_visual(self.btn_test_out, "idle", 0)

        # Settings menu
        self.btn_settings.setToolTip(self._tr("Настройки", "Settings"))
        self.act_transcribe_session.setText(self._tr("Повторная транскрипция...", "Re-transcribe session..."))
        self.act_asr_settings.setText(self._tr("Распознавание...", "Transcription settings..."))
        self.act_check_gpu.setText(self._tr("Проверить GPU (CUDA)", "Check GPU (CUDA)"))
        self.act_install_cuda.setText(self._tr("Установить CUDA runtime (pip)", "Install CUDA runtime (pip)"))
        self.act_live.setText(self._tr("Лайв транскрипция (черновик)", "Live transcription (draft)"))
        self.act_on_top.setText(self._tr("Поверх всех окон", "Always on top"))
        self.lang_menu.setTitle(self._tr("Язык интерфейса", "UI language"))
        self.act_lang_ru.setText("Русский")
        self.act_lang_en.setText("English")

        try:
            self.asr_dialog.setWindowTitle(self._tr("Распознавание", "Transcription"))
        except Exception:
            pass

        # Start button may be in a non-idle state (waiting/recording).
        if self._recorder is not None:
            self._set_start_button_state("recording", left=self._rec_elapsed_s)
        elif self._start_timer is not None:
            self._set_start_button_state("waiting", left=self._start_countdown_s)
        else:
            self._set_start_button_state("idle")

    def _on_live_toggled(self, enabled: bool) -> None:
        if self._loading_settings:
            return

        self._set("ui/live", bool(enabled))

        # Allow toggling live during recording.
        if self._recorder is None:
            return
        if enabled:
            if self._live_worker is None:
                self._start_live_if_enabled()
        else:
            self._stop_live()

    def _on_open_asr_settings(self) -> None:
        self.asr_dialog.show()
        self.asr_dialog.raise_()
        self.asr_dialog.activateWindow()

    def _on_transcribe_existing_session(self) -> None:
        try:
            if self._recorder is not None or self._start_timer is not None or self._test_worker is not None:
                QMessageBox.information(
                    self,
                    self._tr("Транскрипция", "Transcription"),
                    self._tr(
                        "Остановите запись/тест перед повторной транскрипцией",
                        "Stop recording/testing before re-transcribing",
                    ),
                )
                return

            if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
                QMessageBox.information(
                    self,
                    self._tr("Транскрипция", "Transcription"),
                    self._tr(
                        "Транскрипция уже выполняется",
                        "Transcription is already running",
                    ),
                )
                return

            base = Path(os.getcwd()) / "sessions"
            start_dir = str(base) if base.exists() else os.getcwd()
            session_dir = QFileDialog.getExistingDirectory(
                self,
                self._tr("Выберите папку сессии", "Select a session folder"),
                start_dir,
            )
            if not session_dir:
                return

            root = Path(session_dir)
            label, slug = self._load_session_label_and_slug(root)
            session = SessionPaths(
                root=root,
                meta_json=root / "meta.json",
                label=label,
                slug=slug,
                mic_wav=root / "mic.wav",
                loop_wav=root / "loopback.wav",
                transcript_txt=root / "transcript.txt",
                transcript_json=root / "transcript.json",
            )
            self._last_session = session

            self._start_transcription_for_session(session)
        except Exception:
            QMessageBox.critical(self, self._tr("Ошибка", "Error"), traceback.format_exc())

    def _on_check_gpu(self) -> None:
        # Quick diagnostics for CUDA DLLs.
        try:
            # First, try to load via runtime wheels search path (Windows).
            try:
                from transcripto.windows_cuda import ensure_cublas12_dll_available

                if ensure_cublas12_dll_available():
                    QMessageBox.information(
                        self,
                        self._tr("GPU", "GPU"),
                        self._tr(
                            "cublas64_12.dll загружается (OK).",
                            "cublas64_12.dll is loadable (OK).",
                        ),
                    )
                    return
            except Exception:
                pass

            p = QProcess(self)
            p.setProgram("cmd")
            p.setArguments(["/c", "where cublas64_12.dll"])
            p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            p.start()
            p.waitForFinished(5000)
            out = bytes(p.readAll().data()).decode("utf-8", errors="replace").strip()
            if p.exitCode() == 0 and out:
                QMessageBox.information(
                    self,
                    self._tr("GPU", "GPU"),
                    self._tr(
                        f"Найдено cublas64_12.dll:\n{out}",
                        f"Found cublas64_12.dll:\n{out}",
                    ),
                )
            else:
                QMessageBox.warning(
                    self,
                    self._tr("GPU", "GPU"),
                    self._tr(
                        "cublas64_12.dll не найден.\nПопробуйте 'Установить CUDA runtime (pip)'.",
                        "cublas64_12.dll was not found.\nTry 'Install CUDA runtime (pip)'.",
                    ),
                )
        except Exception:
            QMessageBox.critical(self, self._tr("Ошибка", "Error"), traceback.format_exc())

    def _on_install_cuda(self) -> None:
        # Best-effort install of CUDA runtime wheels via pip (Windows).
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(
                self,
                self._tr("Установка", "Install"),
                self._tr("Уже выполняется процесс", "A process is already running"),
            )
            return

        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(os.getcwd())
        self._proc.setProgram(sys.executable)
        self._proc.setArguments(
            [
                "-m",
                "pip",
                "install",
                "--upgrade",
                "nvidia-cublas-cu12",
                "nvidia-cuda-runtime-cu12",
            ]
        )
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_proc_output)
        self._proc.readyReadStandardError.connect(self._on_proc_output)
        self._proc.finished.connect(self._on_install_cuda_finished)

        self.transcript_view.setPlainText(
            self._tr(
                "Установка CUDA runtime (pip)...\nПосле установки нажмите 'Проверить GPU (CUDA)'.\n",
                "Installing CUDA runtime (pip)...\nAfter install, click 'Check GPU (CUDA)'.\n",
            )
        )
        self.progress.setRange(0, 0)
        self._proc.start()

    def _on_install_cuda_finished(self, exit_code: int, _status) -> None:  # noqa: ANN001
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        proc = self._proc
        self._proc = None

        msg = ""
        if proc is not None:
            try:
                ba = proc.readAll()
                msg = bytes(ba.data()).decode("utf-8", errors="replace") if not ba.isEmpty() else ""
            except Exception:
                msg = ""

        if exit_code != 0:
            QMessageBox.critical(
                self,
                self._tr("Установка", "Install"),
                self._tr(
                    f"pip завершился с кодом {exit_code}.\n{msg}",
                    f"pip exited with code {exit_code}.\n{msg}",
                ),
            )
            return

        QMessageBox.information(
            self,
            self._tr("Установка", "Install"),
            self._tr(
                "CUDA runtime установлен. Перезапустите приложение и проверьте GPU через 'Проверить GPU (CUDA)'.",
                "CUDA runtime installed. Restart the app and run 'Check GPU (CUDA)'.",
            ),
        )

    def _start_transcription_for_session(self, session: SessionPaths) -> None:
        try:
            self._start_transcription_for_session_impl(session)
        except Exception:
            QMessageBox.critical(self, self._tr("Ошибка", "Error"), traceback.format_exc())

    def _start_transcription_for_session_impl(self, session: SessionPaths) -> None:
        def _non_empty(p: Path) -> bool:
            try:
                return p.exists() and p.stat().st_size > 4096
            except Exception:
                return False

        has_mic = _non_empty(session.mic_wav)
        has_out = _non_empty(session.loop_wav)

        if not has_mic and not has_out:
            QMessageBox.warning(
                self,
                self._tr("Пустая запись", "Empty recording"),
                self._tr(
                    "Не найдены непустые файлы mic.wav/loopback.wav в выбранной сессии",
                    "No non-empty mic.wav/loopback.wav found in the selected session",
                ),
            )
            return

        # Derive samplerate/duration from files.
        sr = 48000
        dur = 0.0
        for p in (session.mic_wav, session.loop_wav):
            if not _non_empty(p):
                continue
            try:
                info = sf.info(str(p))
                if getattr(info, "samplerate", None):
                    sr = int(info.samplerate)
                if getattr(info, "frames", None) and sr > 0:
                    dur = max(dur, float(info.frames) / float(sr))
            except Exception:
                pass

        session_root = str(session.root)
        self.transcript_view.setPlainText(
            self._tr(
                f"Повторная транскрипция...\nСессия: {session_root}\n",
                f"Re-transcribing...\nSession: {session_root}\n",
            )
        )

        mic_path = str(session.mic_wav) if has_mic else ""
        out_path = str(session.loop_wav) if has_out else ""

        self.progress.setValue(0)
        self._last_txt = None
        self._last_json = None
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)

        mic_label = self._tr("микрофон (из файла)", "microphone (from file)") if has_mic else ""
        out_label = self._tr("собеседник (из файла)", "other side (from file)") if has_out else ""

        model_name = str(self.model_combo.currentText())
        language = (self.lang_edit.text() or "ru").strip()
        device = str(self.device_combo.currentText())
        compute = str(self.compute_combo.currentText())

        self.btn_start.setEnabled(False)

        self._start_transcription_process(
            session=session,
            mic_path=mic_path,
            out_path=out_path,
            samplerate=sr,
            duration_s=dur,
            mic_label=mic_label,
            out_label=out_label,
        )

    def _on_transcribe_finished(self) -> None:
        # Keep last results, but drop the thread reference.
        self._worker = None
        self.btn_start.setEnabled(True)

    def _start_transcription_process(
        self,
        session: SessionPaths,
        mic_path: str,
        out_path: str,
        samplerate: int,
        duration_s: float,
        mic_label: str,
        out_label: str,
    ) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(
                self,
                self._tr("Транскрипция", "Transcription"),
                self._tr("Транскрипция уже выполняется", "Transcription is already running"),
            )
            return

        # Use an external process to isolate native crashes in ASR backends.
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(os.getcwd())

        py = sys.executable
        script = str(Path(os.getcwd()) / "transcribe.py")

        args = [
            script,
            "--mic",
            mic_path,
            "--loop",
            out_path,
            "--model",
            str(self.model_combo.currentText()),
            "--lang",
            (self.lang_edit.text() or "ru").strip(),
            "--ui-lang",
            str(self._ui_lang),
            "--device",
            str(self.device_combo.currentText()),
            "--compute",
            str(self.compute_combo.currentText()),
            "--mic-label",
            mic_label,
            "--out-label",
            out_label,
            "--session-label",
            str(session.label),
            "--out-txt",
            str(session.transcript_txt),
            "--out-json",
            str(session.transcript_json),
        ]

        # UI: show busy state.
        self.progress.setRange(0, 0)
        self.btn_start.setEnabled(False)
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)

        self._proc.setProgram(py)
        self._proc.setArguments(args)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        self._proc.finished.connect(self._on_proc_finished)
        self._proc.readyReadStandardOutput.connect(self._on_proc_output)
        self._proc.readyReadStandardError.connect(self._on_proc_output)

        self._proc.start()

    def _on_proc_output(self) -> None:
        if self._proc is None:
            return
        try:
            out_ba = self._proc.readAllStandardOutput()
            err_ba = self._proc.readAllStandardError()
            data = (bytes(out_ba.data()).decode("utf-8", errors="replace") if not out_ba.isEmpty() else "")
            data += (bytes(err_ba.data()).decode("utf-8", errors="replace") if not err_ba.isEmpty() else "")
        except Exception:
            return
        if data.strip():
            # Non-intrusive: append to the transcript view.
            cur = self.transcript_view.toPlainText()
            if cur and not cur.endswith("\n"):
                cur += "\n"
            self.transcript_view.setPlainText(cur + data.strip() + "\n")

    def _on_proc_finished(self, exit_code: int, _status) -> None:  # noqa: ANN001
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        proc = self._proc
        self._proc = None

        # Re-enable start.
        self.btn_start.setEnabled(True)

        if exit_code != 0:
            msg = ""
            if proc is not None:
                try:
                    ba = proc.readAll()
                    msg = bytes(ba.data()).decode("utf-8", errors="replace") if not ba.isEmpty() else ""
                except Exception:
                    msg = ""
            QMessageBox.critical(
                self,
                self._tr("Ошибка распознавания", "Transcription error"),
                self._tr(
                    f"Процесс распознавания завершился с кодом {exit_code}.\n{msg}",
                    f"Transcription process exited with code {exit_code}.\n{msg}",
                ),
            )
            self.btn_export_txt.setEnabled(False)
            self.btn_export_json.setEnabled(False)
            return

        # Load results from last session.
        if self._last_session:
            try:
                txt = self._last_session.transcript_txt.read_text(encoding="utf-8")
                js = self._last_session.transcript_json.read_text(encoding="utf-8")
                self._last_txt = txt
                self._last_json = js
                self.transcript_view.setPlainText(txt)
                self.btn_export_txt.setEnabled(True)
                self.btn_export_json.setEnabled(True)
                return
            except Exception as e:
                QMessageBox.critical(
                    self,
                    self._tr("Ошибка", "Error"),
                    self._tr(
                        f"Не удалось прочитать результаты распознавания: {e}",
                        f"Failed to read transcription outputs: {e}",
                    ),
                )

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
            if self._start_timer is not None:
                return
            if self._recorder is not None:
                return

            self.transcript_view.setPlainText(
                self._tr(
                    "Нажата кнопка 'Старт записи'...",
                    "Start button pressed...",
                )
            )

            if not self.chk_mic.isChecked() and not self.chk_out.isChecked():
                QMessageBox.warning(
                    self,
                    self._tr("Ошибка", "Error"),
                    self._tr("Выберите хотя бы один поток записи", "Select at least one recording stream"),
                )
                return
            if self.chk_mic.isChecked() and self.mic_combo.currentIndex() < 0:
                QMessageBox.warning(self, self._tr("Ошибка", "Error"), self._tr("Выберите микрофон", "Select a microphone"))
                return
            if self.chk_out.isChecked() and self.out_combo.currentIndex() < 0:
                QMessageBox.warning(self, self._tr("Ошибка", "Error"), self._tr("Выберите выход", "Select an output source"))
                return

            mic_id = str(self.mic_combo.currentData()) if self.chk_mic.isChecked() else ""
            loop_id = str(self.out_combo.currentData()) if self.chk_out.isChecked() else ""

            self.progress.setValue(0)
            self._last_txt = None
            self._last_json = None
            self._last_session = None
            self.btn_export_txt.setEnabled(False)
            self.btn_export_json.setEnabled(False)

            label = self._prompt_session_label()
            if label is None:
                return

            self._current_session_label = label

            session = self._create_session_paths(label)
            self._last_session = session
            session.root.mkdir(parents=True, exist_ok=True)

            # Persist session metadata for later discovery.
            try:
                meta = {
                    "created_at": now_iso_local(),
                    "label": session.label,
                    "slug": session.slug,
                }
                session.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            loopback_source = self._loopback_source_mode()
            loopback_mic_include_loopback = (self.loop_mode.currentIndex() == 1)
            self._pending_start_params = {
                "mic_id": mic_id,
                "loop_id": loop_id,
                "record_mic": bool(self.chk_mic.isChecked()),
                "record_output": bool(self.chk_out.isChecked()),
                "loopback_source": loopback_source,
                "loopback_mic_id": loop_id,
                "loopback_mic_include_loopback": loopback_mic_include_loopback,
                "mic_path": str(session.mic_wav),
                "loop_path": str(session.loop_wav),
                "session_root": str(session.root),
            }
            self._start_countdown_s = int(self._start_delay_s)
            self.btn_stop.setEnabled(False)
            self._set_start_button_state("waiting", left=self._start_countdown_s)
            self._update_test_buttons()
            self._start_timer = QTimer(self)
            self._start_timer.setInterval(1000)
            self._start_timer.timeout.connect(self._on_start_tick)
            self._on_start_tick()
            self._start_timer.start()
        except Exception:
            self._recorder = None
            self._show_slot_error(self._tr("Не удалось начать запись", "Failed to start recording"))
            self._cancel_pending_start()

    def _prompt_session_label(self) -> str | None:
        last = str(self._settings.value("ui/last_session_label", "")).strip()
        if self._ui_lang == "en":
            items = [
                "Interview",
                "Psychologist",
                "Meeting",
                "Call",
                "Lecture",
                "Other",
            ]
            title = "Session label"
            prompt = "What are you recording?"
        else:
            items = [
                "Интервью",
                "Психолог",
                "Встреча",
                "Звонок",
                "Лекция",
                "Другое",
            ]
            title = "Метка сессии"
            prompt = "Что записываем?"

        if last and last not in items:
            items.insert(0, last)

        value, ok = QInputDialog.getItem(self, title, prompt, items, current=0, editable=True)
        if not ok:
            return None
        label = str(value).strip()
        if not label:
            # Require a label so the folder name is meaningful.
            QMessageBox.warning(
                self,
                self._tr("Ошибка", "Error"),
                self._tr("Введите метку сессии", "Please enter a session label"),
            )
            return None

        self._settings.setValue("ui/last_session_label", label)
        return label

    @staticmethod
    def _translit_ru(s: str) -> str:
        m = {
            "а": "a",
            "б": "b",
            "в": "v",
            "г": "g",
            "д": "d",
            "е": "e",
            "ё": "e",
            "ж": "zh",
            "з": "z",
            "и": "i",
            "й": "y",
            "к": "k",
            "л": "l",
            "м": "m",
            "н": "n",
            "о": "o",
            "п": "p",
            "р": "r",
            "с": "s",
            "т": "t",
            "у": "u",
            "ф": "f",
            "х": "h",
            "ц": "ts",
            "ч": "ch",
            "ш": "sh",
            "щ": "sch",
            "ъ": "",
            "ы": "y",
            "ь": "",
            "э": "e",
            "ю": "yu",
            "я": "ya",
        }
        out = []
        for ch in s:
            lo = ch.lower()
            if lo in m:
                out.append(m[lo])
            else:
                out.append(ch)
        return "".join(out)

    def _slugify_label(self, label: str) -> str:
        s = self._translit_ru(str(label))
        s = s.strip().lower()
        out: list[str] = []
        prev_sep = False
        for ch in s:
            if "a" <= ch <= "z" or "0" <= ch <= "9":
                out.append(ch)
                prev_sep = False
                continue
            if ch in (" ", "-", "_", "."):
                if not prev_sep:
                    out.append("-")
                    prev_sep = True
                continue
            # drop everything else
        slug = "".join(out).strip("-")
        return slug or "session"

    def _load_session_label_and_slug(self, root: Path) -> tuple[str, str]:
        # Prefer meta.json (created by this app).
        try:
            meta_path = root / "meta.json"
            if meta_path.exists():
                obj = json.loads(meta_path.read_text(encoding="utf-8"))
                label = str(obj.get("label", "")).strip()
                slug = str(obj.get("slug", "")).strip()
                if not slug and label:
                    slug = self._slugify_label(label)
                return label, slug
        except Exception:
            pass

        # Fallback: parse folder name "<ts>__<slug>".
        try:
            name = root.name
            if "__" in name:
                slug = name.split("__", 1)[1].strip()
                return "", slug
        except Exception:
            pass

        return "", ""

    def _on_start_tick(self) -> None:
        if self._start_timer is None or self._pending_start_params is None:
            return

        left = int(self._start_countdown_s)
        session_root = str(self._pending_start_params.get("session_root", ""))
        if left <= 0:
            try:
                self._start_timer.stop()
            finally:
                self._start_timer = None

            p = self._pending_start_params
            self._pending_start_params = None
            self._start_countdown_s = 0

            self._start_recording_now(
                mic_id=str(p.get("mic_id", "")),
                loop_id=str(p.get("loop_id", "")),
                record_mic=bool(p.get("record_mic", False)),
                record_output=bool(p.get("record_output", False)),
                mic_path=str(p.get("mic_path", "")),
                loop_path=str(p.get("loop_path", "")),
                loopback_source=str(p.get("loopback_source", "mic")),
                loopback_mic_include_loopback=bool(p.get("loopback_mic_include_loopback", False)),
            )
            return

        if self._ui_lang == "en":
            self.transcript_view.setPlainText(
                f"Recording starts in {left}s...\n"
                f"Session: {session_root}\n"
                "(You have time to open the app/tab/call you want to capture)"
            )
        else:
            self.transcript_view.setPlainText(
                "Запись начнется через "
                f"{left} сек...\n"
                f"Сессия: {session_root}\n"
                "(Можно успеть включить нужный звук/вкладку/звонок)"
            )
        self._set_start_button_state("waiting", left=left)
        self._start_countdown_s = left - 1

    def _start_recording_now(
        self,
        mic_id: str,
        loop_id: str,
        record_mic: bool,
        record_output: bool,
        mic_path: str,
        loop_path: str,
        loopback_source: str,
        loopback_mic_include_loopback: bool,
    ) -> None:
        self._recorder = DualRecorder(
            mic_device_id=mic_id,
            output_speaker_id=loop_id,
            record_mic=record_mic,
            record_output=record_output,
            loopback_source=str(loopback_source),
            loopback_mic_id=loop_id,
            loopback_mic_include_loopback=bool(loopback_mic_include_loopback),
        )
        self._recorder.start(mic_path, loop_path)

        self._start_live_if_enabled()

        self._rec_elapsed_s = 0
        if self._rec_timer is None:
            self._rec_timer = QTimer(self)
            self._rec_timer.setInterval(1000)
            self._rec_timer.timeout.connect(self._on_record_tick)
        self._rec_timer.start()

        self._set_start_button_state("recording")
        self.btn_stop.setEnabled(True)
        self._update_test_buttons()
        session_root = str(self._last_session.root) if self._last_session else ""
        if self._ui_lang == "en":
            self.transcript_view.setPlainText(
                "Recording...\n"
                "If live mode is enabled, the draft will update.\n"
                "Click 'Stop + transcribe' for the final (more accurate) result.\n"
                f"Session: {session_root}"
            )
        else:
            self.transcript_view.setPlainText(
                "Идет запись...\n"
                "Если включен лайв, черновик будет обновляться.\n"
                "Нажмите 'Стоп + транскрибировать' для финальной точности.\n"
                f"Сессия: {session_root}"
            )

    def _on_record_tick(self) -> None:
        if self._recorder is None:
            return
        self._rec_elapsed_s += 1
        mm = self._rec_elapsed_s // 60
        ss = self._rec_elapsed_s % 60
        self._set_start_button_state("recording", left=(mm * 60 + ss))

    def _cancel_pending_start(self) -> None:
        if self._start_timer is not None:
            try:
                self._start_timer.stop()
            finally:
                self._start_timer = None
        self._pending_start_params = None
        self._start_countdown_s = 0
        self._set_start_button_state("idle")
        self._update_test_buttons()

    def _set_start_button_state(self, state: str, left: int | None = None) -> None:
        state = str(state)

        def _refresh(btn: QPushButton) -> None:
            try:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            except Exception:
                pass
            btn.update()

        if state == "waiting":
            n = int(left or 0)
            # Keep enabled so styles apply consistently across native themes.
            self.btn_start.setEnabled(True)
            if self._ui_lang == "en":
                self.btn_start.setText(f"Starting in {max(0, n)}s")
            else:
                self.btn_start.setText(f"Старт через {max(0, n)}с")
            self.btn_start.setStyleSheet(
                "QPushButton, QPushButton:disabled{background:#6B7280;color:white;"
                "border:1px solid #4B5563;padding:6px 10px;border-radius:6px;}"
            )
            _refresh(self.btn_start)
            return

        if state == "recording":
            self.btn_start.setEnabled(True)
            # Show elapsed time on the button.
            t = int(left or 0)
            mm = max(0, t) // 60
            ss = max(0, t) % 60
            if self._ui_lang == "en":
                self.btn_start.setText(f"REC {mm:02d}:{ss:02d}")
            else:
                self.btn_start.setText(f"Запись {mm:02d}:{ss:02d}")
            self.btn_start.setStyleSheet(
                "QPushButton, QPushButton:disabled{background:#16A34A;color:white;"
                "border:1px solid #15803D;padding:6px 10px;border-radius:6px;}"
            )
            _refresh(self.btn_start)
            return

        # idle
        self.btn_start.setEnabled(True)
        self.btn_start.setText(self._tr("Старт записи", "Start recording"))
        self.btn_start.setStyleSheet("")
        _refresh(self.btn_start)

    def _on_always_on_top(self, enabled: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(enabled))
        self.show()

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
        # Avoid Qt aborts like "QThread: Destroyed while thread is still running".
        try:
            if self._start_timer is not None:
                QMessageBox.information(
                    self,
                    self._tr("Подождите", "Please wait"),
                    self._tr("Скоро начнется запись (идет отсчет)", "Recording countdown is running"),
                )
                event.ignore()
                return
            if self._test_worker is not None and self._test_worker.isRunning():
                QMessageBox.information(
                    self,
                    self._tr("Подождите", "Please wait"),
                    self._tr("Идет тест источника", "Source test is running"),
                )
                event.ignore()
                return
            if self._live_worker is not None and self._live_worker.isRunning():
                QMessageBox.information(
                    self,
                    self._tr("Подождите", "Please wait"),
                    self._tr("Идет лайв транскрипция", "Live transcription is running"),
                )
                event.ignore()
                return
            if self._worker is not None and self._worker.isRunning():
                QMessageBox.information(
                    self,
                    self._tr("Подождите", "Please wait"),
                    self._tr("Идет транскрипция", "Transcription is running"),
                )
                event.ignore()
                return
            if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
                QMessageBox.information(
                    self,
                    self._tr("Подождите", "Please wait"),
                    self._tr("Идет транскрипция", "Transcription is running"),
                )
                event.ignore()
                return
        except Exception:
            # If closeEvent fails, prefer safe behavior.
            event.ignore()
            return

        # Persist last known selections.
        self._save_device_selection()
        super().closeEvent(event)

    def _on_stop(self) -> None:
        if not self._recorder:
            return

        try:
            self._on_stop_impl()
        except Exception:
            QMessageBox.critical(self, self._tr("Ошибка", "Error"), traceback.format_exc())
            # Make sure UI is not stuck.
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _on_stop_impl(self) -> None:
        # The original stop handler body (wrapped for safety).
        assert self._recorder is not None

        self.btn_stop.setEnabled(False)
        try:
            rec = self._recorder.stop()
        except Exception as e:
            self._recorder = None
            QMessageBox.critical(self, self._tr("Ошибка записи", "Recording error"), str(e))
            self.btn_start.setEnabled(True)
            return
        finally:
            self._recorder = None

        if self._rec_timer is not None:
            self._rec_timer.stop()

        self._set_start_button_state("idle")
        self._update_test_buttons()

        self._stop_live()

        # Verify recorded files exist and are non-empty.
        if not self._last_session:
            QMessageBox.warning(
                self,
                self._tr("Пустая запись", "Empty recording"),
                self._tr("Не найдены файлы сессии", "Session files were not found"),
            )
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
                self._tr("Пустая запись", "Empty recording"),
                self._tr(
                    "Не удалось записать звук ни с микрофона, ни с выхода. Проверьте выбранные устройства и права доступа к микрофону.",
                    "No audio was captured from microphone or output. Check selected devices and microphone permissions.",
                ),
            )
            self.btn_start.setEnabled(True)
            return

        if getattr(rec, "errors", None):
            msg = "\n".join([f"{k}: {v}" for k, v in rec.errors.items()])
            QMessageBox.information(
                self,
                self._tr("Захват аудио: предупреждение", "Audio capture: warning"),
                self._tr(
                    f"Есть ошибки в одном из потоков:\n{msg}",
                    f"There were errors in one of the streams:\n{msg}",
                ),
            )

        session_root = str(self._last_session.root) if self._last_session else ""
        self.transcript_view.setPlainText(
            self._tr(
                f"Запись остановлена. Идет распознавание...\nСессия: {session_root}\n",
                f"Recording stopped. Transcribing...\nSession: {session_root}\n",
            )
        )

        mic_label = str(self.mic_combo.currentText())
        out_label = str(self.out_combo.currentText())
        model_name = str(self.model_combo.currentText())
        language = (self.lang_edit.text() or "ru").strip()
        device = str(self.device_combo.currentText())
        compute = str(self.compute_combo.currentText())

        mic_path = rec.mic_path if has_mic else ""
        out_path = rec.loopback_path if has_out else ""

        # Transcribe in a separate process to avoid crashing the GUI
        # if a native ASR backend fails.
        assert self._last_session is not None
        self._start_transcription_process(
            session=self._last_session,
            mic_path=mic_path,
            out_path=out_path,
            samplerate=int(rec.samplerate),
            duration_s=float(rec.duration_s),
            mic_label=mic_label,
            out_label=out_label,
        )

    def _start_live_if_enabled(self) -> None:
        self._live_enabled = bool(self.act_live.isChecked())
        self._live_segments = []
        if not self._live_enabled or not self._recorder:
            return

        # Avoid spawning multiple live threads.
        if self._live_worker is not None and self._live_worker.isRunning():
            return

        self._live_worker = LiveWorker(
            parent=self,
            recorder=self._recorder,
            model_name=str(self.model_combo.currentText()),
            language=(self.lang_edit.text() or "ru").strip(),
            ui_lang=str(self._ui_lang),
            session_label=str(self._current_session_label),
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
        self._live_worker.finished.connect(self._on_live_finished)
        self._live_worker.start()

    def _stop_live(self) -> None:
        if self._live_worker is None:
            return
        # Do not drop the reference until the thread actually finishes,
        # otherwise Qt may abort with "QThread: Destroyed while thread is still running".
        self._live_worker.requestInterruption()

    def _on_live_finished(self) -> None:
        # Cleanup after the live worker exits.
        self._live_worker = None

    def _update_test_buttons(self) -> None:
        busy = self._recorder is not None or self._test_worker is not None or self._start_timer is not None

        mic_ok = bool(self.chk_mic.isChecked()) and self.mic_combo.currentIndex() >= 0
        out_ok = bool(self.chk_out.isChecked()) and self.out_combo.currentIndex() >= 0

        self.btn_test_mic.setEnabled((not busy) and mic_ok)
        self.btn_test_out.setEnabled((not busy) and out_ok)

    def _set_test_button_visual(self, btn: QPushButton, phase: str, left_s: int) -> None:
        phase = str(phase)
        left_s = int(max(0, left_s))

        def _refresh() -> None:
            try:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            except Exception:
                pass
            btn.update()

        if phase == "waiting":
            btn.setEnabled(True)
            if self._ui_lang == "en":
                btn.setText(f"Starting in {left_s}s")
            else:
                btn.setText(f"Старт через {left_s}с")
            btn.setStyleSheet(
                "QPushButton, QPushButton:disabled{background:#6B7280;color:white;"
                "border:1px solid #4B5563;padding:6px 10px;border-radius:6px;}"
            )
            _refresh()
            return

        if phase == "recording":
            btn.setEnabled(True)
            if self._ui_lang == "en":
                btn.setText(f"Recording {left_s}s")
            else:
                btn.setText(f"Запись {left_s}с")
            btn.setStyleSheet(
                "QPushButton, QPushButton:disabled{background:#16A34A;color:white;"
                "border:1px solid #15803D;padding:6px 10px;border-radius:6px;}"
            )
            _refresh()
            return

        if phase == "playback":
            btn.setEnabled(True)
            if self._ui_lang == "en":
                btn.setText(f"Playing {left_s}s")
            else:
                btn.setText(f"Воспроизв {left_s}с")
            btn.setStyleSheet(
                "QPushButton, QPushButton:disabled{background:#F97316;color:white;"
                "border:1px solid #EA580C;padding:6px 10px;border-radius:6px;}"
            )
            _refresh()
            return

        # idle
        btn.setEnabled(True)
        if btn is self.btn_test_mic:
            btn.setText(self._tr("Тест микрофона", "Test mic"))
        else:
            btn.setText(self._tr("Тест собеседника", "Test other side"))
        btn.setStyleSheet("")
        _refresh()

    def _start_test(self, kind: str) -> None:
        if self._recorder is not None:
            QMessageBox.information(
                self,
                self._tr("Тест", "Test"),
                self._tr(
                    "Остановите запись перед тестированием источников",
                    "Stop recording before testing sources",
                ),
            )
            return
        if self._test_worker is not None:
            return

        samplerate = 48000
        seconds = 2.0
        blocksize = 2048

        if kind == "mic":
            device_id = str(self.mic_combo.currentData())
            include_loopback = False
            title = self._tr("Тест микрофона", "Mic test")
            active_btn = self.btn_test_mic
        else:
            idx = self.loop_mode.currentIndex()
            device_id = str(self.out_combo.currentData())
            include_loopback = (idx == 1 or idx == 2)
            title = self._tr("Тест собеседника", "Other side test")
            active_btn = self.btn_test_out

        if self._ui_lang == "en":
            self.transcript_view.setPlainText(
                f"{title}...\n"
                f"Recording ~{seconds:.0f}s, then playback.\n"
                "If you hear gaps, try disabling live mode and lowering system load."
            )
        else:
            self.transcript_view.setPlainText(
                f"{title}...\n"
                f"Идет запись ~{seconds:.0f} сек, затем воспроизведение.\n"
                "Если слышны провалы, попробуйте выключить лайв и снизить нагрузку."
            )

        self._test_worker = AudioTestWorker(
            parent=self,
            kind=kind,
            device_id=device_id,
            include_loopback=include_loopback,
            samplerate=samplerate,
            seconds=seconds,
            blocksize=blocksize,
            start_delay_s=3,
            ui_lang=str(self._ui_lang),
        )
        self._test_ui_button = active_btn
        self._test_worker.phase.connect(self._on_test_phase)
        self._test_worker.done.connect(lambda: self._on_test_done(title))
        self._test_worker.failed.connect(lambda msg: self._on_test_failed(title, msg))
        self._update_test_buttons()
        self._test_worker.start()

    def _on_test_mic(self) -> None:
        if not self.chk_mic.isChecked() or self.mic_combo.currentIndex() < 0:
            return
        self._start_test("mic")

    def _on_test_out(self) -> None:
        if not self.chk_out.isChecked() or self.out_combo.currentIndex() < 0:
            return
        self._start_test("out")

    def _on_test_done(self, title: str) -> None:
        self._test_worker = None
        if self._test_ui_timer is not None:
            self._test_ui_timer.stop()
        self._test_ui_timer = None
        self._test_ui_left_s = 0
        self._test_ui_phase = ""
        if self._test_ui_button is not None:
            self._set_test_button_visual(self._test_ui_button, "idle", 0)
        self._test_ui_button = None
        self._update_test_buttons()
        self.transcript_view.setPlainText(
            self._tr(
                f"{title}: готово.",
                f"{title}: done.",
            )
        )

    def _on_test_failed(self, title: str, msg: str) -> None:
        self._test_worker = None
        if self._test_ui_timer is not None:
            self._test_ui_timer.stop()
        self._test_ui_timer = None
        self._test_ui_left_s = 0
        self._test_ui_phase = ""
        if self._test_ui_button is not None:
            self._set_test_button_visual(self._test_ui_button, "idle", 0)
        self._test_ui_button = None
        self._update_test_buttons()
        QMessageBox.critical(self, title, msg)

    def _on_test_phase(self, phase: str, seconds_total: int) -> None:
        if self._test_ui_button is None:
            return

        self._test_ui_phase = str(phase)
        self._test_ui_left_s = int(max(0, seconds_total))
        self._set_test_button_visual(self._test_ui_button, self._test_ui_phase, self._test_ui_left_s)

        if self._test_ui_timer is None:
            self._test_ui_timer = QTimer(self)
            self._test_ui_timer.setInterval(1000)
            self._test_ui_timer.timeout.connect(self._on_test_ui_tick)
        self._test_ui_timer.start()

    def _on_test_ui_tick(self) -> None:
        if self._test_ui_button is None or not self._test_ui_phase:
            return
        if self._test_ui_left_s <= 0:
            return
        self._test_ui_left_s -= 1
        self._set_test_button_visual(self._test_ui_button, self._test_ui_phase, self._test_ui_left_s)

    def _on_live_update(self, txt: str) -> None:
        # Live draft; ignore late updates after recording stopped.
        if self._recorder is None:
            return
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
        QMessageBox.critical(self, self._tr("Ошибка распознавания", "Transcription error"), message)
        self.btn_start.setEnabled(True)
        self.btn_export_txt.setEnabled(False)
        self.btn_export_json.setEnabled(False)

    def _on_export_txt(self) -> None:
        if not self._last_txt:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Сохранить TXT", "Save TXT"),
            "transcript.txt",
            self._tr("Text (*.txt)", "Text (*.txt)"),
        )
        if not path:
            return
        Path(path).write_text(self._last_txt, encoding="utf-8")

    def _on_export_json(self) -> None:
        if not self._last_json:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Сохранить JSON", "Save JSON"),
            "transcript.json",
            self._tr("JSON (*.json)", "JSON (*.json)"),
        )
        if not path:
            return
        Path(path).write_text(self._last_json, encoding="utf-8")

    def _create_session_paths(self, label: str) -> SessionPaths:
        root = Path(os.getcwd()) / "sessions"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = self._slugify_label(label)
        base = root / f"{ts}__{slug}"
        session_root = base
        n = 2
        while session_root.exists():
            session_root = root / f"{ts}__{slug}-{n}"
            n += 1
        return SessionPaths(
            root=session_root,
            meta_json=session_root / "meta.json",
            label=str(label).strip(),
            slug=slug,
            mic_wav=session_root / "mic.wav",
            loop_wav=session_root / "loopback.wav",
            transcript_txt=session_root / "transcript.txt",
            transcript_json=session_root / "transcript.json",
        )


class LiveWorker(QThread):
    update = Signal(str)

    def __init__(
        self,
        parent: QObject | None,
        recorder: DualRecorder,
        model_name: str,
        language: str,
        ui_lang: str,
        session_label: str,
        device: str,
        compute_type: str,
        enable_mic: bool,
        enable_out: bool,
        mic_label: str,
        out_label: str,
        interval_s: float,
        window_s: float,
    ) -> None:
        super().__init__(parent)
        self._recorder = recorder
        self._model_name = model_name
        self._language = language
        self._ui_lang = str(ui_lang).strip().lower() or "ru"
        self._session_label = str(session_label).strip()
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
                ui_lang=self._ui_lang,
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
                    session_label=self._session_label,
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
                txt = to_pretty_txt(meta, merged, ui_lang=self._ui_lang)
                if self._ui_lang == "en":
                    txt += "\n(Live draft: final version will be more accurate after stop)\n"
                else:
                    txt += "\n(Лайв черновик: финальная версия будет точнее после стопа)\n"
                self.update.emit(txt)

                # Sleep in small steps so stop is responsive.
                end = time.monotonic() + self._interval_s
                while time.monotonic() < end:
                    if self.isInterruptionRequested():
                        return
                    time.sleep(0.05)
        except Exception as e:
            if self._ui_lang == "en":
                self.update.emit(f"Live transcription stopped due to an error:\n{e}\n")
            else:
                self.update.emit(f"Лайв транскрипция остановлена из-за ошибки:\n{e}\n")
            return


class AudioTestWorker(QThread):
    done = Signal()
    failed = Signal(str)
    phase = Signal(str, int)  # ("waiting"|"recording"|"playback", seconds_total)

    def __init__(
        self,
        parent: QObject | None,
        kind: str,
        device_id: str,
        include_loopback: bool,
        samplerate: int,
        seconds: float,
        blocksize: int,
        start_delay_s: int = 3,
        ui_lang: str = "ru",
    ) -> None:
        super().__init__(parent)
        self._kind = str(kind)
        self._device_id = str(device_id)
        self._include_loopback = bool(include_loopback)
        self._samplerate = int(samplerate)
        self._seconds = float(seconds)
        self._blocksize = int(blocksize)
        self._start_delay_s = int(start_delay_s)
        self._ui_lang = str(ui_lang).strip().lower() or "ru"

    @staticmethod
    def _fade(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
        n = int(round(float(ms) * 0.001 * float(sr)))
        if n <= 1 or x.size == 0:
            return x

        y = np.asarray(x, dtype=np.float32).copy()
        n = min(n, y.shape[0] // 2)
        if n <= 0:
            return y

        w = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        y[:n, 0] *= w
        y[-n:, 0] *= w[::-1]
        return y

    def run(self) -> None:
        try:
            mic = sc.get_microphone(self._device_id, include_loopback=self._include_loopback)
            if not mic:
                raise RuntimeError("Device not found" if self._ui_lang == "en" else "Устройство не найдено")

            delay_s = max(0, int(self._start_delay_s))
            if delay_s:
                self.phase.emit("waiting", delay_s)
                end = time.monotonic() + float(delay_s)
                while time.monotonic() < end:
                    if self.isInterruptionRequested():
                        return
                    time.sleep(0.05)

            self.phase.emit("recording", int(max(1, round(self._seconds))))
            need = max(1, int(round(self._seconds * self._samplerate)))
            chunks: list[np.ndarray] = []
            got = 0

            with mic.recorder(samplerate=self._samplerate, channels=1) as rec:
                while got < need and not self.isInterruptionRequested():
                    n = min(self._blocksize, need - got)
                    x = rec.record(numframes=n)
                    fx = np.asarray(x, dtype=np.float32, order="C")
                    if fx.ndim == 1:
                        fx = fx.reshape((-1, 1))
                    elif fx.shape[1] != 1:
                        fx = fx[:, :1]
                    chunks.append(fx)
                    got += int(fx.shape[0])

            audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)
            audio = self._fade(audio, self._samplerate, ms=10.0)

            play_s = int(max(1, round(audio.shape[0] / float(self._samplerate))))
            self.phase.emit("playback", play_s)

            speaker = sc.default_speaker()
            if not speaker:
                raise RuntimeError(
                    "Default playback device not found" if self._ui_lang == "en" else "Не найдено устройство воспроизведения по умолчанию"
                )

            # Playback: always through the default speaker/headphones.
            speaker.play(audio, samplerate=self._samplerate, channels=1)

            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))
