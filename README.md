# Neoscry 🎙️📝

[![GitHub stars](https://img.shields.io/github/stars/tensX/Neoscry?logo=github)](https://github.com/tensX/Neoscry/stargazers)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-0f172a)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-PySide6-41CD52)
![ASR](https://img.shields.io/badge/ASR-Whisper-f97316)

Dual-source call/meeting recorder + AI transcription (speech-to-text).

- You: microphone input
- Other side: system output via WASAPI loopback (Windows)

Exports:
- `sessions/<timestamp>/transcript.txt` (readable)
- `sessions/<timestamp>/transcript.json` (structured + timestamps)

---

## Contents

- [English](#english)
- [Keywords (EN)](#keywords-en)
- [Русский](#русский)
- [Ключевые слова (RU)](#ключевые-слова-ru)

---

## English

### What it does

Neoscry records a conversation from two sources (mic + system audio via WASAPI loopback) and produces a final, accurate transcript using Whisper (via `faster-whisper`).

Typical use cases: call recording, meeting notes, interviews, podcasts, voice memos, “system audio to text”.

### Features

- Two streams: mic (you) + output/loopback (other side)
- 3 “other side” modes: any input device, loopback-input device, or loopback derived from a selected speaker
- Live transcription (draft) while recording + final transcription after stopping
- Export to TXT and JSON
- GPU acceleration options: `auto/cuda/cpu` and compute type (`float16`, `int8`, ...)
- Built-in source testing (record a few seconds, then play back)
- Settings menu (gear): UI language (RU/EN), always-on-top, live draft
- Re-transcribe any existing session (pick a `sessions/<timestamp>/` folder)

### Quick start

Requirements:
- Python 3.11+

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python run.py
```

### Platform notes

- Windows: loopback capture uses WASAPI.
- Linux: loopback depends on your audio stack (PulseAudio / PipeWire). You may need to select a monitor source ("*.monitor") as the “other side” input device.

### GPU (CUDA)

If you select `Device = cuda`, make sure CUDA runtime libraries are installed and available in `PATH` (for example `cublas64_12.dll` on Windows). The app includes `Settings -> Check GPU (CUDA)` and an optional `Install CUDA Toolkit (winget)` helper.

### How to use

1) Pick sources:
- Enable `Record microphone` and select your microphone
- Enable `Record output` and select the “other side” source

2) Pick ASR settings:
- Model: best quality is usually `large-v3`
- Device: `cuda` (if available) or `cpu`
- Compute: for CUDA, `float16` is a common choice

3) Optional: enable `Live transcription (draft)`.

4) Click `Start recording`, then `Stop + transcribe` for the final transcript.

### Where results are saved

Each session creates `sessions/<YYYYMMDD-HHMMSS>__<label>/`:

- `mic.wav` - microphone recording (if enabled)
- `loopback.wav` - other-side recording (if enabled)
- `transcript.txt` - formatted transcript
- `transcript.json` - segments with timestamps and speaker labels

### Notes / troubleshooting

- Loopback capture depends on Windows audio drivers and virtual devices (Voicemeeter, etc.).
- If you previously saw `data discontinuity in recording` warnings from `soundcard`, the app now hides them to keep the console clean.

### Settings storage

- Windows: registry `HKEY_CURRENT_USER\Software\Neoscry\Neoscry`
- Linux: via Qt settings (typically under `~/.config/Neoscry/`)

## Keywords (EN)

speech to text, speech-to-text, audio transcription, AI transcription, Whisper, faster-whisper, call recorder, meeting recorder, meeting notes, interview recorder, system audio recording, WASAPI loopback, Windows loopback recording, microphone + desktop audio, dual-source recording, PySide6 GUI, Python

---

## Русский

### Что это

Neoscry записывает разговор из двух источников (микрофон + системный звук через WASAPI loopback) и делает финальную точную транскрипцию на базе Whisper (через `faster-whisper`).

Примеры: запись звонков, заметки по встречам, интервью, подкасты, диктовка, “системный звук в текст”.

### Возможности

- Два потока: микрофон (вы) + собеседник (выход / loopback)
- 3 режима источника собеседника: любое input-устройство, loopback-input, или loopback от выбранного устройства вывода
- Лайв транскрипция (черновик) во время записи + финальная транскрипция после остановки
- Экспорт: TXT и JSON
- Ускорение на GPU: `auto/cuda/cpu` и `compute` (`float16`, `int8`, ...)
- Тест источников (пишет пару секунд, затем воспроизводит)
- Меню настроек (⚙): язык интерфейса (RU/EN), поверх всех окон, лайв-черновик
- Повторная транскрипция любой сохраненной сессии (выбор папки `sessions/<timestamp>/`)

### Быстрый старт

Требования:
- Python 3.11+

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Запуск:

```bash
python run.py
```

### Как пользоваться

1) Выберите источники:
- Включите `Записывать микрофон` и выберите микрофон
- Включите `Записывать выход` и выберите источник собеседника

2) Выберите настройки распознавания:
- `Модель`: для максимальной точности чаще всего `large-v3`
- `Устройство`: `cuda` (если доступно) или `cpu`
- `Compute`: для CUDA обычно выбирают `float16`

3) (Опционально) включите `Лайв транскрипция` — это черновик во время записи.

4) Нажмите `Старт записи`, затем `Стоп + транскрибировать` для финального результата.

### Где лежат результаты

После каждой сессии создается папка `sessions/<YYYYMMDD-HHMMSS>__<метка>/`:

- `mic.wav` — запись микрофона (если включено)
- `loopback.wav` — запись собеседника (если включено)
- `transcript.txt` — отформатированный транскрипт
- `transcript.json` — сегменты (таймкоды, спикер, текст)

### Примечания / устранение проблем

- На Windows loopback/виртуальные устройства зависят от драйверов (Voicemeeter и т.п.).
- Раньше могли появляться warnings `data discontinuity in recording` (soundcard). Теперь приложение их скрывает, чтобы не засорять консоль.

### Где хранятся настройки

- Windows: реестр `HKEY_CURRENT_USER\Software\Neoscry\Neoscry`
- Linux: настройки Qt (обычно `~/.config/Neoscry/`)

## Ключевые слова (RU)

распознавание речи, речь в текст, транскрибация, транскрипция, AI транскрипция, Whisper, faster-whisper, запись звонков, запись встреч, заметки встреч, запись интервью, запись системного звука, WASAPI loopback, Windows loopback, запись микрофона и системного звука, два источника, PySide6, Python
