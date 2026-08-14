# GigaAM Voice Recognition

Локальный стенд: Docker Compose + FastAPI + веб-UI для записи голоса и распознавания через **GigaAM-v3** (`v3_e2e_rnnt`) на CPU.

## Диск и место

**Перед первым запуском проверьте место:**

```bash
chmod +x scripts/check-disk.sh
./scripts/check-disk.sh
```

| Что | Размер (ориентир) |
|-----|-------------------|
| Веса модели (checkpoint + tokenizer) | ~450 MB |
| Docker image (PyTorch CPU + gigaam + diart) | ~2–3 GB |
| RAM при inference | 4–6 GB (+0.3–1 GB для diart) |

Кэш модели хранится **локально в проекте**: `./data/gigaam` (CDN Sber, не `~/.cache`).

Если на диске мало места:

```bash
docker image prune -a   # удалить неиспользуемые images (часто десятки GB)
```

## Быстрый старт

```bash
cp .env.example .env
# для DIAR_BACKEND=diart добавьте DIAR_HF_TOKEN=... (gated pyannote)
docker-compose up --build
```

Откройте http://localhost:8000

Первый запуск: сборка image + скачивание checkpoint (~450 MB). Healthcheck ждёт до 5 минут.

## API

- `GET /health` — статус модели и `diarization.backend`
- `POST /transcribe` — multipart file upload (webm/wav и др.)
- `WebSocket /ws/stream` — pseudo-streaming (re-transcription буфера)

## Диаризация (только streaming)

Переключатель в `.env`: `DIAR_BACKEND=sherpa|diart|off` (дефолт **sherpa**).

| Backend | Суть | Модели |
|---------|------|--------|
| `sherpa` | Charoite-like: pyannote-seg ONNX + ERes2Net | `./data/diar/` (~44 MB) |
| `diart` | online pipeline [diart](https://github.com/juanmc2005/diart) + pyannote | HF cache `./data/diart/`, нужен `DIAR_HF_TOKEN` |
| `off` | только ASR | — |

В UI статус показывает `diar: sherpa` / `diar: diart`. Смена бэкенда — правка env + `docker-compose up -d`.

Макс. **2 спикера**. Batch `/transcribe` без диаризации.

## Конфигурация

См. `.env.example`: `MODEL_NAME`, `DEVICE`, `GIGAAM_CACHE`, `DIAR_BACKEND`, `MAX_AUDIO_SECONDS`.
