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
| Docker image (PyTorch CPU + gigaam) | ~1.5–2 GB |
| RAM при inference | 4–6 GB |

Кэш модели хранится **локально в проекте**: `./data/gigaam` (CDN Sber, не `~/.cache`).

Если на диске мало места:

```bash
docker image prune -a   # удалить неиспользуемые images (часто десятки GB)
```

## Быстрый старт

```bash
cp .env.example .env
docker-compose up --build
```

Откройте http://localhost:8000

Первый запуск: сборка image + скачивание checkpoint (~450 MB). Healthcheck ждёт до 5 минут.

## API

- `GET /health` — статус модели
- `POST /transcribe` — multipart file upload (webm/wav и др.)
- `WebSocket /ws/stream` — pseudo-streaming (re-transcription буфера)

## Ограничения

- `transcribe`: до **25 с** на чанк; длинное аудио режется на 25-секундные фрагменты.
- Макс. длина: **60 с** (`MAX_AUDIO_SECONDS`).
- **Нативный streaming** в публичном inference API недоступен; UI — pseudo-streaming.

## Конфигурация

См. `.env.example`: `MODEL_NAME`, `DEVICE`, `GIGAAM_CACHE`, `MAX_AUDIO_SECONDS`.
