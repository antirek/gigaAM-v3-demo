# План проекта: GigaAM Voice Recognition

## Цель

Собрать локальный стенд для распознавания речи на модели **GigaAM** (Sber, v3): запуск в Docker Compose, веб-страница для записи голоса и отправки в модель, проверка batch- и потокового (streaming) режимов на CPU.

## Контекст по модели

| Параметр | Детали |
|----------|--------|
| Модель | `ai-sage/GigaAM-v3` на Hugging Face |
| Репозиторий | [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) |
| Размер | ~220–240M параметров |
| Язык | Русский (v3); есть multilingual-линейка |
| Варианты | `ssl`, `ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt` |
| Рекомендуемый стек | `torch==2.8.0`, `torchaudio==2.8.0`, `transformers==4.57.1` |
| CPU | Модель достаточно лёгкая для CPU; inference будет медленнее, чем на GPU, но реализуемо |
| Streaming | В архитектуре предусмотрен chunkwise attention и streaming fine-tuning; нужно проверить, что доступно в публичном inference-коде и HF-ревизиях |

**Рекомендуемый вариант для демо:** `e2e_rnnt` или `e2e_ctc` — end-to-end с пунктуацией и нормализацией текста.

---

## Архитектура (целевая)

```
┌─────────────────┐     HTTP/WebSocket     ┌──────────────────┐
│  Web UI         │ ◄────────────────────► │  API (FastAPI)   │
│  (record audio) │                        │  + GigaAM model  │
└─────────────────┘                        └──────────────────┘
         │                                          │
         └──────────── docker-compose ──────────────┘
```

### Сервисы

1. **`model-api`** — Python (FastAPI):
   - загрузка модели при старте;
   - `POST /transcribe` — распознавание целого аудиофайла (batch);
   - `WebSocket /stream` или chunked HTTP — потоковое распознавание (если поддерживается);
   - health-check `GET /health`.

2. **`web`** — статическая страница или лёгкий фронтенд:
   - записать голос через `MediaRecorder` (браузер);
   - отправить WAV/WebM на API;
   - отобразить текст;
   - (опционально) режим «говорить в реальном времени» через WebSocket.

3. **Общие требования Docker:**
   - volume для кэша моделей Hugging Face (`~/.cache/huggingface`);
   - ограничение RAM (ориентир: 4–8 GB для CPU inference);
   - `platform: linux/amd64` при необходимости.

---

## Этапы работ

### Этап 0 — Подготовка и разведка

- [ ] Клонировать/изучить официальный репозиторий GigaAM и примеры inference
- [ ] Проверить загрузку `ai-sage/GigaAM-v3` (revision `e2e_rnnt`) локально на CPU
- [ ] Зафиксировать время inference на коротком (5–10 с) и длинном (30–60 с) аудио
- [ ] Найти в коде/доках API для streaming inference (chunk size, causal convolutions)
- [ ] Подготовить 2–3 тестовые аудиофайлы (русская речь, разная длина)

**Результат:** короткий отчёт «batch работает / streaming доступен или нет» + выбранный вариант модели.

### Этап 1 — Backend API

- [ ] Создать `backend/` с FastAPI
- [ ] Dockerfile для API (CPU-only PyTorch)
- [ ] Endpoint batch transcription (`multipart/form-data` или raw audio bytes)
- [ ] Валидация формата: WAV 16 kHz mono (конвертация через `torchaudio` / `ffmpeg` при необходимости)
- [ ] Логирование времени inference и ошибок
- [ ] Переменные окружения: `MODEL_ID`, `MODEL_REVISION`, `DEVICE=cpu`, `HF_TOKEN` (если нужен)

### Этап 2 — Docker Compose

- [ ] `docker-compose.yml` с сервисом `model-api`
- [ ] Volume для HF cache (чтобы не скачивать модель при каждом перезапуске)
- [ ] Healthcheck и порты (например, API `:8000`)
- [ ] Документация запуска: `docker compose up --build`

### Этап 3 — Web UI

- [ ] Простая HTML/JS страница (или Vite + vanilla/minimal React)
- [ ] Кнопки: Record / Stop / Send
- [ ] Превью записанного аудио
- [ ] Отображение результата распознавания
- [ ] Обработка ошибок (модель не готова, таймаут, пустой аудио)
- [ ] Сервис `web` в docker-compose (nginx или встроенный static в FastAPI)

### Этап 4 — Streaming (исследование + прототип)

- [ ] Проверить, есть ли в `trust_remote_code` модели метод/streaming API
- [ ] Если есть — реализовать WebSocket: клиент шлёт чанки аудио, сервер возвращает partial transcripts
- [ ] Если нет в HF-обёртке — попробовать inference из `salute-developers/GigaAM` с chunkwise causal mode
- [ ] Сравнить качество и latency: batch vs streaming (chunk 1s / 2s / 4s)
- [ ] UI: toggle «Потоковый режим» + отображение промежуточных результатов

**Критерий успеха streaming:** текст появляется по мере речи с задержкой &lt; 2–3 с на CPU (ориентир, зависит от железа).

### Этап 5 — Полировка

- [ ] README с инструкцией запуска
- [ ] `.env.example`
- [ ] Базовые smoke-тесты API
- [ ] (опционально) ограничение размера файла, rate limit

---

## Структура репозитория (план)

```
gigaam-recognition/
├── PLAN.md                 # этот файл
├── README.md
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py        # загрузка GigaAM, transcribe
│   │   └── streaming.py    # streaming logic (если доступно)
│   └── tests/
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── samples/                # тестовые аудио (опционально)
```

---

## Риски и ограничения

| Риск | Митигация |
|------|-----------|
| Медленный CPU inference | Ограничить длину аудио; квантизация; позже — GPU |
| Streaming не в HF-обёртке | Использовать код из официального репо; или симулировать chunked batch |
| Больший размер Docker image | Multi-stage build; CPU-only torch |
| Первый запуск долгий (download модели) | HF cache volume; документировать размер (~1 GB+) |
| Браузер шлёт WebM, модель ждёт WAV | Конвертация на backend |

---

## Вопросы (нужны ответы перед/во время реализации)

### Модель и окружение

1. **Какой вариант модели предпочтительнее?**
   - `e2e_rnnt` (пунктуация + нормализация, RNN-T) — рекомендуемый default
   - `e2e_ctc` (пунктуация + нормализация, CTC)
   - `ctc` / `rnnt` (без e2e нормализации)
   - Нужна multilingual-линейка или только русский v3?

2. **Есть ли GPU на машине, где будет запуск?**
   - Сейчас планируем CPU-only; если GPU доступен — можно добавить `DEVICE=cuda` и ускорить inference.

3. **Ограничения по RAM и диску?**
   - Ориентир: 4–8 GB RAM, ~2–3 GB на диск (модель + зависимости + Docker image).

4. **Нужен ли Hugging Face token (`HF_TOKEN`)?**
   - Для публичной модели `ai-sage/GigaAM-v3` обычно не нужен; уточнить при первой загрузке.

### Функциональные требования

5. **Максимальная длина аудио для batch-режима?**
   - Например: 30 с, 60 с, 5 минут?

6. **Что важнее для streaming: скорость отклика или качество текста?**
   - Меньший chunk → быстрее partial results, но выше WER (по paper GigaAM).

7. **Нужна ли поддержка загрузки файла (не только записи с микрофона)?**

8. **Нужна ли история распознаваний / сохранение на сервере?**
   - Для демо можно без персистентности.

### UI и развёртывание

9. **Достаточно простой одностраничной демо или нужен более «продуктовый» UI?**

10. **Целевой URL/порт и нужен ли HTTPS локально?**

11. **Один пользователь (локальная демо) или несколько параллельных запросов?**
    - На CPU параллельность сильно ограничена.

### Приоритеты

12. **Что сделать в первой итерации (MVP)?**
    - Предлагаемый MVP: Docker Compose + batch API + страница с записью голоса.
    - Streaming — этап 2 после проверки доступности в коде модели.

---

## Следующий шаг

После ответов на вопросы выше (или с дефолтами: `e2e_rnnt`, CPU, MVP batch + UI) — начать **Этап 0** (локальная проверка модели) и параллельно скелет **Этап 1–2** (backend + docker-compose).

---

## Дефолты (если не уточнено)

| Параметр | Default |
|----------|---------|
| Модель | `ai-sage/GigaAM-v3`, revision `e2e_rnnt` |
| Device | `cpu` |
| Max audio length | 60 секунд |
| MVP scope | batch transcription + web record UI |
| Streaming | investigate в Этапе 0, прототип в Этапе 4 |
