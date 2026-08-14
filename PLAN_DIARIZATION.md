# План: realtime-диаризация в стриминге

> **Статус:** live-переключатель `DIAR_BACKEND=sherpa|diart|off` — см. [`PLAN_DIART.md`](PLAN_DIART.md).

Источники:
- [Charoite_audio](https://github.com/charoiteai/Charoite_audio)
- [docs/ru/DIARIZATION.md](https://github.com/charoiteai/Charoite_audio/blob/main/docs/ru/DIARIZATION.md)
- [docs/ARCHITECTURE.md](https://github.com/charoiteai/Charoite_audio/blob/main/docs/ARCHITECTURE.md)
- скрипт `scripts/get_models.py` (модели ERes2Net / pyannote)

---

## Что делает Charoite (кратко)

Два прохода:

| Проход | Когда | Как | Зачем |
|--------|-------|-----|-------|
| **Live** | во время встречи | эмбеддинг голоса (ERes2Net ONNX, 512-dim) → трекер «Собеседник 1/2/…» | быстрый черновик |
| **Offline** | после Стоп | sherpa-onnx + сегментация pyannote, чистка эха, LLM для имён | финальная стенограмма |

Живой режим **не** угадывает имена — только номера голосов. Эмбеддинги только в RAM.

### Ключевой урок из их бенча (DER)

| Движок | DER | Голосов |
|--------|-----|---------|
| сегментация + эмбеддинг по кускам речи | **0.246** | 4/4 |
| трекер по таймерным чанкам 3 с (`live-legacy`) | 0.725 | 1/4 |
| offline sherpa | 0.296 | 3/4 |

**Вывод:** резать аудио по таймеру (как наши 20 с сегменты ASR) для эмбеддинга — плохо: в один кусок попадают два голоса → смешанный вектор → все схлопываются в одного спикера.

Нужно: **сначала границы речи (VAD/сегментация), потом эмбеддинг по куску одного голоса**, потом уже вешать метку на ASR-текст (в чанке — голос, который звучал дольше).

### Модели (размеры)

| Модель | Размер | Роль |
|--------|--------|------|
| `eres2net-base` (дефолт) | ~40 MB | live embeddings |
| `eres2net-en` | ~27 MB | легче, EN-корпус |
| `eres2netv2` | ~71 MB | точнее, тяжелее |
| pyannote seg 3.0 (sherpa) | ~6–7 MB | границы/оверлапы (offline / «правильный» live) |
| pyannote int8 | ~3 MB | то же, меньше |

Upstream: [3D-Speaker](https://github.com/modelscope/3D-Speaker) (Apache-2.0), зеркала ONNX под sherpa-onnx.

---

## Как это ложится на наш стенд

Сейчас у нас:
- GigaAM `v3_e2e_rnnt` — ASR
- WebSocket pseudo-streaming — скользящие сегменты ~20 с
- один канал (микрофон браузера)
- UI: сплошной текст без меток спикеров

Целевое поведение в **потоковом** режиме:
```
Спикер 1: Раз, два, три…
Спикер 2: Интересно в мире этом…
Спикер 1: Давай проверим…
```

### Предлагаемый MVP (live only)

```
WebM chunks → WAV
     │
     ├─ ASR (GigaAM, как сейчас, сегменты ~20 с)
     │
     └─ Diar pipeline на том же WAV:
           1) VAD / энергия / (опц.) pyannote-seg → speech spans
           2) ERes2Net ONNX → embedding 512-d на каждый span
           3) SpeakerTracker (cosine ≥ threshold → тот же спикер,
              иначе новый «Спикер N»)
           4) Для ASR-сегмента выбрать majority speaker по времени
     │
     └─ WebSocket partial/final: { utterances: [{speaker, text, t0, t1}, …] }
```

**UI:** в stream-режиме рендер списка реплик с цветными метками `Спикер N`, а не один `pre`.

**Без модели** (`embedding.onnx` нет): стрим работает как сейчас, без меток; в `/health` — `diarization: disabled`.

### Что сознательно НЕ делаем в MVP

- Offline re-pass после Стоп (sherpa full diarization)
- LLM для имён («это Милена»)
- Два канала mic + system audio (у Charoite есть; у нас один mic)
- Сохранение voice prints на диск

---

## Диск (важно)

Сейчас на машине **~10 GB свободно (98%)**.  
Live-диаризация сама по себе лёгкая: **+30–50 MB** модели + onnxruntime уже есть в image (зависимость gigaam).

Риски:
- не тянуть SenseVoice / большие STT (~228 MB) — нам не нужны;
- не раздувать Docker image лишним;
- кэш модели класть в `./data/diar/` (как `./data/gigaam/`).

Перед скачиванием — снова `./scripts/check-disk.sh`.

---

## Этапы реализации (после согласования)

### Этап D0 — согласование
- ответы на вопросы ниже
- выбор модели эмбеддинга и порога

### Этап D1 — backend diar
- [ ] `scripts/get_diar_model.sh` — скачать ERes2Net → `data/diar/embedding.onnx`
- [ ] `backend/app/diarization.py` — ONNX embedder + SpeakerTracker
- [ ] лёгкий VAD (энергия / silero / pyannote-seg — см. вопрос 3)
- [ ] интеграция в `StreamSession`: на commit сегмента — majority speaker
- [ ] расширить JSON: `utterances[]`, `speakers_count`
- [ ] `/health`: `diarization: ready|disabled`

### Этап D2 — UI
- [ ] toggle «Различать голоса» (только stream)
- [ ] список реплик с метками и цветами
- [ ] fallback: если diar off — старый сплошной текст

### Этап D3 — (опционально) offline re-pass
- [ ] pyannote-seg + sherpa clustering после `finalize`
- [ ] переразметка всей записи и замена live-черновика

---

## Оценка трудозатрат / качества

| Подход | Сложность | Ожидаемое качество | Диск |
|--------|-----------|--------------------|------|
| A. Эмбеддинг на наших 20 с ASR-чанках (без VAD) | низкая | плохо (как Charoite legacy DER~0.7) | ~40 MB |
| **B. VAD/spans + ERes2Net + tracker (MVP)** | средняя | приемлемо для демо 2–4 голоса | ~40–50 MB |
| C. Полный sherpa offline (+ live draft) | высокая | лучше финал | ~50 MB + deps |

**Рекомендация:** вариант **B**.

---

## Решения (2026-08-13)

| Вопрос | Решение |
|--------|---------|
| Скоп | Только streaming (batch без diar) |
| Спикеры | max 2 |
| «VAD» как у Charoite | **pyannote segmentation ONNX** через sherpa-onnx (не Silero) |
| Embedding | eres2net-base (~38 MB) |
| Threshold | 0.62 (как Charoite SegmentTracker) |
| Модели | `data/diar/embedding.onnx`, `data/diar/segmentation.onnx` |

---

## Вопросы к обсуждению

1. **Скоуп MVP:** только live-метки в стриме (B) или сразу ещё offline после Стоп (C)?
2. **Сколько спикеров ожидаем?** 2 (диалог), до 4, или «неизвестно»?
3. **Сегментация речи для эмбеддингов:**
   - (a) простой energy/VAD — минимум зависимостей;
   - (b) silero-vad (лёгкий);
   - (c) pyannote-seg ONNX (~6 MB) — ближе к Charoite «правильному» live.
4. **Какая embedding-модель?**
   - `eres2net-base` (~40 MB) — дефолт Charoite;
   - `eres2net-en` (~27 MB) — если мало места;
   - `eres2netv2` (~71 MB) — точнее, тяжелее.
5. **Порог cosine** (`live_diarize_threshold`, у них 0.45): оставляем 0.45 или делаем слайдер в UI?
6. **UI:** toggle «Различать голоса» по умолчанию ON или OFF?
7. **Диск:** ок скачать ~40 MB в `./data/diar/` при ~10 GB free? Нужна ли предварительная чистка Docker (`docker image prune`)?
8. **Batch-режим:** диаризация там тоже нужна или только streaming?

---

## Дефолты (если не уточнено)

| Параметр | Default |
|----------|---------|
| Скоп | Live MVP (B), без offline |
| Модель | `eres2net-base` → `data/diar/embedding.onnx` |
| Сегментация | silero-vad или energy (уточнить в Q3) |
| Threshold | 0.45 |
| UI toggle | ON в stream, если модель на диске |
| Batch | без диаризации в первой итерации |

---

## Следующий шаг

Ответь на вопросы 1–8 (можно коротко: «дефолты ок») → реализуем D1+D2 и пересоберём Docker.
