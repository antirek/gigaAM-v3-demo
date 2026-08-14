# План: переключатель бэкенда диаризации (`sherpa` / `diart`)

Цель: выбрать движок live-диаризации через env, без смены UI/API контракта.
ASR (GigaAM) не трогаем. Диаризация по-прежнему **только streaming**.

---

## Контекст

Сейчас работает вариант **Charoite-like**:

- `sherpa-onnx` + `pyannote` segmentation ONNX + ERes2Net embedding
- свой `SegmentSpeakerTracker` (пороги, sticky, turn_gap, absorb fragments)
- модели в `./data/diar/` (~44 MB)
- env: `DIAR_*`

Второй вариант — **[diart](https://github.com/juanmc2005/diart)**:

- готовый online pipeline: segmentation + embedding + incremental clustering
- обычно модели pyannote с Hugging Face (gated, нужен token)
- нативные streaming-окна, меньше ручных эвристик
- CPU поддерживается; модели ~30–35 MB, runtime тяжелее из‑за pyannote

---

## Целевой контракт

### Env

```bash
# off | sherpa | diart
DIAR_BACKEND=sherpa

DIAR_ENABLED=true          # legacy: если false → как DIAR_BACKEND=off
DIAR_MAX_SPEAKERS=2

# --- sherpa (текущий) ---
DIAR_THRESHOLD=0.62
DIAR_MIN_NEW_SEC=0.8
DIAR_SWITCH_MARGIN=0.08
DIAR_STICKY=0.08
DIAR_TURN_GAP=0.35
DIAR_EMB_MODEL=/data/diar/embedding.onnx
DIAR_SEG_MODEL=/data/diar/segmentation.onnx

# --- diart ---
DIAR_HF_TOKEN=             # или HUGGING_FACE_HUB_TOKEN
DIAR_DIART_SEG_MODEL=pyannote/segmentation-3.0
DIAR_DIART_EMB_MODEL=pyannote/embedding
DIAR_DIART_STEP=0.5        # сдвиг окна, сек
DIAR_DIART_LATENCY=0.5     # latency pipeline, сек (уточнить по API diart)
DIAR_DIART_CACHE=/data/diart
```

### Единый интерфейс трекера

Оба бэкенда реализуют один протокол (как сейчас `label_spans`):

```python
class SpeakerTracker(Protocol):
    def label_spans(
        self, chunk: np.ndarray, update_centroids: bool = True
    ) -> list[tuple[float, float, int]]:
        """(start_s, end_s, speaker_1based) относительно chunk."""

    @property
    def voices(self) -> int: ...
```

`StreamSession` не знает, sherpa это или diart — только `create_tracker()`.

### Health

```json
"diarization": {
  "enabled": true,
  "backend": "sherpa" | "diart" | "off",
  "ready": true,
  "mode": "segments",
  "max_speakers": 2,
  "error": ""
}
```

UI: в статусе показывать `diar: sherpa` / `diar: diart`.

---

## Архитектура файлов (предложение)

```
backend/app/
  diarization/
    __init__.py          # DiarizationService + factory по DIAR_BACKEND
    base.py              # Protocol / общие типы
    sherpa_backend.py    # текущий SegmentSpeakerTracker (перенос)
    diart_backend.py     # обёртка diart → label_spans
  diarization.py         # deprecated shim → re-export (или удалить после переноса)
  streaming.py           # без изменений логики, только сервис
```

Либо проще без пакета: `diarization.py` (factory) + `diar_sherpa.py` + `diar_diart.py`.

---

## Как встроить diart в наш stream

Сейчас: WebM буфер → WAV → вырезаем окно → `label_spans(audio)` → ASR по spans.

Для diart варианты:

| Вариант | Суть | Плюсы | Минусы |
|---------|------|-------|--------|
| **A. Chunked API** | На каждый наш WAV-chunk вызываем diart pipeline step / push samples | Ближе к текущему коду | Нужно аккуратно стыковать sample stream |
| **B. Параллельный mic sink** | diart слушает свой source | «Каноничный» diart | Плохо стыкуется с WebSocket webm от браузера |
| **C. Hybrid** | diart только для меток на уже накопленном PCM ring-buffer | Проще отладка | Может быть не pure-streaming |

**Рекомендация плана:** вариант **A** — кормить diart PCM 16 kHz mono из тех же окон, что и сейчас, маппить annotation → `label_spans`.

Ограничение `max_speakers=2`: если diart API позволяет — задать; иначе пост-фильтр / слияние лишних кластеров (уточнить в Q).

---

## Этапы

### E0 — согласование
- ответы на вопросы ниже
- проверка диска перед установкой pyannote/diart

### E1 — рефакторинг sherpa
- [x] вынести текущий код в `sherpa_backend`
- [x] `DIAR_BACKEND=sherpa|off` (поведение = сегодня)
- [x] health отдаёт `backend`
- [ ] регрессия: stream + 2 спикера как в `result3`/`result4` (ручной A/B)

### E2 — зависимости diart
- [x] добавить `diart` (+ нужный `pyannote.audio` pin) в requirements / optional extra
- [x] volume `./data/diart` для HF cache
- [x] скрипт/док: accept HF licenses + `HF_TOKEN`
- [x] замер роста Docker image и RAM

### E3 — diart backend
- [x] `DiartSpeakerTracker` с `label_spans`
- [x] session-scoped pipeline (центроиды не шарятся между сокетами)
- [x] `DIAR_BACKEND=diart`
- [x] fallback: если token/моделей нет → `ready=false`, stream без меток (как сейчас без onnx)

### E4 — UX / сравнение
- [x] статус в UI: backend name
- [ ] (опционально) короткий smoke: один диалог, два прогона sherpa vs diart, лог в `result_diart.log`
- [x] обновить `README.md` + `PLAN_DIARIZATION.md`

### E5 — (опционально позже)
- UI-переключатель backend без пересборки (сейчас только env → restart)
- offline re-pass

---

## Диск и RAM (оценка)

| | sherpa (сейчас) | diart (добавка) |
|--|--|--|
| Модели | ~44 MB | ~30–35 MB |
| Image | уже есть torch + sherpa | + pyannote/diart (сотни MB+) |
| RAM diar | умеренно | ориентир 0.3–1 GB сверху |
| HF token | нет | обычно да |

Перед E2: `./scripts/check-disk.sh`, при необходимости `docker image prune`.

---

## Риски

| Риск | Митигация |
|------|-----------|
| Gated pyannote без token | `ready=false` + понятный `error` в `/health` |
| Раздувание image | optional deps / отдельный Dockerfile stage later |
| diart API ≠ наши окна | spike E3 на 1–2 часа до полной интеграции |
| Хуже на коротких репликах | A/B на тех же сценариях, что `result*.log` |
| Два спикера | проверить параметр max_speakers / clustering |

---

## Вопросы

1. **Дефолт после внедрения?** `sherpa` (безопасный) или сразу `diart`?
2. **HF token:** есть ли у тебя token с accepted conditions на `pyannote/segmentation-3.0` и `pyannote/embedding`? Если нет — ок ли завести?
3. **Модели diart:** дефолт pyannote embedding или ONNX wespeaker (чуть предсказуемее по latency)?
4. **Установка зависимостей:**
   - (a) всегда в том же image (проще switch);
   - (b) optional: diart только если `DIAR_BACKEND=diart` / extra requirements (легче image для sherpa-only).
5. **Переключение:** только через env + `docker-compose up -d` (MVP) или ещё toggle в UI (сложнее, нужен reload моделей)?
6. **max_speakers=2 для diart:** жёстко резать до 2 или дать diart самому решать число голосов?
7. **Диск:** ок скачать +~35 MB моделей и риск +сотни MB image при текущем свободном месте?
8. **Критерий успеха spike:** субъективно лучше на том же диалоге «магазин/выходные», или нужен численный DER?
9. **Если diart не встаёт в Docker с первого раза:** оставляем stub + plan follow-up или блокируем мерж до зелёного A/B?

---

## Дефолты (если не уточнено)

| Параметр | Default |
|----------|---------|
| `DIAR_BACKEND` | `sherpa` |
| Зависимости | (a) в одном image |
| Переключение | только env |
| max_speakers | 2 |
| diart models | pyannote seg 3.0 + pyannote embedding |
| UI | показать backend в status |
| Успех | субъективный A/B на 1–2 диалогах |

---

## Следующий шаг

Реализовано: `DIAR_BACKEND=sherpa|diart|off`, health/UI показывают backend, HF token через `.env`.

Осталось по желанию: ручной A/B на том же диалоге (`result_diart.log`) и offline re-pass.
