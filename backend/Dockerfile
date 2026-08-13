FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    "gigaam @ git+https://github.com/salute-developers/GigaAM.git" \
    -r requirements.txt

COPY backend/app /app/app
COPY web /app/web

ENV PYTHONUNBUFFERED=1 \
    GIGAAM_CACHE=/data/gigaam \
    DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
