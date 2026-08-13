import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum

import gigaam

from app.audio_utils import cleanup_paths, get_duration_seconds, split_wav_fixed

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelStatus:
    state: ModelState
    message: str = ""
    load_time_s: float | None = None


class GigaAMService:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._status = ModelStatus(state=ModelState.LOADING, message="Model not loaded yet")
        self.model_name = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
        self.device = os.getenv("DEVICE", "cpu")
        self.cache_dir = os.getenv("GIGAAM_CACHE", "/data/gigaam")
        self.max_audio_seconds = float(os.getenv("MAX_AUDIO_SECONDS", "60"))

    @property
    def status(self) -> ModelStatus:
        return self._status

    def load(self) -> None:
        start = time.perf_counter()
        self._status = ModelStatus(state=ModelState.LOADING, message="Downloading/loading model...")
        logger.info(
            "Loading model %s on %s (cache=%s)",
            self.model_name,
            self.device,
            self.cache_dir,
        )
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            model = gigaam.load_model(
                self.model_name,
                device=self.device,
                fp16_encoder=False,
                use_flash=False,
                download_root=self.cache_dir,
            )
            self._model = model
            elapsed = time.perf_counter() - start
            self._status = ModelStatus(
                state=ModelState.READY,
                message="Model ready",
                load_time_s=elapsed,
            )
            logger.info("Model loaded in %.1fs", elapsed)
        except Exception as exc:
            self._status = ModelStatus(state=ModelState.ERROR, message=str(exc))
            logger.exception("Failed to load model")
            raise

    def transcribe_wav(self, wav_path: str, enforce_max: bool = True) -> dict:
        if self._status.state != ModelState.READY or self._model is None:
            raise RuntimeError(self._status.message or "Model is not ready")

        duration = get_duration_seconds(wav_path)
        if enforce_max and duration > self.max_audio_seconds:
            raise ValueError(
                f"Audio too long: {duration:.1f}s (max {self.max_audio_seconds:.0f}s)"
            )

        with self._lock:
            start = time.perf_counter()
            chunks = split_wav_fixed(wav_path)
            texts: list[str] = []
            for chunk in chunks:
                result = self._model.transcribe(chunk)
                text = result.text if hasattr(result, "text") else str(result)
                if text:
                    texts.append(text.strip())
            elapsed = time.perf_counter() - start

        result_text = " ".join(texts).strip()
        cleanup_paths([c for c in chunks if c != wav_path])

        return {
            "text": result_text,
            "duration_s": duration,
            "inference_s": elapsed,
            "chunks": len(chunks),
            "streaming_native": False,
        }


gigaam_service = GigaAMService()
