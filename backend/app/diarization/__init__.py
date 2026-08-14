"""Diarization backends: sherpa (default) and diart."""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def load_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class SpeakerTracker(Protocol):
    def label_spans(
        self, chunk: np.ndarray, update_centroids: bool = True
    ) -> list[tuple[float, float, int]]:
        """Return (start_s, end_s, speaker_1based) relative to chunk."""

    @property
    def voices(self) -> int: ...


class DiarizationService:
    def __init__(self) -> None:
        raw = os.getenv("DIAR_BACKEND", "").strip().lower()
        enabled = os.getenv("DIAR_ENABLED", "true").lower() in {"1", "true", "yes"}
        if raw in {"off", "none", "0", "false"}:
            self.backend = "off"
        elif raw in {"sherpa", "diart"}:
            self.backend = raw
        elif not enabled:
            self.backend = "off"
        else:
            self.backend = "sherpa"

        self.max_speakers = int(os.getenv("DIAR_MAX_SPEAKERS", "2"))
        self.emb_path = Path(os.getenv("DIAR_EMB_MODEL", "/data/diar/embedding.onnx"))
        self.seg_path = Path(os.getenv("DIAR_SEG_MODEL", "/data/diar/segmentation.onnx"))
        self.threshold = float(os.getenv("DIAR_THRESHOLD", "0.62"))
        self.min_new = float(os.getenv("DIAR_MIN_NEW_SEC", "0.8"))
        self.switch_margin = float(os.getenv("DIAR_SWITCH_MARGIN", "0.08"))
        self.sticky = float(os.getenv("DIAR_STICKY", "0.08"))
        self.turn_gap = float(os.getenv("DIAR_TURN_GAP", "0.35"))

        self.hf_token = (
            os.getenv("DIAR_HF_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or None
        )
        self.diart_cache = Path(os.getenv("DIAR_DIART_CACHE", "/data/diart"))
        self.diart_seg = os.getenv("DIAR_DIART_SEG_MODEL", "pyannote/segmentation-3.0")
        self.diart_emb = os.getenv("DIAR_DIART_EMB_MODEL", "pyannote/embedding")
        self.diart_duration = float(os.getenv("DIAR_DIART_DURATION", "5"))
        self.diart_step = float(os.getenv("DIAR_DIART_STEP", "0.5"))
        # Open Speakers 2 on one mic, then stabilize labels with majority/sticky.
        self.diart_latency = float(os.getenv("DIAR_DIART_LATENCY", "2.0"))
        self.diart_tau = float(os.getenv("DIAR_DIART_TAU_ACTIVE", "0.45"))
        self.diart_rho = float(os.getenv("DIAR_DIART_RHO_UPDATE", "0.15"))
        self.diart_delta = float(os.getenv("DIAR_DIART_DELTA_NEW", "0.5"))
        self.diart_min_turn = float(os.getenv("DIAR_DIART_MIN_TURN", "0.7"))
        self.diart_turn_gap = float(os.getenv("DIAR_DIART_TURN_GAP", "0.25"))
        self.diart_max_frag = float(os.getenv("DIAR_DIART_MAX_FRAG", "0.55"))

        self._ready = False
        self._error = ""
        self._mode = "disabled"

    @property
    def status(self) -> dict:
        return {
            "enabled": self.backend != "off",
            "backend": self.backend,
            "ready": self._ready,
            "mode": self._mode,
            "max_speakers": self.max_speakers,
            "error": self._error,
        }

    def load(self) -> None:
        if self.backend == "off":
            self._ready = False
            self._mode = "disabled"
            self._error = ""
            logger.info("Diarization off")
            return

        if self.backend == "sherpa":
            self._load_sherpa()
            return

        if self.backend == "diart":
            self._load_diart()
            return

        self._ready = False
        self._mode = "error"
        self._error = f"unknown DIAR_BACKEND={self.backend}"

    def _load_sherpa(self) -> None:
        if not self.emb_path.exists() or not self.seg_path.exists():
            self._ready = False
            self._mode = "disabled"
            self._error = f"missing sherpa models ({self.seg_path}, {self.emb_path})"
            logger.warning("Diarization unavailable: %s", self._error)
            return
        try:
            import sherpa_onnx  # noqa: F401

            self._ready = True
            self._mode = "segments"
            self._error = ""
            logger.info("Diarization ready: backend=sherpa max_speakers=%s", self.max_speakers)
        except Exception as exc:
            self._ready = False
            self._mode = "error"
            self._error = str(exc)
            logger.exception("Failed to init sherpa diarization")

    def _load_diart(self) -> None:
        if not self.hf_token:
            self._ready = False
            self._mode = "disabled"
            self._error = "DIAR_BACKEND=diart requires DIAR_HF_TOKEN or HF_TOKEN"
            logger.warning("Diarization unavailable: %s", self._error)
            return
        if not self.emb_path.exists():
            self._ready = False
            self._mode = "disabled"
            self._error = (
                f"diart hybrid needs ERes2Net assign model at {self.emb_path}"
            )
            logger.warning("Diarization unavailable: %s", self._error)
            return
        try:
            os.environ.setdefault("HF_HOME", str(self.diart_cache))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.diart_cache / "hub"))
            # Ensure hub client sees the token even if a nested call omits it.
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
            self.diart_cache.mkdir(parents=True, exist_ok=True)

            from app.diarization.diart_backend import DiartSpeakerTracker

            # Probe model load once (downloads into HF cache).
            probe = DiartSpeakerTracker(
                seg_model=self.diart_seg,
                emb_model=self.diart_emb,
                hf_token=self.hf_token,
                assign_emb_model=self.emb_path,
                max_speakers=self.max_speakers,
                duration=self.diart_duration,
                step=self.diart_step,
                latency=self.diart_latency,
                tau_active=self.diart_tau,
                rho_update=self.diart_rho,
                delta_new=self.diart_delta,
                min_turn=self.diart_min_turn,
                turn_gap=self.diart_turn_gap,
                max_frag=self.diart_max_frag,
                threshold=self.threshold,
                min_new=self.min_new,
                switch_margin=self.switch_margin,
            )
            silence = np.zeros(int(self.diart_duration * SAMPLE_RATE), dtype=np.float32)
            probe.label_spans(silence, update_centroids=True)
            del probe
            self._ready = True
            self._mode = "diart"
            self._error = ""
            logger.info(
                "Diarization ready: backend=diart+eres2net max_speakers=%s",
                self.max_speakers,
            )
        except Exception as exc:
            self._ready = False
            self._mode = "error"
            self._error = str(exc)
            logger.exception("Failed to init diart")

    def create_tracker(self) -> SpeakerTracker:
        if self.backend == "sherpa":
            from app.diarization.sherpa_backend import SherpaSpeakerTracker

            return SherpaSpeakerTracker(
                seg_model=self.seg_path,
                emb_model=self.emb_path,
                threshold=self.threshold,
                max_speakers=self.max_speakers,
                min_new=self.min_new,
                switch_margin=self.switch_margin,
                sticky=self.sticky,
                turn_gap=self.turn_gap,
            )
        if self.backend == "diart":
            from app.diarization.diart_backend import DiartSpeakerTracker

            return DiartSpeakerTracker(
                seg_model=self.diart_seg,
                emb_model=self.diart_emb,
                hf_token=self.hf_token or "",
                assign_emb_model=self.emb_path,
                max_speakers=self.max_speakers,
                duration=self.diart_duration,
                step=self.diart_step,
                latency=self.diart_latency,
                tau_active=self.diart_tau,
                rho_update=self.diart_rho,
                delta_new=self.diart_delta,
                min_turn=self.diart_min_turn,
                turn_gap=self.turn_gap,
                max_frag=max(self.diart_max_frag, 0.9),
                threshold=self.threshold,
                min_new=self.min_new,
                switch_margin=self.switch_margin,
            )
        raise RuntimeError("Diarization backend is off")


diarization_service = DiarizationService()
