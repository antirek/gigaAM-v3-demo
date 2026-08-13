"""
Live speaker diarization for streaming, following Charoite's SegmentTracker approach.

Charoite does NOT use classic Silero VAD for the good live path. They use:
  pyannote segmentation ONNX (via sherpa-onnx) → speech spans
  + ERes2Net embedding ONNX → speaker vectors
  + cosine tracker with sticky/hysteresis

We cap at 2 speakers and return per-span labels (not one majority label
for a whole ASR window — that was overwriting early "Спикер 1" text).
"""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def load_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


class SegmentSpeakerTracker:
    """Charoite-style live tracker: pyannote segments + ERes2Net embeddings."""

    def __init__(
        self,
        seg_model: Path,
        emb_model: Path,
        threshold: float = 0.62,
        min_segment: float = 0.4,
        min_new: float = 1.5,
        max_speakers: int = 2,
        sticky: float = 0.10,
        switch_margin: float = 0.08,
        turn_gap: float = 0.35,
    ) -> None:
        import sherpa_onnx

        self.sr = SAMPLE_RATE
        self.threshold = threshold
        self.min_segment = min_segment
        # New voice needs a longer clean span — short "угу" creates phantom Speaker 2.
        self.min_new = min_new
        self.max_speakers = max_speakers
        self.sticky = sticky
        self.switch_margin = switch_margin
        self.turn_gap = turn_gap
        # IMPORTANT: do NOT force num_clusters=max_speakers on each window.
        # That invents a fake second voice inside single-speaker stretches
        # (see result.log: "Четыре, пять. Давай, говори." → Speakers 2).
        # Global tracker already caps at max_speakers.
        self._diar = sherpa_onnx.OfflineSpeakerDiarization(
            sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(seg_model)
                    )
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(emb_model)
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=-1,
                    threshold=0.8,
                ),
                min_duration_on=0.25,
                min_duration_off=0.3,
            )
        )
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model))
        )
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._last: int | None = None

    def _embed(self, piece: np.ndarray) -> np.ndarray | None:
        stream = self._ex.create_stream()
        stream.accept_waveform(self.sr, piece)
        stream.input_finished()
        if not self._ex.is_ready(stream):
            return None
        emb = np.asarray(self._ex.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else None

    def _update(self, index: int, emb: np.ndarray) -> None:
        k = self._counts[index]
        centroid = (self._centroids[index] * k + emb) / (k + 1)
        self._centroids[index] = centroid / float(np.linalg.norm(centroid))
        self._counts[index] += 1

    def _assign(
        self,
        emb: np.ndarray,
        seconds: float,
        update: bool = True,
        gap_before: float = 0.0,
    ) -> int | None:
        """Return 0-based speaker index.

        When only Speakers 1 exists, open Speakers 2 if cosine to Speakers 1
        is below match threshold and the span is long enough. Do not sticky-
        return Speakers 1 before that check (regression in result2.log).

        When both voices are known, a pause before the span is treated as a
        turn change: keep the previous speaker only if they are clearly closer.
        """
        if not self._centroids:
            if update:
                self._centroids.append(emb)
                self._counts.append(1)
            else:
                # First voice in a partial: still report Speakers 1 without training.
                pass
            return 0

        sims = [float(np.dot(emb, c)) for c in self._centroids]
        cur = (self._last - 1) if self._last and self._last - 1 < len(self._centroids) else None
        cur_sim = sims[cur] if cur is not None else -1.0
        best = int(np.argmax(sims))
        best_sim = sims[best]

        # --- Both speakers known ---
        if len(self._centroids) >= self.max_speakers:
            other = 1 - cur if cur is not None else best
            other_sim = sims[other] if 0 <= other < len(sims) else -1.0
            # After a pause, bias toward the other speaker (Q→A turn taking).
            if cur is not None and gap_before >= self.turn_gap:
                if other_sim + 0.04 >= cur_sim and other_sim >= 0.30:
                    if update:
                        self._update(other, emb)
                    return other
                # Keep current only if they clearly win.
                if cur_sim - other_sim >= self.switch_margin:
                    if update and cur_sim >= 0.35:
                        self._update(cur, emb)
                    return cur
            if cur is not None and best != cur:
                if best_sim >= cur_sim and best_sim >= 0.30:
                    if update:
                        self._update(best, emb)
                    return best
                if cur_sim - best_sim >= self.switch_margin:
                    if update and cur_sim >= 0.35:
                        self._update(cur, emb)
                    return cur
            if update and best_sim >= 0.35:
                self._update(best, emb)
            return best if best_sim >= 0.30 else (cur if cur is not None else best)

        # --- Only Speakers 1 known ---
        sim1 = sims[0]

        # Strong match → Speakers 1
        if sim1 >= self.threshold:
            if update:
                self._update(0, emb)
            return 0

        # Not a match to Speakers 1 → open Speakers 2
        create_min = 0.55 if sim1 < 0.45 else self.min_new
        if seconds >= create_min and len(self._centroids) < self.max_speakers:
            # Always register the new centroid (even during partial), otherwise
            # provisional Speakers 2 breaks _last / sims indexing.
            self._centroids.append(emb)
            self._counts.append(1)
            return len(self._centroids) - 1

        # Short / ambiguous → stay Speakers 1
        if update and sim1 >= 0.35:
            self._update(0, emb)
        return 0

    def label_spans(
        self, chunk: np.ndarray, update_centroids: bool = True
    ) -> list[tuple[float, float, int]]:
        """
        Return speech spans as (start_s, end_s, speaker_1based) relative to chunk.
        Adjacent same-speaker spans with tiny gaps are merged.
        """
        if chunk is None or len(chunk) < int(self.min_segment * self.sr):
            return []

        try:
            segments = self._diar.process(chunk).sort_by_start_time()
        except Exception:
            logger.exception("Diarization process failed")
            return []

        spans: list[tuple[float, float, int]] = []
        prev_end = 0.0
        for seg in segments:
            start = float(seg.start)
            end = float(seg.end)
            a, b = int(start * self.sr), int(end * self.sr)
            piece = chunk[a:b]
            seconds = (b - a) / self.sr
            if seconds < self.min_segment:
                continue
            emb = self._embed(piece)
            if emb is None:
                continue
            gap_before = max(0.0, start - prev_end)
            who = self._assign(
                emb, seconds, update=update_centroids, gap_before=gap_before
            )
            if who is None:
                if self._last is not None:
                    spans.append((start, end, self._last))
                    prev_end = end
                continue
            speaker = who + 1
            self._last = speaker
            spans.append((start, end, speaker))
            prev_end = end

        return self._merge_spans(spans)

    def _merge_spans(
        self, spans: list[tuple[float, float, int]], gap: float = 0.35
    ) -> list[tuple[float, float, int]]:
        if not spans:
            return []
        merged = [spans[0]]
        for start, end, speaker in spans[1:]:
            prev_start, prev_end, prev_speaker = merged[-1]
            if speaker == prev_speaker and start - prev_end <= gap:
                merged[-1] = (prev_start, end, speaker)
            else:
                merged.append((start, end, speaker))
        return merged

    def label(self, chunk: np.ndarray) -> int | None:
        """Majority speaker for chunk (legacy helper)."""
        spans = self.label_spans(chunk, update_centroids=True)
        if not spans:
            return self._last
        talk: dict[int, float] = {}
        for start, end, speaker in spans:
            talk[speaker] = talk.get(speaker, 0.0) + (end - start)
        self._last = max(talk, key=lambda k: talk[k])
        return self._last

    @property
    def voices(self) -> int:
        return len(self._centroids)


class DiarizationService:
    def __init__(self) -> None:
        self.enabled = os.getenv("DIAR_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.emb_path = Path(os.getenv("DIAR_EMB_MODEL", "/data/diar/embedding.onnx"))
        self.seg_path = Path(os.getenv("DIAR_SEG_MODEL", "/data/diar/segmentation.onnx"))
        self.max_speakers = int(os.getenv("DIAR_MAX_SPEAKERS", "2"))
        self.threshold = float(os.getenv("DIAR_THRESHOLD", "0.62"))
        self.min_new = float(os.getenv("DIAR_MIN_NEW_SEC", "0.8"))
        self.switch_margin = float(os.getenv("DIAR_SWITCH_MARGIN", "0.08"))
        self.sticky = float(os.getenv("DIAR_STICKY", "0.08"))
        self.turn_gap = float(os.getenv("DIAR_TURN_GAP", "0.35"))
        self._ready = False
        self._error = ""
        self._mode = "disabled"

    @property
    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "ready": self._ready,
            "mode": self._mode,
            "max_speakers": self.max_speakers,
            "threshold": self.threshold,
            "min_new_sec": self.min_new,
            "switch_margin": self.switch_margin,
            "sticky": self.sticky,
            "turn_gap": self.turn_gap,
            "embedding_model": str(self.emb_path),
            "segmentation_model": str(self.seg_path),
            "error": self._error,
        }

    def load(self) -> None:
        if not self.enabled:
            self._mode = "disabled"
            self._ready = False
            logger.info("Diarization disabled by config")
            return
        if not self.emb_path.exists():
            self._mode = "disabled"
            self._error = f"missing {self.emb_path}"
            logger.warning("Diarization unavailable: %s", self._error)
            return
        if not self.seg_path.exists():
            self._mode = "disabled"
            self._error = (
                f"missing {self.seg_path} "
                "(need pyannote segmentation for Charoite-quality live)"
            )
            logger.warning("Diarization unavailable: %s", self._error)
            return
        try:
            import sherpa_onnx  # noqa: F401

            self._ready = True
            self._mode = "segments"
            self._error = ""
            logger.info(
                "Diarization ready (pyannote-seg + ERes2Net, max_speakers=%s)",
                self.max_speakers,
            )
        except Exception as exc:
            self._ready = False
            self._mode = "error"
            self._error = str(exc)
            logger.exception("Failed to init diarization")

    def create_tracker(self) -> SegmentSpeakerTracker:
        return SegmentSpeakerTracker(
            seg_model=self.seg_path,
            emb_model=self.emb_path,
            threshold=self.threshold,
            max_speakers=self.max_speakers,
            min_new=self.min_new,
            switch_margin=self.switch_margin,
            sticky=self.sticky,
            turn_gap=self.turn_gap,
        )


diarization_service = DiarizationService()
