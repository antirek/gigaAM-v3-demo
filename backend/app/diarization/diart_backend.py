"""diart online segmentation + ERes2Net speaker assignment."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature

from app.diarization import SAMPLE_RATE
from app.diarization.assigner import EmbeddingAssigner

logger = logging.getLogger(__name__)


class DiartSpeakerTracker:
    """
    diart finds speech regions (online). Speakers are assigned with the same
    ERes2Net sticky tracker that works well for sherpa — diart labels alone
    thrash on a single close mic (see result-diart1.log).
    """

    def __init__(
        self,
        seg_model: str,
        emb_model: str,
        hf_token: str,
        assign_emb_model: Path,
        max_speakers: int = 2,
        duration: float = 5.0,
        step: float = 0.5,
        latency: float = 2.0,
        tau_active: float = 0.45,
        rho_update: float = 0.15,
        delta_new: float = 0.5,
        min_turn: float = 0.7,
        turn_gap: float = 0.35,
        max_frag: float = 0.9,
        threshold: float = 0.62,
        min_new: float = 0.8,
        switch_margin: float = 0.08,
    ) -> None:
        from diart import SpeakerDiarization, SpeakerDiarizationConfig
        from diart.models import EmbeddingModel, PowersetAdapter, SegmentationModel
        from pyannote.audio import Model

        self.sr = SAMPLE_RATE
        self.max_speakers = max_speakers
        self.duration = duration
        self.step = step
        self.turn_gap = turn_gap
        self._win = int(round(duration * self.sr))
        self._step = int(round(step * self.sr))

        def load_segmentation():
            model = Model.from_pretrained(seg_model, token=hf_token)
            specs = getattr(model, "specifications", None)
            if specs is not None and getattr(specs, "powerset", False):
                model = PowersetAdapter(model)
            return model

        def load_embedding():
            try:
                return Model.from_pretrained(emb_model, token=hf_token)
            except Exception:
                from pyannote.audio.pipelines.speaker_verification import (
                    PretrainedSpeakerEmbedding,
                )

                return PretrainedSpeakerEmbedding(emb_model, token=hf_token)

        segmentation = SegmentationModel(load_segmentation)
        embedding = EmbeddingModel(load_embedding)
        self._config = SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            duration=duration,
            step=step,
            latency=min(latency, duration),
            max_speakers=max_speakers,
            device=torch.device("cpu"),
            sample_rate=self.sr,
            tau_active=tau_active,
            rho_update=rho_update,
            delta_new=delta_new,
        )
        self._pipeline = SpeakerDiarization(self._config)
        self._assigner = EmbeddingAssigner(
            emb_model=assign_emb_model,
            threshold=threshold,
            min_new=min_new,
            max_speakers=max_speakers,
            switch_margin=switch_margin,
            turn_gap=turn_gap,
            max_frag=max_frag,
        )
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._next_win = 0
        self._emitted_until = 0.0

    def _snapshot(self) -> dict[str, Any]:
        return {
            "clustering": copy.deepcopy(self._pipeline.clustering),
            "chunk_buffer": list(self._pipeline.chunk_buffer),
            "pred_buffer": list(self._pipeline.pred_buffer),
            "buffer": self._buffer.copy(),
            "next_win": self._next_win,
            "emitted_until": self._emitted_until,
            "timestamp_shift": self._pipeline.timestamp_shift,
            "assigner": self._assigner.snapshot(),
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        self._pipeline.clustering = snap["clustering"]
        self._pipeline.chunk_buffer = snap["chunk_buffer"]
        self._pipeline.pred_buffer = snap["pred_buffer"]
        self._pipeline.timestamp_shift = snap["timestamp_shift"]
        self._buffer = snap["buffer"]
        self._next_win = snap["next_win"]
        self._emitted_until = snap["emitted_until"]
        self._assigner.restore(snap["assigner"])

    def _speech_regions(self, chunk_origin: int) -> list[tuple[float, float]]:
        """Run diart; return newly finalized speech intervals relative to chunk."""
        raw: list[tuple[float, float]] = []
        chunk_origin_t = chunk_origin / self.sr
        chunk_len_t = (len(self._buffer) - chunk_origin) / self.sr

        while self._next_win + self._win <= len(self._buffer):
            piece = self._buffer[self._next_win : self._next_win + self._win]
            data = piece.reshape(-1, 1)
            start_t = self._next_win / self.sr
            resolution = SlidingWindow(
                start=start_t, duration=1.0 / self.sr, step=1.0 / self.sr
            )
            feat = SlidingWindowFeature(data, resolution)
            try:
                outputs = self._pipeline([feat])
            except Exception:
                logger.exception("diart pipeline step failed")
                self._next_win += self._step
                continue

            for annotation, _waveform in outputs:
                for segment, _track, _label in annotation.itertracks(yield_label=True):
                    abs_start = max(float(segment.start), self._emitted_until)
                    abs_end = float(segment.end)
                    if abs_end - abs_start < 0.2:
                        continue
                    rel_start = max(0.0, abs_start - chunk_origin_t)
                    rel_end = min(chunk_len_t, abs_end - chunk_origin_t)
                    if rel_end - rel_start < 0.2:
                        continue
                    raw.append((rel_start, rel_end))
                    self._emitted_until = max(self._emitted_until, abs_end)

            self._next_win += self._step

        return self._merge_regions(raw)

    def _merge_regions(
        self, regions: list[tuple[float, float]], gap: float = 0.35
    ) -> list[tuple[float, float]]:
        if not regions:
            return []
        regions = sorted(regions)
        merged = [regions[0]]
        for start, end in regions[1:]:
            ps, pe = merged[-1]
            if start <= pe + gap:
                merged[-1] = (ps, max(pe, end))
            else:
                merged.append((start, end))
        return merged

    def label_spans(
        self, chunk: np.ndarray, update_centroids: bool = True
    ) -> list[tuple[float, float, int]]:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size < int(0.3 * self.sr):
            return []

        snap = None if update_centroids else self._snapshot()
        try:
            origin = len(self._buffer)
            self._buffer = np.concatenate([self._buffer, audio])
            if len(self._buffer) < self._win:
                return []
            regions = self._speech_regions(origin)
            if not regions:
                return []
            # Assign on the newly appended chunk only (regions are chunk-relative).
            spans = self._assigner.label_regions(
                audio, regions, update=update_centroids
            )
            if spans:
                logger.info(
                    "diart+eres spans=%s speakers=%s voices=%s",
                    len(spans),
                    sorted({s for *_, s in spans}),
                    self._assigner.voices,
                )
            return spans
        finally:
            if snap is not None:
                self._restore(snap)

    @property
    def voices(self) -> int:
        return min(self._assigner.voices, self.max_speakers)

    @property
    def last_speaker(self) -> int | None:
        return self._assigner.last_speaker
