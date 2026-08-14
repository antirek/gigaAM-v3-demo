"""Shared embedding-based speaker assignment (ERes2Net via sherpa-onnx)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.diarization import SAMPLE_RATE

logger = logging.getLogger(__name__)


class EmbeddingAssigner:
    """Sticky 2-speaker assigner used by sherpa and diart hybrids."""

    def __init__(
        self,
        emb_model: Path,
        threshold: float = 0.62,
        min_new: float = 0.8,
        max_speakers: int = 2,
        switch_margin: float = 0.08,
        turn_gap: float = 0.35,
        max_frag: float = 0.9,
    ) -> None:
        import sherpa_onnx

        self.sr = SAMPLE_RATE
        self.threshold = threshold
        self.min_new = min_new
        self.max_speakers = max_speakers
        self.switch_margin = switch_margin
        self.turn_gap = turn_gap
        self.max_frag = max_frag
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model))
        )
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._last: int | None = None

    def _embed(self, piece: np.ndarray) -> np.ndarray | None:
        stream = self._ex.create_stream()
        stream.accept_waveform(self.sr, piece.astype(np.float32))
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
        if not self._centroids:
            if update:
                self._centroids.append(emb)
                self._counts.append(1)
            return 0

        sims = [float(np.dot(emb, c)) for c in self._centroids]
        cur = (
            (self._last - 1)
            if self._last and self._last - 1 < len(self._centroids)
            else None
        )
        cur_sim = sims[cur] if cur is not None else -1.0
        best = int(np.argmax(sims))
        best_sim = sims[best]

        if len(self._centroids) >= self.max_speakers:
            other = 1 - cur if cur is not None else best
            other_sim = sims[other] if 0 <= other < len(sims) else -1.0
            min_turn = 0.7
            if cur is not None and gap_before >= self.turn_gap and seconds >= min_turn:
                if other_sim + 0.04 >= cur_sim and other_sim >= 0.30:
                    if update:
                        self._update(other, emb)
                    return other
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

        sim1 = sims[0]
        if sim1 >= self.threshold:
            if update:
                self._update(0, emb)
            return 0

        create_min = 0.55 if sim1 < 0.45 else self.min_new
        if seconds >= create_min and len(self._centroids) < self.max_speakers:
            if update:
                self._centroids.append(emb)
                self._counts.append(1)
            return len(self._centroids) - 1

        if update and sim1 >= 0.35:
            self._update(0, emb)
        return 0

    def label_regions(
        self,
        audio: np.ndarray,
        regions: list[tuple[float, float]],
        update: bool = True,
    ) -> list[tuple[float, float, int]]:
        """Assign speakers to (start_s, end_s) regions relative to audio."""
        spans: list[tuple[float, float, int]] = []
        prev_end = 0.0
        for start, end in regions:
            a, b = int(start * self.sr), int(end * self.sr)
            piece = audio[a:b]
            seconds = (b - a) / self.sr
            if seconds < 0.3:
                continue
            emb = self._embed(piece)
            if emb is None:
                if self._last is not None:
                    spans.append((start, end, self._last))
                    prev_end = end
                continue
            gap = 0.0 if not spans else max(0.0, start - prev_end)
            who = self._assign(emb, seconds, update=update, gap_before=gap)
            if who is None:
                if self._last is not None:
                    spans.append((start, end, self._last))
                    prev_end = end
                continue
            speaker = who + 1
            self._last = speaker
            spans.append((start, end, speaker))
            prev_end = end
        return self._merge(self._absorb(spans))

    def _absorb(
        self, spans: list[tuple[float, float, int]]
    ) -> list[tuple[float, float, int]]:
        if len(spans) < 2:
            return spans
        out = [spans[0]]
        for start, end, speaker in spans[1:]:
            prev_start, prev_end, prev_speaker = out[-1]
            dur = end - start
            gap = start - prev_end
            if speaker != prev_speaker and dur <= self.max_frag and 0 <= gap <= 0.45:
                out[-1] = (prev_start, end, prev_speaker)
                self._last = prev_speaker
                continue
            out.append((start, end, speaker))
        # A short B A → A
        if len(out) < 3:
            return out
        fixed = [out[0]]
        i = 1
        while i < len(out):
            if i + 1 < len(out):
                a0, a1, a_spk = fixed[-1]
                b0, b1, b_spk = out[i]
                c0, c1, c_spk = out[i + 1]
                if (
                    a_spk == c_spk != b_spk
                    and (b1 - b0) <= self.max_frag
                    and b0 - a1 <= 0.45
                    and c0 - b1 <= 0.45
                ):
                    fixed[-1] = (a0, c1, a_spk)
                    self._last = a_spk
                    i += 2
                    continue
            fixed.append(out[i])
            i += 1
        return fixed

    def _merge(
        self, spans: list[tuple[float, float, int]], gap: float = 0.35
    ) -> list[tuple[float, float, int]]:
        if not spans:
            return []
        merged = [spans[0]]
        for start, end, speaker in spans[1:]:
            ps, pe, pspk = merged[-1]
            if speaker == pspk and start - pe <= gap:
                merged[-1] = (ps, end, speaker)
            else:
                merged.append((start, end, speaker))
        return merged

    @property
    def voices(self) -> int:
        return len(self._centroids)

    @property
    def last_speaker(self) -> int | None:
        return self._last

    def snapshot(self) -> dict:
        return {
            "centroids": [c.copy() for c in self._centroids],
            "counts": list(self._counts),
            "last": self._last,
        }

    def restore(self, snap: dict) -> None:
        self._centroids = [c.copy() for c in snap["centroids"]]
        self._counts = list(snap["counts"])
        self._last = snap["last"]
