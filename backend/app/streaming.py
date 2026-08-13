"""
Pseudo-streaming via sliding segments.

Native causal streaming is not in the public GigaAM API. We convert the growing
WebM buffer to WAV, periodically commit ~20s segments, and keep only a short
open tail for partial transcripts — so recognition continues past 60s.
"""

import logging
import os
import time

from app.audio_utils import (
    cleanup_paths,
    convert_to_wav,
    extract_wav_segment,
    get_duration_seconds,
    save_upload_to_temp,
)
from app.model import GigaAMService

logger = logging.getLogger(__name__)

STREAM_PARTIAL_SECONDS = float(os.getenv("STREAM_CHUNK_SECONDS", "3"))
STREAM_SEGMENT_SECONDS = float(os.getenv("STREAM_SEGMENT_SECONDS", "20"))


class StreamSession:
    def __init__(self, service: GigaAMService) -> None:
        self.service = service
        self.buffer = bytearray()
        self.last_transcribe_at = 0.0
        self.suffix = ".webm"
        self.committed_text: list[str] = []
        self.committed_until = 0.0
        self.partial_text = ""
        self.busy = False

    def add_chunk(self, data: bytes, suffix: str | None = None) -> None:
        if suffix:
            self.suffix = suffix
        self.buffer.extend(data)

    def should_transcribe(self) -> bool:
        if self.busy or len(self.buffer) < 1000:
            return False
        now = time.monotonic()
        return (now - self.last_transcribe_at) >= STREAM_PARTIAL_SECONDS

    def _display_text(self) -> str:
        parts = [t for t in self.committed_text if t]
        if self.partial_text:
            parts.append(self.partial_text)
        return " ".join(parts).strip()

    def _materialize_wav(self) -> tuple[str, str, float]:
        raw_path = save_upload_to_temp(bytes(self.buffer), suffix=self.suffix)
        wav_path = convert_to_wav(raw_path)
        duration = get_duration_seconds(wav_path)
        return raw_path, wav_path, duration

    def _transcribe_range(self, wav_path: str, start: float, end: float) -> str:
        if end - start < 0.3:
            return ""
        segment = extract_wav_segment(wav_path, start, end - start)
        try:
            result = self.service.transcribe_wav(segment, enforce_max=False)
            return (result["text"] or "").strip()
        finally:
            cleanup_paths([segment])

    def transcribe_partial(self) -> dict | None:
        if not self.should_transcribe():
            return None

        self.busy = True
        self.last_transcribe_at = time.monotonic()
        raw_path = ""
        wav_path = ""
        try:
            raw_path, wav_path, duration = self._materialize_wav()
            inference_total = 0.0

            while duration - self.committed_until >= STREAM_SEGMENT_SECONDS:
                seg_start = self.committed_until
                seg_end = self.committed_until + STREAM_SEGMENT_SECONDS
                t0 = time.perf_counter()
                text = self._transcribe_range(wav_path, seg_start, seg_end)
                inference_total += time.perf_counter() - t0
                if text:
                    self.committed_text.append(text)
                self.committed_until = seg_end
                self.partial_text = ""

            open_start = self.committed_until
            if duration - open_start >= 0.5:
                t0 = time.perf_counter()
                self.partial_text = self._transcribe_range(wav_path, open_start, duration)
                inference_total += time.perf_counter() - t0
            else:
                self.partial_text = ""

            return {
                "type": "partial",
                "text": self._display_text(),
                "duration_s": duration,
                "inference_s": inference_total,
                "committed_until_s": self.committed_until,
                "mode": "pseudo_streaming",
                "note": "Sliding-segment pseudo-streaming (not native causal streaming)",
            }
        finally:
            cleanup_paths([p for p in (raw_path, wav_path) if p])
            self.busy = False

    def finalize(self) -> dict:
        if not self.buffer:
            return {
                "type": "final",
                "text": self._display_text(),
                "mode": "pseudo_streaming",
            }

        raw_path = ""
        wav_path = ""
        try:
            raw_path, wav_path, duration = self._materialize_wav()
            inference_total = 0.0

            while duration - self.committed_until >= STREAM_SEGMENT_SECONDS:
                seg_start = self.committed_until
                seg_end = self.committed_until + STREAM_SEGMENT_SECONDS
                t0 = time.perf_counter()
                text = self._transcribe_range(wav_path, seg_start, seg_end)
                inference_total += time.perf_counter() - t0
                if text:
                    self.committed_text.append(text)
                self.committed_until = seg_end

            if duration - self.committed_until >= 0.3:
                t0 = time.perf_counter()
                text = self._transcribe_range(wav_path, self.committed_until, duration)
                inference_total += time.perf_counter() - t0
                if text:
                    self.committed_text.append(text)
                self.committed_until = duration

            self.partial_text = ""
            return {
                "type": "final",
                "text": self._display_text(),
                "duration_s": duration,
                "inference_s": inference_total,
                "mode": "pseudo_streaming",
            }
        finally:
            cleanup_paths([p for p in (raw_path, wav_path) if p])
