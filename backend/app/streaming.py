"""
Pseudo-streaming via sliding segments + optional live diarization.

When diarization is on, ASR runs per speech span (pyannote), not one majority
label for a whole 20s window — that overwrote early "Спикер 1" lines.
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
from app.diarization import DiarizationService, load_wav_mono
from app.model import GigaAMService

logger = logging.getLogger(__name__)

STREAM_PARTIAL_SECONDS = float(os.getenv("STREAM_CHUNK_SECONDS", "3"))
STREAM_SEGMENT_SECONDS = float(os.getenv("STREAM_SEGMENT_SECONDS", "12"))


class StreamSession:
    def __init__(
        self,
        service: GigaAMService,
        diar_service: DiarizationService | None = None,
    ) -> None:
        self.service = service
        self.diar_service = diar_service
        self.buffer = bytearray()
        self.last_transcribe_at = 0.0
        self.suffix = ".webm"
        self.committed_utterances: list[dict] = []
        self.committed_until = 0.0
        self.partial_utterances: list[dict] = []
        self.busy = False
        self.tracker = None
        if diar_service and diar_service._ready:
            try:
                self.tracker = diar_service.create_tracker()
            except Exception:
                logger.exception("Failed to create speaker tracker for session")
                self.tracker = None

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
        lines: list[str] = []
        for item in self.committed_utterances + self.partial_utterances:
            speaker = item.get("speaker")
            text = item.get("text") or ""
            if not text:
                continue
            if speaker:
                lines.append(f"Спикер {speaker}: {text}")
            else:
                lines.append(text)
        return "\n".join(lines).strip()

    def _utterances_payload(self) -> list[dict]:
        items = list(self.committed_utterances)
        for item in self.partial_utterances:
            if item.get("text"):
                items.append({**item, "partial": True})
        return items

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

    def _process_range(
        self, wav_path: str, start: float, end: float, *, update_centroids: bool
    ) -> list[dict]:
        """ASR (+ optional per-span diarization) for [start, end)."""
        if end - start < 0.3:
            return []

        if self.tracker is None:
            text = self._transcribe_range(wav_path, start, end)
            if not text:
                return []
            return [
                {
                    "speaker": None,
                    "text": text,
                    "start_s": round(start, 2),
                    "end_s": round(end, 2),
                }
            ]

        segment = extract_wav_segment(wav_path, start, end - start)
        try:
            audio = load_wav_mono(segment)
            spans = self.tracker.label_spans(
                audio, update_centroids=update_centroids
            )
        except Exception:
            logger.exception("Span labeling failed")
            spans = []
        finally:
            cleanup_paths([segment])

        utterances: list[dict] = []
        for rel_start, rel_end, speaker in spans:
            abs_start = start + rel_start
            abs_end = start + rel_end
            if abs_end - abs_start < 0.35:
                continue
            text = self._transcribe_range(wav_path, abs_start, abs_end)
            if not text:
                continue
            utterances.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "start_s": round(abs_start, 2),
                    "end_s": round(abs_end, 2),
                }
            )

        if utterances:
            return self._merge_utterances(utterances)

        # Fallback: whole window, keep last known speaker.
        text = self._transcribe_range(wav_path, start, end)
        if not text:
            return []
        return [
            {
                "speaker": self.tracker._last if self.tracker else None,
                "text": text,
                "start_s": round(start, 2),
                "end_s": round(end, 2),
            }
        ]

    def _merge_utterances(self, utterances: list[dict]) -> list[dict]:
        if not utterances:
            return []
        merged = [dict(utterances[0])]
        for item in utterances[1:]:
            prev = merged[-1]
            same_speaker = prev.get("speaker") == item.get("speaker")
            close = float(item["start_s"]) - float(prev["end_s"]) <= 0.4
            if same_speaker and close:
                prev["text"] = f"{prev['text']} {item['text']}".strip()
                prev["end_s"] = item["end_s"]
            else:
                merged.append(dict(item))
        return merged

    def _response(self, msg_type: str, duration: float, inference_s: float) -> dict:
        return {
            "type": msg_type,
            "text": self._display_text(),
            "utterances": self._utterances_payload(),
            "duration_s": duration,
            "inference_s": inference_s,
            "committed_until_s": self.committed_until,
            "speakers_count": self.tracker.voices if self.tracker else 0,
            "diarization": bool(self.tracker),
            "mode": "pseudo_streaming",
            "note": "Per-span ASR + Charoite-style live diarization"
            if self.tracker
            else "Sliding-segment pseudo-streaming (diarization off)",
        }

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
                pieces = self._process_range(
                    wav_path, seg_start, seg_end, update_centroids=True
                )
                inference_total += time.perf_counter() - t0
                self.committed_utterances.extend(pieces)
                self.committed_until = seg_end
                self.partial_utterances = []

            open_start = self.committed_until
            if duration - open_start >= 0.5:
                t0 = time.perf_counter()
                # Partials must not train centroids — commit window owns updates.
                self.partial_utterances = self._process_range(
                    wav_path, open_start, duration, update_centroids=False
                )
                inference_total += time.perf_counter() - t0
            else:
                self.partial_utterances = []

            return self._response("partial", duration, inference_total)
        finally:
            cleanup_paths([p for p in (raw_path, wav_path) if p])
            self.busy = False

    def finalize(self) -> dict:
        if not self.buffer:
            return self._response("final", 0.0, 0.0)

        raw_path = ""
        wav_path = ""
        try:
            raw_path, wav_path, duration = self._materialize_wav()
            inference_total = 0.0

            while duration - self.committed_until >= STREAM_SEGMENT_SECONDS:
                seg_start = self.committed_until
                seg_end = self.committed_until + STREAM_SEGMENT_SECONDS
                t0 = time.perf_counter()
                pieces = self._process_range(
                    wav_path, seg_start, seg_end, update_centroids=True
                )
                inference_total += time.perf_counter() - t0
                self.committed_utterances.extend(pieces)
                self.committed_until = seg_end

            if duration - self.committed_until >= 0.3:
                t0 = time.perf_counter()
                pieces = self._process_range(
                    wav_path,
                    self.committed_until,
                    duration,
                    update_centroids=True,
                )
                inference_total += time.perf_counter() - t0
                self.committed_utterances.extend(pieces)
                self.committed_until = duration

            self.partial_utterances = []
            return self._response("final", duration, inference_total)
        finally:
            cleanup_paths([p for p in (raw_path, wav_path) if p])
