import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 25


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-nostdin", "-y", *args]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        logger.error("ffmpeg failed: %s", stderr)
        raise RuntimeError("Audio conversion failed") from exc


def save_upload_to_temp(data: bytes, suffix: str = ".webm") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def convert_to_wav(input_path: str) -> str:
    output_path = input_path + ".wav"
    _run_ffmpeg(
        [
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "wav",
            output_path,
        ]
    )
    return output_path


def get_duration_seconds(wav_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_wav_segment(wav_path: str, start: float, duration: float) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    _run_ffmpeg(
        [
            "-i",
            wav_path,
            "-ss",
            str(max(0.0, start)),
            "-t",
            str(max(0.1, duration)),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "wav",
            out_path,
        ]
    )
    return out_path


def split_wav_fixed(wav_path: str, chunk_seconds: int = MAX_CHUNK_SECONDS) -> list[str]:
    duration = get_duration_seconds(wav_path)
    if duration <= chunk_seconds:
        return [wav_path]

    chunks: list[str] = []
    start = 0.0
    index = 0
    while start < duration:
        chunk_path = extract_wav_segment(wav_path, start, chunk_seconds)
        chunks.append(chunk_path)
        start += chunk_seconds
        index += 1
    return chunks


def cleanup_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp file: %s", path)
