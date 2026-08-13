import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.audio_utils import cleanup_paths, convert_to_wav, save_upload_to_temp
from app.diarization import diarization_service
from app.model import ModelState, gigaam_service
from app.streaming import StreamSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _load_services() -> None:
    gigaam_service.load()
    diarization_service.load()


@asynccontextmanager
async def lifespan(_: FastAPI):
    thread = threading.Thread(target=_load_services, daemon=True)
    thread.start()
    yield


app = FastAPI(title="GigaAM Recognition API", lifespan=lifespan)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    status = gigaam_service.status
    return {
        "status": status.state,
        "message": status.message,
        "model_name": gigaam_service.model_name,
        "device": gigaam_service.device,
        "load_time_s": status.load_time_s,
        "max_audio_seconds": gigaam_service.max_audio_seconds,
        "streaming_native": False,
        "streaming_mode": "pseudo_streaming",
        "diarization": diarization_service.status,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if gigaam_service.status.state == ModelState.LOADING:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if gigaam_service.status.state == ModelState.ERROR:
        raise HTTPException(status_code=500, detail=gigaam_service.status.message)

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    raw_path = save_upload_to_temp(data, suffix=suffix)
    wav_path = convert_to_wav(raw_path)
    try:
        return gigaam_service.transcribe_wav(wav_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_paths([raw_path, wav_path])


@app.websocket("/ws/stream")
async def stream_transcribe(websocket: WebSocket):
    await websocket.accept()

    if gigaam_service.status.state != ModelState.READY:
        await websocket.send_json(
            {
                "type": "error",
                "message": gigaam_service.status.message or "Model not ready",
            }
        )
        await websocket.close()
        return

    session = StreamSession(gigaam_service, diarization_service)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                session.add_chunk(message["bytes"])
                partial = await asyncio.to_thread(session.transcribe_partial)
                if partial:
                    await websocket.send_json(partial)
            elif "text" in message and message["text"] == "finalize":
                final = await asyncio.to_thread(session.finalize)
                await websocket.send_json(final)
                break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except Exception as exc:
        logger.exception("Streaming error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
