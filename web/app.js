const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("fileInput");
const audioPreview = document.getElementById("audioPreview");
const metaText = document.getElementById("metaText");
const resultText = document.getElementById("resultText");
const resultUtterances = document.getElementById("resultUtterances");
const timingText = document.getElementById("timingText");

let mediaRecorder = null;
let chunks = [];
let recordedBlob = null;
let streamSocket = null;
let pollTimer = null;
let diarReady = false;

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function renderResult(data) {
  const utterances = data.utterances || [];
  if (data.diarization && utterances.length) {
    resultText.classList.add("hidden");
    resultUtterances.classList.remove("hidden");
    resultUtterances.innerHTML = utterances
      .filter((u) => u.text)
      .map((u) => {
        const spk = u.speaker || "?";
        const partial = u.partial ? " partial" : "";
        return `<div class="utterance speaker-${spk}${partial}">
          <span class="speaker-tag">Спикер ${spk}</span>
          <span class="utterance-text">${escapeHtml(u.text)}</span>
        </div>`;
      })
      .join("");
  } else {
    resultUtterances.classList.add("hidden");
    resultText.classList.remove("hidden");
    resultText.textContent = data.text || "—";
  }

  const committed =
    data.committed_until_s != null ? `, committed: ${data.committed_until_s.toFixed(0)}s` : "";
  const speakers =
    data.speakers_count != null ? `, speakers: ${data.speakers_count}` : "";
  if (data.inference_s != null) {
    timingText.textContent =
      `Inference: ${data.inference_s.toFixed(2)}s, audio: ${data.duration_s?.toFixed(1) ?? "?"}s${committed}${speakers} (${data.mode || data.type})`;
  }
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function pollHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    statusDot.className = "status-dot";
    diarReady = Boolean(data.diarization?.ready);
    const diarNote = diarReady
      ? ` · diar: ${data.diarization.mode}, max ${data.diarization.max_speakers}`
      : data.diarization?.error
        ? ` · diar off: ${data.diarization.error}`
        : " · diar off";

    if (data.status === "ready") {
      statusDot.classList.add("ready");
      statusText.textContent = `Модель готова (${data.device}, загрузка ${data.load_time_s?.toFixed(1) ?? "?"}s)${diarNote}`;
      recordBtn.disabled = false;
      sendBtn.disabled = !recordedBlob && getMode() === "batch";
    } else if (data.status === "loading") {
      statusText.textContent = "Загрузка модели... (первый запуск может занять несколько минут)";
      recordBtn.disabled = true;
      sendBtn.disabled = true;
    } else {
      statusDot.classList.add("error");
      statusText.textContent = `Ошибка: ${data.message}`;
      recordBtn.disabled = true;
      sendBtn.disabled = true;
    }
  } catch (err) {
    statusDot.classList.add("error");
    statusText.textContent = "API недоступен";
    recordBtn.disabled = true;
    sendBtn.disabled = true;
  }
}

pollTimer = setInterval(pollHealth, 2000);
pollHealth();

recordBtn.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  recordedBlob = null;
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
    if (getMode() === "stream" && streamSocket?.readyState === WebSocket.OPEN) {
      streamSocket.send(e.data);
    }
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    recordedBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    audioPreview.src = URL.createObjectURL(recordedBlob);
    metaText.textContent = `Записано: ${(recordedBlob.size / 1024).toFixed(1)} KB`;
    sendBtn.disabled = getMode() === "batch" ? false : true;
    if (getMode() === "stream" && streamSocket?.readyState === WebSocket.OPEN) {
      streamSocket.send("finalize");
    }
  };

  if (getMode() === "stream") {
    resultText.classList.remove("hidden");
    resultUtterances.classList.add("hidden");
    resultText.textContent = diarReady
      ? "Потоковое распознавание + диаризация..."
      : "Потоковое распознавание (диаризация недоступна)...";
    streamSocket = new WebSocket(`${location.origin.replace("http", "ws")}/ws/stream`);
    streamSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "error") {
        resultUtterances.classList.add("hidden");
        resultText.classList.remove("hidden");
        resultText.textContent = `Ошибка: ${data.message}`;
        timingText.textContent = "";
        return;
      }
      renderResult(data);
    };
    streamSocket.onclose = () => {
      if (!String(resultText.textContent || "").startsWith("Ошибка:")) {
        timingText.textContent = (timingText.textContent || "") + " · поток закрыт";
      }
    };
    await new Promise((resolve, reject) => {
      streamSocket.onopen = resolve;
      streamSocket.onerror = reject;
    });
  }

  mediaRecorder.start(500);
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  sendBtn.disabled = true;
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  recordBtn.disabled = false;
  stopBtn.disabled = true;
});

sendBtn.addEventListener("click", async () => {
  if (!recordedBlob) return;
  resultUtterances.classList.add("hidden");
  resultText.classList.remove("hidden");
  resultText.textContent = "Распознавание...";
  timingText.textContent = "";
  const form = new FormData();
  form.append("file", recordedBlob, "recording.webm");
  const res = await fetch("/transcribe", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    resultText.textContent = `Ошибка: ${data.detail || res.statusText}`;
    return;
  }
  resultText.textContent = data.text || "—";
  timingText.textContent = `Inference: ${data.inference_s.toFixed(2)}s, audio: ${data.duration_s.toFixed(1)}s, chunks: ${data.chunks}`;
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  recordedBlob = file;
  audioPreview.src = URL.createObjectURL(file);
  metaText.textContent = `Файл: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  sendBtn.disabled = getMode() !== "batch";
  if (getMode() === "batch") {
    resultUtterances.classList.add("hidden");
    resultText.classList.remove("hidden");
    resultText.textContent = "Распознавание...";
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      resultText.textContent = `Ошибка: ${data.detail || res.statusText}`;
      return;
    }
    resultText.textContent = data.text || "—";
    timingText.textContent = `Inference: ${data.inference_s.toFixed(2)}s`;
  }
});

document.querySelectorAll('input[name="mode"]').forEach((el) => {
  el.addEventListener("change", () => {
    sendBtn.disabled = getMode() === "stream" || !recordedBlob;
  });
});
