const video = document.getElementById("video");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const statusChip = document.querySelector("[data-status]");
const verdictEl = document.getElementById("verdict");
const deviceEl = document.getElementById("device");
const bar = document.getElementById("probability-bar");
const probText = document.getElementById("probability-text");
const latencyEl = document.getElementById("latency");
const lastUpdateEl = document.getElementById("last-update");
const canvas = document.getElementById("frame-canvas");
const ctx = canvas.getContext("2d");
const deepfakeLabel = document.getElementById("deepfake-label");
const deepfakeProb = document.getElementById("deepfake-prob");
const deepfakeAge = document.getElementById("deepfake-age");
const blinkOverlay = document.getElementById("blink-overlay");
const blinkText = document.getElementById("blink-text");
const behaviorScoreEl = document.getElementById("behavior-score");
const behaviorRiskEl = document.getElementById("behavior-risk");
const behaviorUpdatedEl = document.getElementById("behavior-updated");
const behaviorHistoryEl = document.getElementById("behavior-history");
const behaviorFlagsEl = document.getElementById("behavior-flags");
const API_BASE = window.APP_CONFIG?.apiBase || "";
const BEHAVIOR_ENABLED = window.APP_CONFIG?.behaviorEnabled ?? true;
const SESSION_ID =
  window.crypto?.randomUUID?.() ||
  `session-${Math.random().toString(36).slice(2, 9)}`;

let stream = null;
let sendTimer = null;
let sending = false;
const TARGET_WIDTH = 640;
let calibrationActive = false;
let calibrationStart = 0;
let calibrationDeadline = 0;
let blinkCount = 0;
let lastBlinkAt = 0;
let calibrationTimerId = null;
let behaviorTimer = null;
let mouseBuffer = [];
let keyBuffer = [];
let lastKeyTime = null;

thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

startBtn.addEventListener("click", () => {
  startCamera().catch((err) => handleError(err));
});

stopBtn.addEventListener("click", () => stopCamera());

async function startCamera() {
  if (stream) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 540, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    // Retry with a lower resolution fallback
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 360, facingMode: "user" },
        audio: false,
      });
    } catch (err2) {
      throw new Error(`Camera access error: ${err2.message || err2}`);
    }
  }

  video.srcObject = stream;
  await video.play();

  startBtn.disabled = true;
  stopBtn.disabled = false;
  updateStatus("Connecting...", "pending");

  // Start blink calibration (3 blinks within 10s)
  calibrationActive = true;
  calibrationStart = Date.now();
  calibrationDeadline = calibrationStart + 10000;
  blinkCount = 0;
  lastBlinkAt = 0;
  blinkOverlay.classList.add("active");
  blinkText.textContent = "Blink 3 times to calibrate";
  if (calibrationTimerId) {
    clearTimeout(calibrationTimerId);
  }
  calibrationTimerId = setTimeout(() => {
    if (calibrationActive) {
      finishCalibration(false);
    }
  }, 10000);

  sendTimer = setInterval(() => captureAndSend(), 240);
  startBehaviorCapture();
}

function stopCamera() {
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  stopBehaviorCapture();

  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateStatus("Detection stopped", "idle");
  verdictEl.textContent = "Idle";
  probText.textContent = "--%";
  bar.style.width = "0%";
}

function startBehaviorCapture() {
  mouseBuffer = [];
  keyBuffer = [];
  const mouseHandler = (ev) => {
    mouseBuffer.push({
      t: performance.now(),
      x: ev.clientX,
      y: ev.clientY,
      click: ev.type === "click",
      kind: ev.type,
    });
  };
  const keyHandler = (ev) => {
    const now = performance.now();
    const dt = lastKeyTime ? now - lastKeyTime : 0;
    keyBuffer.push({ dt });
    lastKeyTime = now;
  };
  document.addEventListener("mousemove", mouseHandler);
  document.addEventListener("click", mouseHandler);
  document.addEventListener("keydown", keyHandler);
  behaviorTimer = setInterval(() => sendBehavior(), 3500);
  // Store handlers for cleanup
  startBehaviorCapture.mouseHandler = mouseHandler;
  startBehaviorCapture.keyHandler = keyHandler;
}

function stopBehaviorCapture() {
  if (behaviorTimer) {
    clearInterval(behaviorTimer);
    behaviorTimer = null;
  }
  document.removeEventListener("mousemove", startBehaviorCapture.mouseHandler);
  document.removeEventListener("click", startBehaviorCapture.mouseHandler);
  document.removeEventListener("keydown", startBehaviorCapture.keyHandler);
  mouseBuffer = [];
  keyBuffer = [];
  lastKeyTime = null;
}

function updateStatus(text, state) {
  statusChip.textContent = text;
  statusChip.classList.remove("real", "spoof", "fake");

  if (state === "real") {
    statusChip.classList.add("real");
  } else if (state === "spoof" || state === "fake") {
    statusChip.classList.add("fake");
  }
}

function updateReadout(result) {
  const prob =
    result.smoothed_spoof_probability ??
    result.probabilities?.spoof ??
    0;
  const percentage = Math.round(prob * 100);

  probText.textContent = `${percentage}%`;
  bar.style.width = `${percentage}%`;
  bar.style.background = prob > 0.7 ? "linear-gradient(90deg,#ef476f,#f5c04f)" : "";

  const rawLabel = (result.label || "Unknown").toLowerCase();
  const displayLabel = rawLabel === "real" ? "Correct" : rawLabel === "fake" ? "Incorrect" : "Unknown";
  verdictEl.textContent = displayLabel;
  deviceEl.textContent = `Device: ${result.device ?? "unknown"}`;
  latencyEl.textContent = `${Math.round((result.inference_time ?? 0) * 1000)} ms`;
  lastUpdateEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;

  const state = rawLabel === "real" ? "real" : "fake";
  updateStatus(displayLabel ?? "Unknown", state);

  // Deepfake status (interval-based)
  if (result.deepfake) {
    const fakeProb = result.deepfake.probabilities?.fake ?? null;
    const realProb = result.deepfake.probabilities?.real ?? null;
    const flippedLabel =
      (result.deepfake.label ?? "").toLowerCase() === "fake" ? "Real" : "Fake";
    deepfakeLabel.textContent = flippedLabel;
    if (fakeProb !== null) {
      deepfakeProb.textContent = `Fake: ${(fakeProb * 100).toFixed(1)}% | Real: ${
        realProb !== null ? (realProb * 100).toFixed(1) : "--"
      }%`;
    } else {
      deepfakeProb.textContent = "--";
    }
    if (result.deepfake_last_checked) {
      const ageSec = Math.max(
        0,
        (Date.now() / 1000) - result.deepfake_last_checked
      ).toFixed(1);
      deepfakeAge.textContent = `Last: ${ageSec}s ago`;
    } else {
      deepfakeAge.textContent = "Last: --";
    }
  } else {
    deepfakeLabel.textContent = "Not checked";
    deepfakeProb.textContent = "--";
    deepfakeAge.textContent = "Last: --";
  }

  if (calibrationActive) {
    const now = Date.now();
    const remainingSec = Math.max(0, Math.ceil((calibrationDeadline - now) / 1000));
    const blinkDetected = !!result.blink?.blink_detected;
    if (blinkDetected && now - lastBlinkAt > 700) {
      blinkCount += 1;
      lastBlinkAt = now;
    }
    if (blinkCount >= 3 && now <= calibrationDeadline) {
      finishCalibration(true);
    } else if (now > calibrationDeadline) {
      finishCalibration(false);
    } else {
      blinkOverlay.classList.add("active");
      blinkText.textContent = `Blink 3 times (${blinkCount}/3) — ${remainingSec}s left`;
    }
  }
}

function renderBehavior(data) {
  if (!behaviorScoreEl) return;
  behaviorScoreEl.textContent = `${data.behavior_score ?? "--"}`;
  behaviorRiskEl.textContent = `Risk: ${data.risk_level ?? "--"}`;
  const now = new Date().toLocaleTimeString();
  behaviorUpdatedEl.textContent = `Last: ${now}`;
  if (data.flags && Object.keys(data.flags).length > 0) {
    behaviorFlagsEl.textContent = Object.keys(data.flags).join(", ");
  } else {
    behaviorFlagsEl.textContent = "None";
  }
}

async function fetchRecentBehavior() {
  if (!behaviorHistoryEl) return;
  try {
    const res = await fetch(`${API_BASE}/api/behavior/recent?session_id=${encodeURIComponent(SESSION_ID)}`);
    const data = await res.json();
    if (res.ok && data.samples) {
      const scores = data.samples.slice(0, 5).map((s) => s.score).join(" • ");
      behaviorHistoryEl.textContent = scores || "--";
    }
  } catch (err) {
    console.error("behavior recent error", err);
  }
}

function finishCalibration(success) {
  calibrationActive = false;
  if (calibrationTimerId) {
    clearTimeout(calibrationTimerId);
    calibrationTimerId = null;
  }
  if (success) {
    thresholdInput.value = 0.85;
  } else {
    thresholdInput.value = 0.4;
  }
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  blinkOverlay.classList.remove("active");
}

async function captureAndSend() {
  if (sending || !stream) return;
  sending = true;

  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.min(1, TARGET_WIDTH / vw);
  canvas.width = Math.floor(vw * scale);
  canvas.height = Math.floor(vh * scale);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.75);
  const payload = {
    image: dataUrl,
    threshold: Number(thresholdInput.value),
    session_id: SESSION_ID,
    deepfake: true,
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`API error ${res.status}`);
    }

    const result = await res.json();
    updateReadout(result);
  } catch (err) {
    handleError(err);
  } finally {
    sending = false;
  }
}

async function sendBehavior() {
  if (!BEHAVIOR_ENABLED) return;
  if (!mouseBuffer.length && !keyBuffer.length) return;
  const payload = {
    session_id: SESSION_ID,
    user_id: sessionStorage.getItem("fas_user") || null,
    mouse: mouseBuffer.map((m) => ({
      t: m.t,
      x: m.x,
      y: m.y,
      click: !!m.click,
      kind: m.kind || "move",
    })),
    keys: keyBuffer.map((k) => ({ dt: k.dt || 0 })),
  };
  mouseBuffer = [];
  keyBuffer = [];
  try {
    const res = await fetch(`${API_BASE}/api/behavior/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (res.ok) {
      renderBehavior(data);
      fetchRecentBehavior();
    }
  } catch (err) {
    console.error("behavior send error", err);
  }
}

function handleError(err) {
  console.error(err);
  const message = err?.message ? `Error: ${err.message}` : "Error — check camera/API";
  updateStatus(message, "fake");
  verdictEl.textContent = "No signal";
  probText.textContent = "--";
}

window.addEventListener("beforeunload", () => stopCamera());
