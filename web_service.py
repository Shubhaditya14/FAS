"""FastAPI service exposing face anti-spoofing inference for the JS frontend."""

import base64
import os
import random
import smtplib
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import torch
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModelForImageClassification
import face_recognition
import hashlib
import json
import uuid
from pymongo import MongoClient, ASCENDING

from inference import FASInference, TemporalSmoothing


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "web"
INDEX_FILE = FRONTEND_DIR / "index.html"

DEFAULT_CHECKPOINT = os.getenv("FAS_CHECKPOINT", "pth/AntiSpoofing_bin_128.pth")
REQUESTED_DEVICE = os.getenv("FAS_DEVICE")
PROBABILITY_MODE = os.getenv("FAS_PROB_MODE", "spoof").lower()
DEEPFAKE_MODEL_ID = os.getenv("DEEPFAKE_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model")
DEEPFAKE_INTERVAL = float(os.getenv("DEEPFAKE_INTERVAL", "15"))
ENABLE_DEEPFAKE = os.getenv("ENABLE_DEEPFAKE", "1") != "0"
BLINK_TIMEOUT = float(os.getenv("BLINK_TIMEOUT", "8"))
BLINK_DIFF_THRESHOLD = float(os.getenv("BLINK_DIFF_THRESHOLD", "8"))
USER_DB_PATH = BASE_DIR / "data" / "users.json"
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))
LOGIN_SPOOF_THRESHOLD = max(
    0.0, min(1.0, float(os.getenv("LOGIN_SPOOF_THRESHOLD", "0.2")))
)
SESSION_TTL = float(os.getenv("SESSION_TTL", "7200"))  # seconds
OTP_EXPIRY_SECONDS = int(os.getenv("OTP_EXPIRY_SECONDS", "300"))
OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))
OTP_REQUIRED_FOR_LOGIN = os.getenv("OTP_REQUIRED_FOR_LOGIN", "0") == "1"
OTP_DEV_MODE = os.getenv("OTP_DEV_MODE", "1") == "1"
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("FROM_EMAIL", SMTP_USER or "noreply@example.com")
BEHAVIOR_ENABLED = os.getenv("BEHAVIOR_ENABLED", "1") != "0"
BEHAVIOR_SCORE_THRESHOLD = float(os.getenv("BEHAVIOR_SCORE_THRESHOLD", "40"))
BEHAVIOR_SAMPLE_WINDOW_MS = int(os.getenv("BEHAVIOR_SAMPLE_WINDOW_MS", "4000"))
BEHAVIOR_TTL_SECONDS = int(os.getenv("BEHAVIOR_TTL_SECONDS", "3600"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "face")


def resolve_device(requested: Optional[str]) -> str:
    """Pick the best available device."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_inference() -> FASInference:
    """Create a global inference object."""
    checkpoint_path = Path(DEFAULT_CHECKPOINT)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Set FAS_CHECKPOINT to the desired model path."
        )

    device = resolve_device(REQUESTED_DEVICE)
    return FASInference(
        checkpoint_paths=str(checkpoint_path),
        device=device,
        temporal_smoothing=False,
        probability_mode=PROBABILITY_MODE,
    )


inference = load_inference()
hf_processor: Optional[AutoImageProcessor] = None
hf_model: Optional[AutoModelForImageClassification] = None
hf_device: Optional[torch.device] = None
last_deepfake_check: Dict[str, Optional[float]] = {"timestamp": None}
deepfake_lock = Lock()

prev_gray = None
last_blink_time = time.time()
blink_lock = Lock()

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
users_col = db["users"]
sessions_col = db["sessions"]
otp_col = db["otp_sessions"]
behavior_col = db["behavior_samples"]
users_col.create_index([("username", ASCENDING)], unique=True)
users_col.create_index([("email", ASCENDING)], unique=True)
sessions_col.create_index([("expires", ASCENDING)], expireAfterSeconds=0)
otp_col.create_index([("session_token", ASCENDING)], unique=True)
otp_col.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
behavior_col.create_index([("session_id", ASCENDING)])
behavior_col.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)


def load_deepfake_model() -> Tuple[AutoImageProcessor, AutoModelForImageClassification, torch.device]:
    """Load the Hugging Face deepfake detector."""
    global hf_processor, hf_model, hf_device
    if hf_processor is not None and hf_model is not None and hf_device is not None:
        return hf_processor, hf_model, hf_device

    device = inference.device
    processor = AutoImageProcessor.from_pretrained(DEEPFAKE_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(DEEPFAKE_MODEL_ID)
    model.to(device)
    model.eval()

    hf_processor = processor
    hf_model = model
    hf_device = device
    return processor, model, device


def run_deepfake(image: Image.Image) -> Dict[str, object]:
    """Run deepfake detector and return structured result."""
    processor, model, device = load_deepfake_model()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    id2label = model.config.id2label
    label_probs = {id2label[i].lower(): float(probs[i].item()) for i in range(len(probs))}
    fake_prob = label_probs.get("fake") or label_probs.get("deepfake") or max(label_probs.values())
    real_prob = label_probs.get("real") or (1.0 - fake_prob)
    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = max(0.0, min(1.0, real_prob))
    label = "Fake" if fake_prob >= real_prob else "Real"
    return {
        "label": label,
        "probabilities": {"fake": fake_prob, "real": real_prob},
        "model_id": DEEPFAKE_MODEL_ID,
    }
app = FastAPI(title="FAS Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


class FramePayload(BaseModel):
    """Request payload for a single frame."""

    image: str = Field(..., description="Base64 data URL for the frame")
    threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Spoof probability threshold"
    )
    session_id: Optional[str] = Field(
        None,
        description="Client session id for temporal smoothing",
    )
    deepfake: bool = Field(
        True,
        description="Whether to run deepfake detector (interval-based)",
    )


class SignupPayload(BaseModel):
    username: str
    password: str
    email: str
    phone: str
    aadhaar_id: Optional[str] = None
    otp_session: Optional[str] = None
    image: str  # data URL


class LoginPayload(BaseModel):
    username: str
    password: str
    image: str  # data URL
    otp_session: Optional[str] = None


class OTPInitPayload(BaseModel):
    username: Optional[str] = None
    email: str
    aadhaar_id: Optional[str] = None


class OTPVerifyPayload(BaseModel):
    session_token: str
    otp: str = Field(..., pattern=r"^\d{4,8}$")


class BehaviorEvent(BaseModel):
    t: float
    x: Optional[float] = None
    y: Optional[float] = None
    click: Optional[bool] = False
    kind: Optional[str] = "move"


class KeyTiming(BaseModel):
    dt: float  # milliseconds between keys or dwell


class BehaviorRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    mouse: List[BehaviorEvent] = Field(default_factory=list)
    keys: List[KeyTiming] = Field(default_factory=list)



class VerifyPayload(BaseModel):
    token: str
    image: str  # data URL


SMOOTHERS: Dict[str, TemporalSmoothing] = {}
SMOOTHER_LOCK = Lock()


def get_smoother(session_id: str) -> TemporalSmoothing:
    """Get or create a smoother for the given session."""
    with SMOOTHER_LOCK:
        if session_id not in SMOOTHERS:
            SMOOTHERS[session_id] = TemporalSmoothing(window_size=5, alpha=0.6)
        return SMOOTHERS[session_id]


def decode_image(data_url: str) -> Image.Image:
    """Decode a base64 data URL into a PIL image."""
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        image_bytes = base64.b64decode(encoded)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc


def detect_blink(gray_frame: np.ndarray) -> Dict[str, object]:
    """Simple blink detection using frame difference spikes."""
    global prev_gray, last_blink_time
    now = time.time()
    blink_detected = False
    with blink_lock:
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_frame)
            mean_diff = float(diff.mean())
            if mean_diff > BLINK_DIFF_THRESHOLD:
                last_blink_time = now
                blink_detected = True
        prev_gray = gray_frame

    seconds_since_blink = now - last_blink_time
    prompt = seconds_since_blink > BLINK_TIMEOUT
    return {
        "blink_detected": blink_detected,
        "seconds_since_blink": seconds_since_blink,
        "prompt": prompt,
        "message": "Blink 3 times now" if prompt else "",
    }


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def generate_otp() -> str:
    """Generate a numeric OTP of configured length."""
    return "".join(str(random.randint(0, 9)) for _ in range(OTP_LENGTH))


def hash_otp(otp: str, salt: Optional[str] = None) -> str:
    salt = salt or uuid.uuid4().hex
    return hashlib.sha256(f"{salt}:{otp}".encode("utf-8")).hexdigest(), salt


def verify_hashed_otp(otp: str, otp_hash: str, salt: str) -> bool:
    return otp_hash == hashlib.sha256(f"{salt}:{otp}".encode("utf-8")).hexdigest()


def send_otp_email(email: str, otp: str, user_name: Optional[str] = None) -> None:
    """Send or log OTP email; dev mode prints to console."""
    subject = "Your login OTP"
    body = f"Hello {user_name or ''}, your OTP is {otp}. It expires in {OTP_EXPIRY_SECONDS} seconds."
    if OTP_DEV_MODE or not SMTP_HOST or not SMTP_USER:
        print(f"[DEV OTP] Email={email} OTP={otp}")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = email
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def create_otp_session(email: str, aadhaar: Optional[str], username: Optional[str]) -> str:
    """Create an OTP session and return the session token."""
    otp = generate_otp()
    otp_hash, salt = hash_otp(otp)
    session_token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(seconds=OTP_EXPIRY_SECONDS)
    otp_col.delete_many({"email": email})
    otp_col.insert_one(
        {
            "session_token": session_token,
            "otp_hash": otp_hash,
            "salt": salt,
            "email": email,
            "aadhaar": aadhaar,
            "username": username,
            "expires_at": expires_at,
            "verified": False,
        }
    )
    send_otp_email(email, otp, username)
    return session_token


def verify_otp_session(session_token: str, otp: str) -> Dict[str, object]:
    session = otp_col.find_one({"session_token": session_token})
    if not session:
        raise HTTPException(status_code=400, detail="OTP session not found")
    if session.get("verified"):
        return session
    if datetime.utcnow() > session["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired")
    if not verify_hashed_otp(otp, session["otp_hash"], session["salt"]):
        raise HTTPException(status_code=401, detail="Invalid OTP")
    otp_col.update_one({"session_token": session_token}, {"$set": {"verified": True}})
    session["verified"] = True
    return session


def get_face_encoding(image: Image.Image) -> np.ndarray:
    """Return a single face encoding with multiple detection fallbacks."""
    arr = np.array(image)

    def haar_locations(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return []
        cascade = cv2.CascadeClassifier(str(cascade_path))
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    attempts = []
    attempts.append(
        (
            "hog-2",
            arr,
            face_recognition.face_locations(arr, number_of_times_to_upsample=2, model="hog"),
        )
    )
    attempts.append(
        (
            "hog-3",
            arr,
            face_recognition.face_locations(arr, number_of_times_to_upsample=3, model="hog"),
        )
    )

    # Resize to a wider frame for small inputs
    if arr.shape[1] < 720:
        scale = 720 / float(arr.shape[1])
        new_h = int(arr.shape[0] * scale)
        wider = cv2.resize(arr, (720, new_h), interpolation=cv2.INTER_LINEAR)
        attempts.append(
            (
                "hog-wide",
                wider,
                face_recognition.face_locations(
                    wider, number_of_times_to_upsample=2, model="hog"
                ),
            )
        )

    # CNN detector (skip quietly if unavailable)
    try:
        attempts.append(
            (
                "cnn",
                arr,
                face_recognition.face_locations(arr, number_of_times_to_upsample=1, model="cnn"),
            )
        )
    except Exception:
        pass

    # OpenCV Haar cascade fallback
    haar_locs = haar_locations(arr)
    attempts.append(("haar", arr, haar_locs))

    for label, frame, locations in attempts:
        if not locations:
            continue
        encodings = face_recognition.face_encodings(
            frame,
            known_face_locations=locations,
            num_jitters=1,
            model="small",
        )
        if encodings:
            return encodings[0]

    raise HTTPException(
        status_code=400,
        detail="No face detected for enrollment/verification. Center your face, good lighting, and retry.",
    )


def validate_signup_payload(payload: SignupPayload) -> None:
    if not payload.username.strip():
        raise HTTPException(status_code=400, detail="Username is required")
    if "@" not in payload.email or "." not in payload.email:
        raise HTTPException(status_code=400, detail="Valid email required")
    digits = "".join(ch for ch in payload.phone if ch.isdigit())
    if len(digits) < 7:
        raise HTTPException(status_code=400, detail="Valid phone required")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if payload.aadhaar_id:
        aadhaar_digits = "".join(ch for ch in payload.aadhaar_id if ch.isdigit())
        if len(aadhaar_digits) != 16:
            raise HTTPException(status_code=400, detail="Aadhaar ID must be 16 digits")


def evaluate_basic_match(image: Image.Image, encoding_ref: np.ndarray) -> Dict[str, object]:
    """Run a lightweight face-embedding comparison plus a simple spoof score."""
    encoding_live = get_face_encoding(image)
    distances = face_recognition.face_distance([encoding_ref], encoding_live)
    distance = float(distances[0])
    match = bool(distance <= FACE_MATCH_THRESHOLD)

    # Basic spoof check using the existing FAS model
    fas_result = inference.predict_image(image, threshold=0.5)
    spoof_prob = fas_result["probabilities"]["spoof"]
    spoof_label = fas_result["label"]
    spoof_flag = spoof_prob >= 0.5

    access_granted = match and not spoof_flag

    return {
        "face_match": match,
        "face_distance": distance,
        "spoof_probability": spoof_prob,
        "spoof_label": spoof_label,
        "deepfake": None,
        "access_granted": access_granted,
    }


def score_behavior(mouse: List[BehaviorEvent], keys: List[KeyTiming]) -> Dict[str, object]:
    """Basic rule-based behavior score using mouse/key dynamics."""
    if not mouse and not keys:
        return {
            "behavior_score": 50,
            "risk_level": "MEDIUM",
            "flags": {"insufficient_data": True},
            "explainability": {"overall": "No behavioral data"},
            "confidence": 0.1,
        }

    # Mouse features
    speeds: List[float] = []
    jitter: List[float] = []
    clicks = 0
    last = None
    for ev in mouse:
        if ev.click:
            clicks += 1
        if ev.x is None or ev.y is None:
            continue
        if last:
            dt = max(1e-3, (ev.t - last["t"]) / 1000.0)
            dist = ((ev.x - last["x"]) ** 2 + (ev.y - last["y"]) ** 2) ** 0.5
            speeds.append(dist / dt)
            jitter.append(abs(dist - last["dist"]) if "dist" in last else 0.0)
            last = {"x": ev.x, "y": ev.y, "t": ev.t, "dist": dist}
        else:
            last = {"x": ev.x, "y": ev.y, "t": ev.t, "dist": 0.0}

    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    speed_std = float(np.std(speeds)) if speeds else 0.0
    jitter_mean = float(np.mean(jitter)) if jitter else 0.0

    # Key features
    key_intervals = [k.dt for k in keys if k.dt > 0]
    key_var = float(np.var(key_intervals)) if key_intervals else 0.0
    key_entropy = float(np.log(len(key_intervals) + 1.0) / np.log(50.0)) if key_intervals else 0.0

    score = 80.0
    flags: Dict[str, bool] = {}
    explain: Dict[str, str] = {}

    if avg_speed > 1500:
        score -= 20
        flags["mouse_too_fast"] = True
        explain["mouse_speed"] = f"High speed {avg_speed:.1f}px/s"
    elif avg_speed < 50 and mouse:
        score -= 10
        flags["mouse_too_slow"] = True
        explain["mouse_speed"] = "Very slow movement"
    else:
        explain["mouse_speed"] = "Normal movement"

    if jitter_mean < 0.5 and mouse:
        score -= 15
        flags["mouse_bot_like"] = True
        explain["mouse_jitter"] = "Low jitter (regular path)"
    else:
        explain["mouse_jitter"] = "Natural jitter"

    if speed_std < 20 and mouse:
        score -= 10
        flags["mouse_regular_speed"] = True
        explain["mouse_variance"] = "Very consistent speed"
    else:
        explain["mouse_variance"] = "Speed variance ok"

    if key_intervals:
        if key_var < 100:
            score -= 10
            flags["typing_regular"] = True
            explain["typing"] = "Very regular typing cadence"
        else:
            explain["typing"] = "Typing cadence varied"
        if key_entropy < 0.3:
            score -= 5
            flags["typing_low_entropy"] = True
    else:
        explain["typing"] = "No typing data"

    score = max(0, min(100, score))
    risk = "LOW" if score >= BEHAVIOR_SCORE_THRESHOLD + 20 else "MEDIUM"
    if score < BEHAVIOR_SCORE_THRESHOLD:
        risk = "HIGH"

    return {
        "behavior_score": int(score),
        "risk_level": risk,
        "flags": flags,
        "explainability": explain,
        "confidence": 0.5 + (0.5 if mouse or key_intervals else 0),
        "features": {
            "avg_speed": avg_speed,
            "speed_std": speed_std,
            "jitter_mean": jitter_mean,
            "key_var": key_var,
            "key_entropy": key_entropy,
            "clicks": clicks,
            "mouse_events": len(mouse),
            "key_events": len(key_intervals),
        },
    }
def create_session(username: str) -> str:
    token = uuid.uuid4().hex
    sessions_col.insert_one(
        {
            "token": token,
            "username": username,
            "expires": time.time() + SESSION_TTL,
        }
    )
    return token


def validate_session(token: str) -> str:
    data = sessions_col.find_one({"token": token})
    if not data or data["expires"] < time.time():
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return data["username"]


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    checkpoint = DEFAULT_CHECKPOINT
    device = str(inference.device)
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "device": device,
        "probability_mode": PROBABILITY_MODE,
    }


@app.get("/")
async def serve_index():
    """Serve the frontend if present."""
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "Frontend not found. Check /ui once assets are built."}


@app.get("/auth")
async def serve_auth():
    """Serve the auth page."""
    auth_file = FRONTEND_DIR / "auth.html"
    if auth_file.exists():
        return FileResponse(auth_file)
    return {"message": "Auth frontend not found."}


@app.post("/api/predict")
async def predict(payload: FramePayload) -> Dict:
    """Score a single video frame."""
    image = decode_image(payload.image)
    raw_result = inference.predict_image(image, threshold=payload.threshold)

    label = raw_result["label"]
    is_real = raw_result["is_real"]
    spoof_prob = raw_result["probabilities"]["spoof"]

    smoothed_spoof = None

    if payload.session_id:
        smoother = get_smoother(payload.session_id)
        smoothed_spoof = smoother.update(spoof_prob)
        is_real = smoothed_spoof < payload.threshold
        label = "Real" if is_real else "Fake"

    # Deepfake detection (interval-based)
    deepfake_result = None
    now = time.time()
    if ENABLE_DEEPFAKE and payload.deepfake:
        with deepfake_lock:
            last_ts = last_deepfake_check.get("timestamp")
            if last_ts is None or (now - last_ts) >= DEEPFAKE_INTERVAL:
                deepfake_result = run_deepfake(image)
                last_deepfake_check["timestamp"] = now
                last_deepfake_check["result"] = deepfake_result
            else:
                deepfake_result = last_deepfake_check.get("result")

    # Blink detection (per-frame difference)
    gray_small = np.array(image.resize((160, 120)).convert("L"))
    blink_info = detect_blink(gray_small)

    response = {
        "label": label,
        "is_real": is_real,
        "device": str(inference.device),
        "inference_time": raw_result["inference_time"],
        "probabilities": raw_result["probabilities"],
        "smoothed_spoof_probability": smoothed_spoof,
        "deepfake": deepfake_result,
        "deepfake_last_checked": last_deepfake_check.get("timestamp"),
        "blink": blink_info,
    }

    return response


@app.post("/api/auth/signup")
async def signup(payload: SignupPayload) -> Dict[str, object]:
    """Register a user with a reference face encoding."""
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()
    phone = payload.phone.strip()
    validate_signup_payload(payload)
    existing = users_col.find_one({"username": username})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    existing_email = users_col.find_one({"email": email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    if not payload.image:
        raise HTTPException(status_code=400, detail="Capture a photo before signup")
    if not payload.otp_session:
        raise HTTPException(status_code=400, detail="Verify OTP before signup")
    otp_session = otp_col.find_one({"session_token": payload.otp_session, "verified": True})
    if not otp_session or otp_session.get("email") != email:
        raise HTTPException(status_code=400, detail="OTP not verified for this email")
    if payload.aadhaar_id:
        aadhaar_digits = "".join(ch for ch in payload.aadhaar_id if ch.isdigit())
        if otp_session.get("aadhaar") and otp_session["aadhaar"] != aadhaar_digits:
            raise HTTPException(status_code=400, detail="OTP session Aadhaar mismatch")

    image = decode_image(payload.image)
    encoding = get_face_encoding(image)
    users_col.insert_one(
        {
            "username": username,
            "password_hash": hash_password(payload.password),
            "encoding": encoding.tolist(),
            "email": email,
            "phone": phone,
            "aadhaar_id": payload.aadhaar_id,
            "created_at": time.time(),
        }
    )
    otp_col.delete_many({"session_token": payload.otp_session})
    token = create_session(username)
    return {"status": "ok", "token": token}


@app.post("/api/auth/login")
async def login(payload: LoginPayload) -> Dict[str, object]:
    """Login with password plus a basic face match check."""
    username = payload.username.strip().lower()
    user = users_col.find_one({"username": username})
    if not user or user["password_hash"] != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not payload.image:
        raise HTTPException(status_code=400, detail="Capture a photo before login")

    image = decode_image(payload.image)
    encoding_ref: Optional[np.ndarray] = None
    if user.get("encoding"):
        encoding_ref = np.array(user["encoding"])
    else:
        encoding_ref = get_face_encoding(image)
        users_col.update_one({"username": username}, {"$set": {"encoding": encoding_ref.tolist()}})

    access = evaluate_basic_match(image, encoding_ref)
    if not access["access_granted"]:
        raise HTTPException(status_code=401, detail="Face mismatch or spoof detected")

    if OTP_REQUIRED_FOR_LOGIN:
        if not payload.otp_session:
            raise HTTPException(status_code=400, detail="OTP required for login")
        otp_session = otp_col.find_one({"session_token": payload.otp_session, "verified": True})
        if not otp_session or otp_session.get("email") != user.get("email"):
            raise HTTPException(status_code=401, detail="OTP verification failed")
        otp_col.delete_many({"session_token": payload.otp_session})

    token = create_session(username)
    return {
        "status": "ok",
        "token": token,
        "access": access,
    }


@app.post("/api/auth/verify")
async def verify(payload: VerifyPayload) -> Dict[str, object]:
    """Verify token plus a basic face match check."""
    username = validate_session(payload.token)
    user = users_col.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User missing")

    if not payload.image:
        raise HTTPException(status_code=400, detail="Capture a photo before verification")

    image = decode_image(payload.image)
    encoding_ref: Optional[np.ndarray] = None
    if user.get("encoding"):
        encoding_ref = np.array(user["encoding"])
    else:
        encoding_ref = get_face_encoding(image)
        users_col.update_one({"username": username}, {"$set": {"encoding": encoding_ref.tolist()}})

    access = evaluate_basic_match(image, encoding_ref)
    if not access["access_granted"]:
        raise HTTPException(status_code=401, detail="Face mismatch or spoof detected")

    return {"status": "ok", "user": username, **access}


@app.post("/api/otp/init")
async def otp_init(payload: OTPInitPayload) -> Dict[str, object]:
    """Start an OTP session for signup/login."""
    email = payload.email.strip().lower()
    aadhaar_digits = "".join(ch for ch in (payload.aadhaar_id or "") if ch.isdigit()) or None
    if aadhaar_digits and len(aadhaar_digits) != 16:
        raise HTTPException(status_code=400, detail="Aadhaar ID must be 16 digits")
    session_token = create_otp_session(email=email, aadhaar=aadhaar_digits, username=payload.username)
    masked = email[:2] + "***" + email[email.find("@") :]
    return {
        "status": "ok",
        "session_token": session_token,
        "email_masked": masked,
        "expires_in": OTP_EXPIRY_SECONDS,
    }


@app.post("/api/otp/verify")
async def otp_verify(payload: OTPVerifyPayload) -> Dict[str, object]:
    """Verify an OTP session."""
    session = verify_otp_session(payload.session_token, payload.otp)
    return {
        "status": "ok",
        "email": session.get("email"),
        "aadhaar": session.get("aadhaar"),
        "session_token": payload.session_token,
    }


@app.post("/api/behavior/score")
async def behavior_score(payload: BehaviorRequest) -> Dict[str, object]:
    """Ingest behavior events, score them, and persist a sample."""
    if not BEHAVIOR_ENABLED:
        return {
            "behavior_score": 70,
            "risk_level": "MEDIUM",
            "flags": {"disabled": True},
            "explainability": {"overall": "Behavior module disabled"},
            "confidence": 0.0,
        }
    result = score_behavior(payload.mouse, payload.keys)
    doc = {
        "session_id": payload.session_id,
        "user_id": payload.user_id,
        "score": result["behavior_score"],
        "risk_level": result["risk_level"],
        "flags": result.get("flags", {}),
        "features": result.get("features", {}),
        "timestamp": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(seconds=BEHAVIOR_TTL_SECONDS),
    }
    behavior_col.insert_one(doc)
    return result


@app.get("/api/behavior/recent")
async def behavior_recent(session_id: str) -> Dict[str, object]:
    """Return recent behavior samples for a session."""
    if not BEHAVIOR_ENABLED:
        return {"samples": []}
    cursor = behavior_col.find({"session_id": session_id}).sort("timestamp", -1).limit(30)
    samples = []
    for item in cursor:
        samples.append(
            {
                "score": item.get("score"),
                "risk_level": item.get("risk_level"),
                "flags": item.get("flags"),
                "features": item.get("features"),
                "timestamp": item.get("timestamp"),
            }
        )
    return {"samples": samples}
