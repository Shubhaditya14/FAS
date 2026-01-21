const viewLanding = document.getElementById("flow-landing");
const viewWorkspace = document.getElementById("flow-workspace");
const flowHint = document.getElementById("flow-hint");
const loginFormEl = document.getElementById("login-form");
const signupFormEl = document.getElementById("signup-form");
const goLoginBtn = document.getElementById("go-login");
const goSignupBtn = document.getElementById("go-signup");
const backHomeBtn = document.getElementById("back-home");
const switchLoginBtn = document.getElementById("switch-login");
const switchSignupBtn = document.getElementById("switch-signup");

const videoEl = document.getElementById("auth-video");
const canvasEl = document.getElementById("auth-canvas");
const ctx = canvasEl.getContext("2d");
const previewEl = document.getElementById("capture-preview");
const statusChip = document.querySelector("[data-auth-status]");

// Login fields
const loginUsernameEl = document.getElementById("login-username");
const loginPasswordEl = document.getElementById("login-password");
const loginOtpEl = document.getElementById("login-otp");
const captureLoginBtn = document.getElementById("capture-login");
const loginBtn = document.getElementById("login");
const loginMessageEl = document.getElementById("login-message");
const requestLoginOtpBtn = document.getElementById("request-login-otp");
const verifyLoginOtpBtn = document.getElementById("verify-login-otp");

// Signup fields
const signupUsernameEl = document.getElementById("signup-username");
const signupEmailEl = document.getElementById("signup-email");
const signupPhoneEl = document.getElementById("signup-phone");
const signupAadhaarEl = document.getElementById("signup-aadhaar");
const signupPasswordEl = document.getElementById("signup-password");
const signupConfirmEl = document.getElementById("signup-confirm");
const signupOtpEl = document.getElementById("signup-otp");
const captureSignupBtn = document.getElementById("capture-signup");
const signupBtn = document.getElementById("signup");
const signupMessageEl = document.getElementById("signup-message");
const requestSignupOtpBtn = document.getElementById("request-signup-otp");
const verifySignupOtpBtn = document.getElementById("verify-signup-otp");

// Readout
const sessionStatus = document.getElementById("session-status");
const sessionUser = document.getElementById("session-user");
const faceMatchEl = document.getElementById("face-match");
const faceDistEl = document.getElementById("face-distance");
const spoofEl = document.getElementById("spoof-info");
const deepfakeEl = document.getElementById("deepfake-info");

const API_BASE = window.APP_CONFIG?.apiBase || "";
const OTP_REQUIRED_LOGIN = window.APP_CONFIG?.otpRequiredLogin ?? false;
const VERIFY_INTERVAL_MS = 15000;

let stream = null;
let verifyTimer = null;
let token = null;
let activeForm = "login";
const capturedImages = {
  login: null,
  signup: null,
};
const otpState = {
  loginToken: null,
  signupToken: null,
  loginVerified: false,
  signupVerified: false,
};

async function parseJsonResponse(res) {
  const contentType = res.headers.get("content-type") || "";
  const text = await res.text();
  if (contentType.includes("application/json")) {
    try {
      return JSON.parse(text);
    } catch (err) {
      console.warn("Failed to parse JSON response", err);
    }
  }
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text || res.statusText };
  }
}

async function fetchJson(url, options, defaultError = "Request failed") {
  const res = await fetch(url, options);
  const data = await parseJsonResponse(res);
  if (!res.ok) {
    const detail = (data && data.detail) || data?.message || data?.error || res.statusText || defaultError;
    throw new Error(detail || defaultError);
  }
  return data;
}

function showView(view) {
  if (view === "landing") {
    viewLanding.classList.remove("hidden");
    viewWorkspace.classList.add("hidden");
    stopCamera();
  } else {
    viewLanding.classList.add("hidden");
    viewWorkspace.classList.remove("hidden");
  }
}

function showForm(form) {
  activeForm = form;
  if (form === "login") {
    loginFormEl.classList.remove("hidden");
    signupFormEl.classList.add("hidden");
    flowHint.textContent = "Capture a photo, then log in.";
    loginMessageEl.textContent = "Capture a photo, then log in.";
  } else {
    signupFormEl.classList.remove("hidden");
    loginFormEl.classList.add("hidden");
    flowHint.textContent = "Fill details, capture a photo, then sign up.";
    signupMessageEl.textContent = "Fill details, capture a photo, then sign up.";
  }
  if (previewEl) {
    previewEl.classList.add("hidden");
  }
  setStatus("Camera idle", null);
  otpState.loginVerified = false;
  otpState.signupVerified = false;
}

async function initCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 800, height: 450, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    throw new Error(`Camera access error: ${err.message || err}`);
  }
  videoEl.srcObject = stream;
  await videoEl.play();
  setStatus("Camera ready", "real");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  setStatus("Camera idle", null);
}

function setStatus(text, state) {
  statusChip.textContent = text;
  statusChip.classList.remove("real", "fake", "spoof");
  if (state === "real") statusChip.classList.add("real");
  if (state === "fake" || state === "spoof") statusChip.classList.add("fake");
}

function snap() {
  const w = videoEl.videoWidth || 640;
  const h = videoEl.videoHeight || 360;
  canvasEl.width = w;
  canvasEl.height = h;
  ctx.drawImage(videoEl, 0, 0, w, h);
  return canvasEl.toDataURL("image/jpeg", 0.75);
}

async function captureFrame(mode) {
  await initCamera();
  capturedImages[mode] = snap();
  if (previewEl) {
    previewEl.src = capturedImages[mode];
    previewEl.classList.remove("hidden");
  }
  const targetMsg = mode === "login" ? loginMessageEl : signupMessageEl;
  targetMsg.textContent = "Captured photo. Submit to continue.";
  setStatus("Photo captured", "real");
}

function validateSignupFields() {
  if (!signupUsernameEl.value.trim()) return "Username is required";
  if (!signupEmailEl.value.includes("@")) return "Valid email required";
  if (!signupPhoneEl.value.trim() || signupPhoneEl.value.replace(/\\D/g, "").length < 7) return "Valid phone required";
  if (!signupPasswordEl.value || signupPasswordEl.value.length < 8) return "Password must be at least 8 characters";
  if (signupPasswordEl.value !== signupConfirmEl.value) return "Passwords do not match";
  if (signupAadhaarEl.value && signupAadhaarEl.value.replace(/\\D/g, "").length !== 16) return "Aadhaar must be 16 digits";
  if (!capturedImages.signup) return "Capture a photo first";
  if (!otpState.signupVerified) return "Verify OTP before signup";
  return null;
}

function validateLoginFields() {
  if (!loginUsernameEl.value.trim()) return "Username is required";
  if (!loginPasswordEl.value) return "Password is required";
  if (!capturedImages.login) return "Capture a photo first";
  if (otpStateRequiredForLogin() && !otpState.loginVerified) return "Verify OTP before login";
  return null;
}

function otpStateRequiredForLogin() {
  return Boolean(OTP_REQUIRED_LOGIN);
}

async function requestOtp(mode) {
  const isLogin = mode === "login";
  const email = isLogin ? loginUsernameEl.value || signupEmailEl.value : signupEmailEl.value || signupUsernameEl.value;
  const aadhaar = isLogin ? null : signupAadhaarEl.value;
  if (!email || !email.includes("@")) throw new Error("Enter a valid email before requesting OTP");
  const payload = { email, aadhaar_id: aadhaar, username: isLogin ? loginUsernameEl.value : signupUsernameEl.value };
  const data = await fetchJson(`${API_BASE}/api/otp/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, "OTP request failed");
  if (isLogin) {
    otpState.loginToken = data.session_token;
    loginMessageEl.textContent = `OTP sent to ${data.email_masked || email}`;
  } else {
    otpState.signupToken = data.session_token;
    signupMessageEl.textContent = `OTP sent to ${data.email_masked || email}`;
  }
}

async function verifyOtp(mode) {
  const isLogin = mode === "login";
  const otp = isLogin ? loginOtpEl.value : signupOtpEl.value;
  const tokenVal = isLogin ? otpState.loginToken : otpState.signupToken;
  if (!otp || otp.length < 4) throw new Error("Enter OTP code");
  if (!tokenVal) throw new Error("Request OTP first");
  const payload = { session_token: tokenVal, otp };
  const data = await fetchJson(`${API_BASE}/api/otp/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, "OTP verification failed");
  if (isLogin) {
    otpState.loginVerified = true;
    loginMessageEl.textContent = "OTP verified. Proceed to login.";
  } else {
    otpState.signupVerified = true;
    signupMessageEl.textContent = "OTP verified. Proceed to signup.";
  }
}

async function signup() {
  const validationError = validateSignupFields();
  if (validationError) throw new Error(validationError);
  await initCamera();
  const payload = {
    username: signupUsernameEl.value,
    email: signupEmailEl.value,
    phone: signupPhoneEl.value,
    aadhaar_id: signupAadhaarEl.value,
    password: signupPasswordEl.value,
    otp_session: otpState.signupToken,
    image: capturedImages.signup,
  };
  const data = await fetchJson(`${API_BASE}/api/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, "Signup failed");
  handleAuthSuccess(signupUsernameEl.value, data.token, null, "signup");
}

async function login() {
  const validationError = validateLoginFields();
  if (validationError) throw new Error(validationError);
  await initCamera();
  const payload = {
    username: loginUsernameEl.value,
    password: loginPasswordEl.value,
    otp_session: otpState.loginToken,
    image: capturedImages.login,
  };
  const data = await fetchJson(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, "Login failed");
  handleAuthSuccess(loginUsernameEl.value, data.token, data.access, "login");
}

function handleAuthSuccess(username, tokenValue, access = null, source = "login") {
  token = tokenValue;
  sessionStorage.setItem("fas_token", tokenValue);
  sessionStorage.setItem("fas_user", username);
  sessionStatus.textContent = "Authenticated";
  sessionUser.textContent = username;

  if (access) {
    updateAccessReadout(access);
    setStatus(access.access_granted ? "Access granted" : "Access blocked", access.access_granted ? "real" : "fake");
    loginMessageEl.textContent = "Login verified. Redirecting to live detection…";
  } else {
    signupMessageEl.textContent = "Signup complete. Redirecting to live detection…";
    setStatus("Ready", "real");
  }

  startVerifyLoop();
  setTimeout(() => {
    window.location.href = "/ui/";
  }, 900);
}

async function verifyOnce() {
  if (!token) return;
  const image = snap();
  const payload = { token, image };
  const data = await fetchJson(`${API_BASE}/api/auth/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, "Verify failed");
  updateAccessReadout(data);
  setStatus(data.access_granted ? "Access granted" : "Access blocked", data.access_granted ? "real" : "fake");
}

function updateAccessReadout(data) {
  faceMatchEl.textContent = data.face_match ? "Match" : "Mismatch";
  faceDistEl.textContent = `Distance: ${data.face_distance.toFixed(3)}`;
  spoofEl.textContent = `Spoof: ${(data.spoof_probability * 100).toFixed(2)}% (${data.spoof_label})`;
  if (data.deepfake) {
    deepfakeEl.textContent = `Deepfake: ${data.deepfake.label} (fake ${(data.deepfake.probabilities.fake * 100).toFixed(1)}%)`;
  } else {
    deepfakeEl.textContent = "Deepfake: not checked";
  }
}

function startVerifyLoop() {
  if (verifyTimer) clearInterval(verifyTimer);
  verifyTimer = setInterval(() => {
    verifyOnce().catch((err) => {
      console.error(err);
      loginMessageEl.textContent = err.message || "Verification error";
    });
  }, VERIFY_INTERVAL_MS);
  verifyOnce().catch((err) => {
    console.error(err);
    loginMessageEl.textContent = err.message || "Verification error";
  });
}

function handleError(targetEl, err, fallback) {
  console.error(err);
  targetEl.textContent = err.message || fallback;
  if (err.message?.toLowerCase().includes("no face detected")) {
    targetEl.textContent = "No face detected. Center your face with good lighting and retry.";
  }
  setStatus(fallback, "fake");
}

goLoginBtn.addEventListener("click", () => {
  showView("workspace");
  showForm("login");
});

goSignupBtn.addEventListener("click", () => {
  showView("workspace");
  showForm("signup");
});

backHomeBtn.addEventListener("click", () => {
  showView("landing");
});

switchLoginBtn.addEventListener("click", () => showForm("login"));
switchSignupBtn.addEventListener("click", () => showForm("signup"));

captureLoginBtn.addEventListener("click", () => {
  captureFrame("login").catch((err) => handleError(loginMessageEl, err, "Capture error"));
});

captureSignupBtn.addEventListener("click", () => {
  captureFrame("signup").catch((err) => handleError(signupMessageEl, err, "Capture error"));
});

loginBtn.addEventListener("click", () => {
  login().catch((err) => handleError(loginMessageEl, err, "Login error"));
});

signupBtn.addEventListener("click", () => {
  signup().catch((err) => handleError(signupMessageEl, err, "Signup error"));
});

requestSignupOtpBtn.addEventListener("click", () => {
  requestOtp("signup").catch((err) => handleError(signupMessageEl, err, "OTP request error"));
});

verifySignupOtpBtn.addEventListener("click", () => {
  verifyOtp("signup").catch((err) => handleError(signupMessageEl, err, "OTP verify error"));
});

requestLoginOtpBtn.addEventListener("click", () => {
  requestOtp("login").catch((err) => handleError(loginMessageEl, err, "OTP request error"));
});

verifyLoginOtpBtn.addEventListener("click", () => {
  verifyOtp("login").catch((err) => handleError(loginMessageEl, err, "OTP verify error"));
});

// Init state
showView("landing");
showForm("login");
