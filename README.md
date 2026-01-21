# Face Anti-Spoofing System (FAS)

This repository hosts a FastAPI-based face anti-spoofing service with optional OTP-backed auth and a web UI for live detection.

## Quick start
1) Create and activate a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2) Install dependencies: `pip install -r requirements.txt`
3) Launch the live service: `bash scripts/run_live.sh` (opens API on port 8000)
4) Visit the UI at `http://localhost:8000/ui/` or auth flow at `http://localhost:8000/auth`

## Notes
- MongoDB should be reachable at `mongodb://localhost:27017` by default; configure via `.env` if needed.
- OTP email delivery uses SMTP settings from `.env`; in dev mode it logs OTPs to the console.
- Model checkpoints and datasets are not in version control—place them under `pth/` or `checkpoints/` locally.
