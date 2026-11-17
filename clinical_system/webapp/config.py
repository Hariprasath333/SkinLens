"""App settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # clinical_system/webapp
SYSTEM_DIR = BASE_DIR.parent                         # clinical_system
PROJECT_DIR = SYSTEM_DIR.parent                      # SkinClass

DATA_DIR = Path(os.getenv("APP_DATA_DIR", SYSTEM_DIR / "instance"))
STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", DATA_DIR / "storage"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite by default. For PostgreSQL set e.g.
#   DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/skinlens
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")


def _secret_key() -> str:
    if os.getenv("APP_SECRET_KEY"):
        return os.environ["APP_SECRET_KEY"]
    f = DATA_DIR / "secret.key"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(48))
    return f.read_text().strip()


SECRET_KEY = _secret_key()
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", 8 * 3600))       # 8-hour shift
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"               # set 1 behind HTTPS
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 15))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "1") == "1"

# Model: "auto" = exported research model if present, else the original SkinLens model.
#        "skinlens" | "timm" | "dummy" (tiny random model, for tests only)
MODEL_KIND = os.getenv("MODEL_KIND", "auto")
EXPORTED_MODEL_DIR = Path(os.getenv("EXPORTED_MODEL_DIR", SYSTEM_DIR / "models" / "best"))
SKINLENS_CHECKPOINT = Path(os.getenv(
    "SKINLENS_CHECKPOINT", PROJECT_DIR / "backup" / "original_model" / "best_skinlens_hybrid.pt"))
SKINLENS_TEMPERATURE = Path(os.getenv(
    "SKINLENS_TEMPERATURE", PROJECT_DIR / "backup" / "original_model" / "temperature.json"))
SKINLENS_ARCHITECTURE = PROJECT_DIR / "backend" / "model" / "architecture.py"

# PNG figures shown on the Model Info page (e.g. ROC curves, confusion matrix from the paper)
PAPER_FIGURES_DIR = Path(os.getenv("PAPER_FIGURES_DIR", SYSTEM_DIR / "paper_figures"))

# prior correction (alpha, log_prior) and confidence-level thresholds
MODEL_SETTINGS = Path(os.getenv("MODEL_SETTINGS", BASE_DIR / "model_settings.json"))

LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", 50))      # more = steadier explanation, slower (~2 s each on CPU)
LIME_SEGMENTS = int(os.getenv("LIME_SEGMENTS", 40))
# confidence levels from model_settings.json: High > high_above, Medium >= medium_from, Low below
_levels = json.loads(MODEL_SETTINGS.read_text(encoding="utf-8")).get("confidence_levels", {})
CONFIDENCE_HIGH = float(_levels.get("high_above", 0.80))
CONFIDENCE_MEDIUM = float(_levels.get("medium_from", 0.50))

# Validation Lab (admin only): test data and where each run's raw results and report are written
VALIDATION_DATA_DIR = Path(os.getenv("VALIDATION_DATA_DIR", SYSTEM_DIR / "validation_data"))
VALIDATION_RESULTS_DIR = Path(os.getenv("VALIDATION_RESULTS_DIR", SYSTEM_DIR / "validation_results"))
