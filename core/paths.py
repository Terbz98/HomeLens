"""Folder locations used by the pipeline and the backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIXTURES_DIR = DATA_DIR / "fixtures"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Models trained on real data live in models/real, models trained on the
# synthetic fixtures live in models/demo. The two are never mixed.
REAL_MODEL_DIR = MODELS_DIR / "real"
DEMO_MODEL_DIR = MODELS_DIR / "demo"
