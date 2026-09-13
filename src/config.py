import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", DATA_DIR / "audit.jsonl"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "altur_defense")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "calls")
