import json
import logging
import signal
import time

from src.config import AUDIT_LOG_PATH
from src.db import calls_collection, init_db_indexes
from src.schemas import CallAuditRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("altur.worker")

running = True


def signal_handler(signum, frame):
    global running
    logger.info("Deteniendo worker de forma segura...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def process_record(raw_line: str):
    clean = raw_line.strip()
    if not clean:
        return
    try:
        data = json.loads(clean)
        record = CallAuditRecord(**data)
        payload = {k: v for k, v in record.model_dump().items() if v is not None}

        calls_collection.update_one(
            {"call_id": record.call_id},
            {"$set": payload},
            upsert=True,
        )
        tag = "Decisión" if record.decision and not record.analysis else "Análisis/Unificado"
        logger.info(f"Ingestado {tag} | call_id: {record.call_id}")
    except Exception as err:
        logger.warning(f"Descarte de registro invalido: {err}")


def start_worker():
    init_db_indexes()
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.touch()

    logger.info(f"Escuchando eventos en: {AUDIT_LOG_PATH}")

    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        while running:
            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue
            process_record(line)

    logger.info("Worker finalizado.")


if __name__ == "__main__":
    start_worker()
