"""One scheduler invocation: export historical data and evaluate a new candidate."""

import logging

from src.config import Settings
from src.db import close_mongo_client
from src.export_training import export_active_learning_batch
from src.retrain_pipeline import execute_retraining
from src.storage import exclusive_lock


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level)
    try:
        with exclusive_lock(settings.training_dir / "weekly.lock"):
            path = export_active_learning_batch(settings=settings)
            execute_retraining(path, settings=settings)
    except Exception as exc:
        logging.error("event=weekly_failed error_type=%s", type(exc).__name__)
        raise SystemExit(1) from None
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
