"""Operational counts, without printing document contents or identifiers."""

import logging

from src.db import close_mongo_client, get_calls_collection


def render_dashboard() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        collection = get_calls_collection()
        for status in ("unlabeled", "ready", "excluded"):
            logging.info(
                "event=training_count status=%s count=%d",
                status,
                collection.count_documents({"status_for_training": status}),
            )
    finally:
        close_mongo_client()


if __name__ == "__main__":
    render_dashboard()
