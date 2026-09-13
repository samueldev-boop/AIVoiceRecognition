import logging
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from src.config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME

logger = logging.getLogger("altur.db")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[DATABASE_NAME]
    calls_collection = db[COLLECTION_NAME]
except ConnectionFailure as e:
    logger.error(f"Error conectando a MongoDB Atlas: {e}")
    raise SystemExit("Fallo crítico en conexión con base de datos.")

def init_db_indexes():
    calls_collection.create_index([("call_id", ASCENDING)], unique=True)
    calls_collection.create_index([("status_for_training", ASCENDING)])
    calls_collection.create_index([("timestamp", ASCENDING)])
