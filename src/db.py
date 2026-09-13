import logging

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from src.config import COLLECTION_NAME, DATABASE_NAME, MONGO_URI

logger = logging.getLogger("altur.db")

_client: MongoClient | None = None
_indexes_initialized: bool = False


def get_mongo_client() -> MongoClient:
    """Devuelve un cliente reutilizable de MongoDB con inicializacion perezosa."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            _client.admin.command("ping")
            logger.info("Conexion exitosa a MongoDB Atlas.")
        except PyMongoError as e:
            logger.error("Error conectando a MongoDB Atlas: %s", e)
            raise SystemExit("Fallo critico en conexion con base de datos.") from e
    return _client


def get_calls_collection():
    """Entrega la coleccion e inicializa indices automaticamente una sola vez."""
    global _indexes_initialized
    client = get_mongo_client()
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    if not _indexes_initialized:
        try:
            collection.create_index([("call_id", ASCENDING)], unique=True)
            collection.create_index([("status_for_training", ASCENDING), ("timestamp", DESCENDING)])
            _indexes_initialized = True
        except PyMongoError as e:
            logger.warning("No se pudieron verificar los indices en Atlas: %s", e)

    return collection
