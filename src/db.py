import logging

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from src.config import COLLECTION_NAME, DATABASE_NAME, MONGO_URI

logger = logging.getLogger("altur.db")

_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    """Devuelve el cliente de conexion a Atlas."""
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
    """Retorna la coleccion configurada en Atlas."""
    client = get_mongo_client()
    return client[DATABASE_NAME][COLLECTION_NAME]


def init_db_indexes() -> None:
    """Crea los indices unicos y compuestos en la coleccion."""
    try:
        col = get_calls_collection()
        col.create_index([("call_id", ASCENDING)], unique=True)
        col.create_index([("status_for_training", ASCENDING), ("timestamp", DESCENDING)])
    except PyMongoError as e:
        logger.warning("No se pudieron verificar los indices en Atlas: %s", e)


# Exportaciones directas requeridas por worker y scripts auxiliares
calls_collection = get_calls_collection()
