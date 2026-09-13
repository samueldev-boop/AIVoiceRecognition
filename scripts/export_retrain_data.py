import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DATABASE_NAME", "altur_defense")]
col = db[os.getenv("COLLECTION_NAME", "calls")]

# 1. Simular insercion de una llamada calificada de produccion
col.update_one(
    {"call_id": "call_curada_demo_01"},
    {
        "$set": {
            "call_id": "call_curada_demo_01",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": {
                "is_synthetic": True,
                "confidence": 0.94,
                "probability_synthetic": 0.94,
            },
            "analysis": {
                "stage": "first_turn",
                "latency_s": 0.28,
                "audio_duration_s": 12.4,
                "disagreement": False,
                "turns": [
                    {"channel": 0, "start": 0.5, "end": 2.8},
                    {"channel": 1, "start": 3.0, "end": 5.5},
                ],
            },
            "status_for_training": "ready",
        }
    },
    upsert=True,
)

# 2. Criterio de curaduria: alta certeza, sin contradicciones y con habla valida
filtro_curados = {
    "status_for_training": "ready",
    "analysis.stage": {"$ne": "sin_habla"},
    "decision.confidence": {"$gte": 0.85},
    "analysis.disagreement": False,
}

total = col.count_documents({})
aptos = col.count_documents(filtro_curados)

print(f"Total llamadas en Atlas: {total}")
print(f"Llamadas curadas (Data Buena para reentrenar): {aptos}")

curados = list(col.find(filtro_curados, {"_id": 0, "call_id": 1, "decision": 1, "analysis.stage": 1}))
print("\nMuestras listas para el pipeline:")
print(json.dumps(curados, indent=2))
