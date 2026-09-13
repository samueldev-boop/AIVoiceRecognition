import json
from pathlib import Path
from src.db import calls_collection

OUTPUT_DATASET = Path("data/retrain_dataset.json")

def export_active_learning_batch():
    # 1. Filtrar casos de alta incertidumbre o listos para entrenamiento
    query = {
        "$or": [
            {"prediction.confidence": {"$gte": 0.40, "$lte": 0.60}},
            {"status_for_training": "ready"}
        ]
    }
    
    cursor = calls_collection.find(query, {"_id": 0})
    samples = list(cursor)
    
    if not samples:
        print("[ML Export] No hay llamadas pendientes que requieran reentrenamiento.")
        return

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, default=str)

    print("=" * 50)
    print(f"[ML Pipeline] Lote curado exportado exitosamente:")
    print(f" -> Total muestras seleccionadas: {len(samples)}")
    print(f" -> Guardado en: {OUTPUT_DATASET}")
    print("=" * 50)

if __name__ == "__main__":
    export_active_learning_batch()
