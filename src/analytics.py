import os
import time
from datetime import datetime, timezone
from src.db import calls_collection

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def render_dashboard():
    while True:
        try:
            total_calls = calls_collection.count_documents({})
            synth_calls = calls_collection.count_documents({"decision.is_synthetic": True})
            human_calls = calls_collection.count_documents({"decision.is_synthetic": False})
            ready_to_train = calls_collection.count_documents({"status_for_training": "ready"})
            trained_calls = calls_collection.count_documents({"status_for_training": {"$regex": "^trained"}})

            recent_cursor = calls_collection.find().sort("timestamp", -1).limit(5)
            recent_calls = list(recent_cursor)

            pct_synth = (synth_calls / total_calls * 100) if total_calls > 0 else 0.0

            clear_screen()
            print("=" * 65)
            print(f"      ALTUR DEFENSE — RADAR DE MONITOREO BANCARIO")
            print(f"      Ultima actualizacion: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
            print("=" * 65)
            print(f"  Total de llamadas registradas : {total_calls}")
            print(f"  Ataques IA detectados         : {synth_calls} ({pct_synth:.1f}%)")
            print(f"  Llamadas humanas legtimas    : {human_calls} ({100 - pct_synth:.1f}%)")
            print(f"  Cola de Active Learning (ready): {ready_to_train}")
            print(f"  Llamadas ya reentrenadas       : {trained_calls}")
            print("-" * 65)
            print("  ULTIMAS LLAMADAS ENTRANTE:")
            print(f"  {'CALL ID':<20} | {'TIPO':<10} | {'CONFIANZA':<10} | {'ESTADO'}")
            print("-" * 65)

            for c in recent_calls:
                cid = c.get("call_id", "N/A")[:18]
                dec = c.get("decision", {}) or {}
                tipo = "DEEPFAKE" if dec.get("is_synthetic") else "HUMANO"
                conf = f"{dec.get('confidence', 0.0):.2f}"
                status = c.get("status_for_training", "N/A")
                print(f"  {cid:<20} | {tipo:<10} | {conf:<10} | {status}")

            print("=" * 65)
            print("  Presiona Ctrl+C para salir.")
            time.sleep(1.5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error consultando Atlas: {e}")
            time.sleep(3)

if __name__ == "__main__":
    render_dashboard()
