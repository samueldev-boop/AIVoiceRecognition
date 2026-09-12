"""Comparacion de limpieza y scores de capa con predicciones agrupadas del issue #5.

Uso: python -m analysis.ablation
Primero ejecutar python -m scripts.train_issue5. No selecciona features mirando val.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analysis/issue5"))
    args = parser.parse_args()
    report = json.loads((args.output / "metrics.json").read_text())
    inputs = joblib.load(args.output / "report_inputs.joblib")
    train = [r for r in inputs["records"] if r["meta"]["split"] == "train"]
    y = np.array([int(r["meta"]["label"] == "synthetic") for r in train])
    for budget, candidates in report["candidates"].items():
        print(f"\n{budget}: comparacion fuera de muestra")
        for name, result in candidates.items():
            print(
                f"  {name}: AUC={result['clean']['auc']:.4f}; "
                f"seleccion={result['selection_score']:.4f}"
            )
        chosen = report["budgets"][budget]["selected"]
        layers = inputs["evaluations"][budget][chosen]["layers"]
        for j, layer in enumerate(
            ("ganancia", "codec_bw", "silencio", "prosodia", "conducta", "razon_canal")
        ):
            print(f"  capa {layer}: AUC OOF={roc_auc_score(y, layers[:, j]):.4f}")


if __name__ == "__main__":
    main()
