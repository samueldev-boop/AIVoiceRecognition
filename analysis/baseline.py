"""Entrada compatible al modelo sklearn #5; sin optimizadores ni AUC propios.

Uso: python -m analysis.baseline  (o python analysis/baseline.py)
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_issue5 import main  # noqa: E402

if __name__ == "__main__":
    main()
