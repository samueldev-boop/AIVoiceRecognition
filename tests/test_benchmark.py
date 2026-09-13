"""El benchmark conserva el set robusto y usa el coste de FP al elegir umbral."""

import numpy as np
import pytest


def test_threshold_changes_with_false_positive_cost():
    pytest.importorskip("audiomentations")
    from scripts.benchmark_issue12 import choose_threshold, threshold_cost

    y = np.array([0, 0, 1, 1])
    scores = {"clean": np.array([0.1, 0.6, 0.55, 0.9])}
    inexpensive = choose_threshold(y, scores, fp_cost=0.1)
    expensive = choose_threshold(y, scores, fp_cost=5)
    assert inexpensive == 0.55
    assert expensive == 0.9
    assert threshold_cost(y, scores, expensive, 5) < threshold_cost(y, scores, 0.5, 5)


def test_tree_input_excludes_fragile_groups():
    pytest.importorskip("audiomentations")
    from app import features
    from app.interventions import feature_names
    from scripts.benchmark_issue12 import RobustFeatures

    names = feature_names()
    sample = {"call_features": np.arange(len(names), dtype=float)}
    selected = RobustFeatures().fit_transform([sample])[0].astype(int)
    assert {features.grupo_de(names[i]) for i in selected} == {
        "conducta",
        "prosodia",
        "razon_canal",
    }
