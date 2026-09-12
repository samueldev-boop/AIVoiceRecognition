"""Invariantes de evaluacion: llamadas/voz, fuga en fusion y calibracion, artefacto."""

import joblib
import numpy as np
import pytest
import sklearn

from app import config, features
from app.interventions import EXTRACTOR_VERSION, extract_sample, extract_turns, feature_names
from app.learning import LayeredClassifier, QuantileClipper, grouped_splits, pipeline
from app.model import BUNDLE_VERSION, Modelo, cargar, confidence_policy
from scripts.issue5_data import assign_groups, read_manifest, speaker_mapping
from scripts.train_issue5 import fit_calibrated


@pytest.fixture(scope="module")
def records():
    rng = np.random.default_rng(57)
    result = []
    for i in range(48):
        label = (i // 2) % 2
        result.append(
            {
                "call_id": f"test_{i}",
                "group": f"voice_{i // 2}",
                "split": "train",
                "label": label,
                "budget": "full",
                "call_features": rng.normal(label, 0.8, len(feature_names())),
                "turn_features": rng.normal(label, 0.8, (3, len(feature_names()))),
            }
        )
    return result


@pytest.fixture(scope="module")
def calibrated(records):
    return fit_calibrated(records, {"C": 0.33, "quantile": 0.005})


def test_split_never_separates_calls_of_same_voice(records):
    y = np.array([r["label"] for r in records])
    groups = np.array([r["group"] for r in records])
    tests = []
    for train, test in grouped_splits(y, groups, 5):
        assert not set(groups[train]) & set(groups[test])
        tests.extend(test)
    assert sorted(tests) == list(range(len(records)))


def test_calibration_refits_whole_hierarchy_on_disjoint_groups(records, monkeypatch):
    fits = []
    original_fit = LayeredClassifier.fit

    def spy(self, X, y):
        fits.append(set(r["group"] for r in X))
        return original_fit(self, X, y)

    monkeypatch.setattr(LayeredClassifier, "fit", spy)
    model = fit_calibrated(records, {"C": 0.10, "quantile": 0.005})
    all_groups = {r["group"] for r in records}
    assert len(fits) == 4  # 3 calibracion + refit final
    assert fits.count(all_groups) == 1
    partial_fits = [trained for trained in fits if trained != all_groups]
    for trained, (train, test) in zip(partial_fits, model.cv, strict=True):
        assert trained == {records[i]["group"] for i in train}
        assert not trained & {records[i]["group"] for i in test}


def test_val_cannot_enter_any_fit(records):
    corrupted = [{**r, "split": "val" if i == 0 else "train"} for i, r in enumerate(records)]
    with pytest.raises(ValueError, match="val no puede"):
        LayeredClassifier().fit(corrupted, [r["label"] for r in records])


def test_imputation_scaling_and_outlier_limits_do_not_read_test():
    X = np.array([[0.0, np.nan], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    model = pipeline(quantile=0.1).fit(X, [0, 0, 0, 1, 1, 1])
    median = model["imputer"].statistics_.copy()
    limits = model["clipper"].upper_.copy()
    mean = model["scaler"].mean_.copy()
    p = model.predict_proba([[1e12, np.inf], [-1e12, np.nan]])
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(model["imputer"].statistics_, median)
    np.testing.assert_array_equal(model["clipper"].upper_, limits)
    np.testing.assert_array_equal(model["scaler"].mean_, mean)
    assert median[1] == 30.0
    clipper = QuantileClipper(0.1).fit(np.arange(10).reshape(-1, 1))
    assert clipper.transform([[1e9]])[0, 0] == pytest.approx(8.1)
    unclipped = QuantileClipper(0).fit(np.arange(10).reshape(-1, 1))
    assert unclipped.transform([[1e9]])[0, 0] == 1e9


def test_serialization_scores_and_predicted_class_confidence(
    records, calibrated, tmp_path, monkeypatch
):
    bundle = {
        "version": BUNDLE_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "features": list(feature_names()),
        "sklearn_version": sklearn.__version__,
        "models": {p: calibrated for p in features.Presupuesto},
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(config, "MODEL_PATH", str(path))
    loaded = cargar()
    before = Modelo(bundle).puntuar(records[0])
    after = loaded.puntuar(records[0])
    assert before == after
    response = loaded.predecir(records[0])
    assert 0.5 <= response["confidence"] <= 1
    assert len(response["layer_scores"]) == 6
    assert response["is_synthetic"] == (response["probability_synthetic"] >= 0.5)
    with pytest.raises(ValueError, match="sin intervenciones"):
        loaded.puntuar({**records[0], "turn_features": np.empty((0, len(feature_names())))})


def test_single_bot_layer_lowers_confidence_and_never_increases_it():
    assert confidence_policy(0.99, {"a": 0.99, "b": 0.2, "c": 0.3}) == 0.65
    assert confidence_policy(0.60, {"a": 0.99, "b": 0.2}) == 0.60
    assert confidence_policy(0.99, {"a": 0.99, "b": 0.9}) == 0.99
    assert confidence_policy(0.01, {"a": 0.1}, ["a"]) == 0.9


def test_short_turns_have_explicit_missing_pitch_and_cannot_read_future():
    sr = 8000
    rng = np.random.default_rng(1)
    x = rng.normal(0, 50, (sr * 5, 2)).astype(np.int16)
    x[sr : sr + sr // 5, 0] = 1000
    turns = [{"channel": 1, "start": 0.2, "end": 0.8}, {"channel": 0, "start": 1.0, "end": 1.2}]
    first, _ = extract_turns(x, sr, turns)
    assert np.isnan(first[0, feature_names().index("f0_mean")])
    x[2 * sr :] = rng.integers(-30000, 30000, x[2 * sr :].shape, dtype=np.int16)
    again, _ = extract_turns(x, sr, turns + [{"channel": 0, "start": 3.0, "end": 4.0}])
    np.testing.assert_array_equal(first[0], again[0])
    sample = extract_sample(x, sr, "first_turn", turns=turns)
    assert sample["audio_used_s"] == 1.2
    assert len(sample["turn_features"]) == 1


def test_manifest_and_speaker_contract(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("anon_id,label,split,duration_s\na,human,train,10\nb,synthetic,val,12\n")
    rows, audit = read_manifest(tmp_path)
    assert audit["manifest_nulls"] == 0
    assert speaker_mapping(rows) == ({}, "call")
    mapping = tmp_path / "speakers.csv"
    mapping.write_text("anon_id,speaker_id\na,same_voice\nb,same_voice\n")
    with pytest.raises(ValueError, match="train y val"):
        speaker_mapping(rows, mapping)
    path.write_text("anon_id,label,split,duration_s\na,,train,10\n")
    with pytest.raises(ValueError, match="label/split"):
        read_manifest(tmp_path)


def test_grouping_joins_voice_and_duplicate_caller_pcm_transitively():
    records = [
        {
            "meta": {"anon_id": i, "split": "train"},
            "audit": {"caller_sha256": pcm},
            "samples": {"full": {}},
        }
        for i, pcm in [("a", "x"), ("b", "y"), ("c", "y")]
    ]
    assign_groups(records, {"a": ("speaker:u",), "b": ("speaker:u",), "c": ("speaker:v",)})
    assert {r["samples"]["full"]["group"] for r in records} == {"a"}
    records[-1]["meta"]["split"] = "val"
    with pytest.raises(ValueError, match="train y val"):
        assign_groups(records, {})
