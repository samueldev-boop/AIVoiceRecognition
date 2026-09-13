"""Contratos que afectan a la evaluacion y al audio del servicio."""

import base64
import copy
import io

import numpy as np
import pytest
import soundfile as sf

from app import audio


def test_resampling_is_identical_for_file_and_http(tmp_path):
    sr = 16000
    t = np.arange(sr) / sr
    x = np.column_stack([3000 * np.sin(2 * np.pi * 220 * t), 1000 * np.sin(2 * np.pi * 440 * t)])
    buffer = io.BytesIO()
    sf.write(buffer, x.astype(np.int16), sr, format="WAV", subtype="PCM_16")
    path = tmp_path / "input.wav"
    path.write_bytes(buffer.getvalue())
    local, local_sr = audio.leer_wav(str(path))
    http, http_sr = audio.decodificar(base64.b64encode(buffer.getvalue()).decode())
    assert local_sr == http_sr == 8000
    assert local.shape == (8000, 2)
    np.testing.assert_array_equal(local, http)
    assert audio.remuestrear(local, 8000) is local


def test_augmentation_stays_inside_the_training_fold():
    pytest.importorskip("audiomentations")
    from scripts.train_issue11 import training_samples

    clean = [{"call_id": str(i), "group": str(i // 2), "split": "train"} for i in range(6)]
    data = {
        "clean": clean,
        "physical": [[dict(r), dict(r)] for r in clean],
        "rawboost": [[dict(r)] for r in clean],
    }
    fitted = training_samples(data, [0, 1, 2, 3], "rawboost")
    assert len(fitted) == 16
    assert {r["group"] for r in fitted} == {"0", "1"}
    data["physical"][0][0]["split"] = "val"
    with pytest.raises(ValueError, match="val"):
        training_samples(data, [0], "physical")


def test_promotion_rejects_false_positives_and_fragile_auc():
    pytest.importorskip("audiomentations")
    from scripts.augmentation import CODECS
    from scripts.train_issue11 import promotion_gate

    names = ["clean", "humano_limpio", "bot_evasivo", "bot_sala", *[f"humano_{c}" for c in CODECS]]
    result = {
        "probabilities": dict.fromkeys(names),
        "metrics": {
            name: {
                "auc": 0.99,
                "thresholds": {"0.5": {"confusion_matrix": [[10, 0], [0, 10]], "fnr": 0.0}},
            }
            for name in names
        },
    }
    robust = copy.deepcopy(result)
    assert promotion_gate(result, robust)["passed"]
    result["metrics"]["humano_gsm"]["thresholds"]["0.5"]["confusion_matrix"][0][1] = 1
    assert not promotion_gate(result, robust)["passed"]
    result = copy.deepcopy(robust)
    robust["metrics"]["bot_sala"]["auc"] = 0.85
    assert not promotion_gate(result, robust)["passed"]


def test_codecs_roundtrip_changes_signal_and_preserves_samples():
    pytest.importorskip("audiomentations")
    import shutil
    import subprocess

    from scripts.augmentation import CODECS, codec_roundtrip

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg solo se requiere durante entrenamiento")
    encoders = subprocess.run(
        ["ffmpeg", "-encoders"], capture_output=True, check=True
    ).stdout.decode()
    t = np.arange(8000) / 8000
    signal = (0.2 * np.sin(2 * np.pi * 237 * t) + 0.05 * np.sin(2 * np.pi * 1700 * t)).astype(
        np.float32
    )
    for codec, (encoder, _, _) in CODECS.items():
        if encoder not in encoders:
            continue
        changed = codec_roundtrip(signal, 8000, codec)
        assert changed.shape == signal.shape
        assert np.isfinite(changed).all()
        assert not np.array_equal(changed, signal)
