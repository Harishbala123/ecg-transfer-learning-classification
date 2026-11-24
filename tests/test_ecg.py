import numpy as np
import pandas as pd
import pytest

from src.data.apnea import record_class, subject_split
from src.data.features import beat_series, minute_image, scalogram

SIG = {"context_minutes": 5, "resample_hz": 4.0, "rr_min": 0.3, "rr_max": 2.0, "f_min": 0.01, "f_max": 0.5,
       "n_freqs": 32, "image_width": 64}


def synthetic_ecg(minutes=10, fs=100, hr=60, seed=0):
    """Spike train with a 0.25 Hz amplitude modulation (breathing) and a 0.03 Hz heart-rate oscillation."""
    rng = np.random.default_rng(seed)
    t, beats = 0.0, []
    while t < minutes * 60 - 1:
        rr = 60 / hr * (1 + 0.1 * np.sin(2 * np.pi * 0.03 * t))
        t += rr
        beats.append(t)
    beats = np.array(beats[:-1])
    ecg = rng.normal(0, 0.01, minutes * 60 * fs).astype(np.float32)
    idx = (beats * fs).astype(int)
    ecg[idx] += 1 + 0.2 * np.sin(2 * np.pi * 0.25 * beats)
    return ecg, idx, fs


def test_record_class_thresholds():
    assert [record_class(x) for x in (0, 4, 5, 99, 100, 400)] == ["C", "C", "B", "B", "A", "A"]


def test_subject_split_keeps_subjects_together():
    rows = []
    for s in range(30):
        cls = "ABC"[s % 3]
        for k in range(1 + s % 3):
            rows.append({"record": f"r{s}_{k}", "subject": s, "record_class": cls})
    meta = pd.DataFrame(rows)
    split = subject_split(meta, 0.15, 0.2, 0)
    where = {r: k for k, v in split.items() for r in v}
    assert meta.assign(s=meta.record.map(where)).groupby("subject").s.nunique().max() == 1
    for k in split:   # every split has every record class
        assert set(meta[meta.record.isin(split[k])].record_class) == {"A", "B", "C"}


def test_beat_series_recovers_rr():
    ecg, qrs, fs = synthetic_ecg()
    t, rr, amp = beat_series(ecg, qrs, fs, 0.3, 2.0)
    assert len(rr) > 500 and abs(np.median(rr) - 1.0) < 0.05
    assert amp.std() > 0.05


def test_scalogram_peaks_at_signal_frequency():
    hz = 4.0
    x = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 1 / hz))
    W = scalogram(x, hz, 0.01, 0.5, 64)
    freqs = np.geomspace(0.5, 0.01, 64)
    peak = freqs[np.argmax(np.abs(W[:, 400:800]).mean(1))]
    assert 0.04 < peak < 0.063


def test_minute_image_shape_and_none_at_gaps():
    ecg, qrs, fs = synthetic_ecg()
    t, rr, amp = beat_series(ecg, qrs, fs, 0.3, 2.0)
    img = minute_image(t, rr, amp, 4, SIG)
    assert img.shape == (32, 64, 3) and img.dtype == np.float16 and np.isfinite(img.astype(float)).all()
    assert np.abs(img[..., 2]).max() <= 1.0 + 1e-3
    assert minute_image(t[:10], rr[:10], amp[:10], 4, SIG) is None


tf = pytest.importorskip("tensorflow")


def test_alexnet_shapes_and_frozen_backbone():
    from src.models.alexnet import alexnet_features, build_alexnet_classifier, set_backbone_trainable
    import keras
    inp = keras.Input((224, 224, 3))
    assert keras.Model(inp, alexnet_features(inp)).output_shape == (None, 6, 6, 256)
    m = build_alexnet_classifier((32, 64, 3), 2, [-8, -8, -1], [0, 0, 1], dense_units=16)
    set_backbone_trainable(m, ())
    assert all(not m.get_layer(f"conv{i}").trainable for i in range(1, 6))
    set_backbone_trainable(m, ("conv4", "conv5"))
    assert m.get_layer("conv5").trainable and not m.get_layer("conv3").trainable
    out = m.predict(np.zeros((2, 32, 64, 3), np.float32), verbose=0)
    assert out.shape == (2, 2) and np.allclose(out.sum(1), 1, atol=1e-5)


def test_torch_to_keras_conv_layout():
    from src.models.alexnet import torch_to_keras_conv
    w = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    k = torch_to_keras_conv(w)
    assert k.shape == (4, 5, 3, 2) and k[1, 2, 0, 1] == w[1, 0, 1, 2]
