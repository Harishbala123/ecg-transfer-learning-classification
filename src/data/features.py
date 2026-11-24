"""ECG -> 3-channel time-frequency image.

Why this representation: obstructive apnea produces (1) cyclic variation of heart rate, visible in the RR-interval
series at ~0.01-0.05 Hz, and (2) changes in breathing that modulate the R-wave amplitude (ECG-derived
respiration, EDR, ~0.15-0.4 Hz). Both are slow modulations of the ECG, so the raw 100 Hz waveform is reduced
to these two beat-to-beat series, and their continuous wavelet transforms are stacked as image channels.
That matches the 3-channel image input expected by an ImageNet-pretrained AlexNet.

  channel 0: log power of the RR-interval scalogram       (heart-rate variability)
  channel 1: log power of the EDR scalogram               (respiration surrogate)
  channel 2: cos(phase difference) between the two CWTs   (cardio-respiratory coupling)
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt


def refine_peaks(ecg: np.ndarray, qrs: np.ndarray, fs: int, search_s: float = 0.05) -> np.ndarray:
    """Move each QRS annotation to the largest absolute deflection within ±50 ms (R-peak)."""
    w = int(search_s * fs)
    base = np.median(ecg)
    out = np.empty_like(qrs)
    for i, q in enumerate(qrs):
        lo, hi = max(0, q - w), min(len(ecg), q + w + 1)
        out[i] = lo + int(np.argmax(np.abs(ecg[lo:hi] - base)))
    return out


def beat_series(ecg: np.ndarray, qrs: np.ndarray, fs: int, rr_min: float, rr_max: float):
    """Beat times (s), RR intervals (s) and R amplitudes for clean beats only."""
    peaks = refine_peaks(ecg, qrs, fs)
    t = peaks / fs
    rr = np.diff(t)
    amp = ecg[peaks[1:]] - np.median(ecg)
    t = t[1:]
    ok = (rr >= rr_min) & (rr <= rr_max)
    # reject ectopic / missed beats: > 30 % deviation from the local median RR
    if ok.sum() > 5:
        med = np.convolve(np.where(ok, rr, np.median(rr[ok])), np.ones(5) / 5, mode="same")
        ok &= np.abs(rr - med) <= 0.3 * med
    return t[ok], rr[ok], amp[ok]


def _grid_series(t, v, t0, t1, hz) -> np.ndarray | None:
    m = (t >= t0) & (t < t1)
    if m.sum() < 0.5 * (t1 - t0) * 0.5:     # require >= 50 % of beats expected at 30 bpm
        return None
    grid = np.arange(t0, t1, 1.0 / hz)
    x = np.interp(grid, t[m], v[m])
    return x - x.mean()


def scalogram(x: np.ndarray, hz: float, f_min: float, f_max: float, n_freqs: int) -> np.ndarray:
    """Complex Morlet CWT on log-spaced frequencies (high to low). Returns complex [n_freqs, len(x)]."""
    wavelet = "cmor1.5-1.0"
    freqs = np.geomspace(f_max, f_min, n_freqs)
    scales = pywt.central_frequency(wavelet) * hz / freqs
    coef, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / hz, method="fft")
    return coef


def minute_image(t, rr, amp, minute: int, cfg_sig: dict) -> np.ndarray | None:
    """Image for labelled minute `minute` using a centred context window. Returns float16 [F, W, 3] or None."""
    half = cfg_sig["context_minutes"] // 2
    t0, t1 = (minute - half) * 60.0, (minute + half + 1) * 60.0
    hz = cfg_sig["resample_hz"]
    r = _grid_series(t, rr, t0, t1, hz)
    a = _grid_series(t, amp, t0, t1, hz)
    if r is None or a is None:
        return None
    a = a / (np.std(a) + 1e-6) * np.std(r)   # put EDR on the RR scale; its absolute amplitude is electrode-dependent
    wr = scalogram(r, hz, cfg_sig["f_min"], cfg_sig["f_max"], cfg_sig["n_freqs"])
    wa = scalogram(a, hz, cfg_sig["f_min"], cfg_sig["f_max"], cfg_sig["n_freqs"])
    ch = [np.log10(np.abs(wr) ** 2 + 1e-8), np.log10(np.abs(wa) ** 2 + 1e-8), np.cos(np.angle(wr) - np.angle(wa))]
    width = cfg_sig["image_width"]
    img = np.stack([cv2.resize(c.astype(np.float32), (width, c.shape[0]), interpolation=cv2.INTER_AREA) for c in ch], -1)
    return img.astype(np.float16)


def record_images(rec, cfg_sig: dict):
    """All valid minute images of a record. Minutes without full context (first/last 2) or with too few
    clean beats are skipped. Returns images [n, F, W, 3], labels [n], minute indices [n]."""
    t, rr, amp = beat_series(rec.ecg, rec.qrs, rec.fs, cfg_sig["rr_min"], cfg_sig["rr_max"])
    half = cfg_sig["context_minutes"] // 2
    imgs, ys, ms = [], [], []
    n_min = min(len(rec.minute_labels), len(rec.ecg) // (60 * rec.fs))
    for m in range(half, n_min - half):
        img = minute_image(t, rr, amp, m, cfg_sig)
        if img is not None:
            imgs.append(img)
            ys.append(rec.minute_labels[m])
            ms.append(m)
    return np.stack(imgs), np.array(ys, np.int8), np.array(ms, np.int16)
