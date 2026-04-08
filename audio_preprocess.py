"""Audio preprocessing chain for robust ASR in noisy/background-music scenarios."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, resample_poly

from config import AudioEnhanceParams


def preprocess_audio(audio: np.ndarray, source_sr: int, cfg: AudioEnhanceParams) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    if source_sr != cfg.target_sample_rate:
        gcd = int(np.gcd(source_sr, cfg.target_sample_rate))
        x = resample_poly(x, cfg.target_sample_rate // gcd, source_sr // gcd).astype(np.float32)

    x = highpass(x, cfg.target_sample_rate, cfg.hp_hz)
    x = lowpass(x, cfg.target_sample_rate, cfg.lp_hz)
    x = bandpass(x, cfg.target_sample_rate, cfg.band_low_hz, cfg.band_high_hz)

    x = suppress_background_music(x)
    x = rms_normalize(x, cfg.target_rms)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    c = max(1e-5, cutoff_hz / (sr * 0.5))
    b, a = butter(3, c, btype="high")
    return lfilter(b, a, x).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    c = min(0.999, cutoff_hz / (sr * 0.5))
    b, a = butter(3, c, btype="low")
    return lfilter(b, a, x).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    ny = sr * 0.5
    low = max(1e-5, low_hz / ny)
    high = min(0.999, high_hz / ny)
    if low >= high:
        return x
    b, a = butter(4, [low, high], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def suppress_background_music(x: np.ndarray) -> np.ndarray:
    env = np.abs(x)
    floor = float(np.percentile(env, 25))
    th = floor * 2.0
    y = x.copy()
    y[env < th] *= 0.3
    return y.astype(np.float32)


def rms_normalize(x: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    if rms < 1e-8:
        return x
    return (x * (target_rms / rms)).astype(np.float32)
