"""One feature vector per recording for supervised CPU training and inference."""
from functools import lru_cache

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal.windows import hann

from config import TARGET_SAMPLE_RATE
from preprocessing.audio_loader import load_audio
from preprocessing.audio_segmentation import segment_audio
from preprocessing.spectrogram import log_mel

TONE_FFT = 4096
TONE_HOP = 1536
TONE_FLOOR = 1e-12


def summarize(values, axis):
    quantiles = np.percentile(values, (10, 50, 90), axis=axis)
    return (quantiles[1], quantiles[2] - quantiles[0])


@lru_cache(maxsize=1)
def tone_basis():
    frequencies = rfftfreq(TONE_FFT, 1 / TARGET_SAMPLE_RATE)
    edges = np.geomspace(35, TARGET_SAMPLE_RATE / 2, 81)
    bins = np.unique(np.r_[np.searchsorted(frequencies, edges), frequencies.size])
    return hann(TONE_FFT, sym=False).astype(np.float32), bins


def pool_spectrum(values, bins, reduce):
    pooled = []
    for start, end in zip(bins[:-1], bins[1:]):
        pooled.append(reduce(values[..., start:end], axis=-1))
    return np.concatenate([np.ravel(part) for part in pooled])


def recording_vector(source):
    waveform = load_audio(source)
    chunks = segment_audio(waveform, "").audio
    patches = log_mel(chunks)[:, 0].transpose(1, 0, 2)
    mel = patches.reshape(patches.shape[0], -1)
    envelope, spread = summarize(mel, axis=1)
    centered = envelope - np.mean(envelope)
    spectral_shape = np.concatenate((envelope, centered, np.diff(envelope), np.diff(envelope, n=2)))
    modulation = np.abs(rfft(patches - patches.mean(axis=-1, keepdims=True), axis=-1))
    mod_bands = (modulation[..., 1:3], modulation[..., 3:7], modulation[..., 7:])
    dynamics = np.concatenate((spread, *[np.median(np.mean(band, axis=-1), axis=1) for band in mod_bands]))

    window, bins = tone_basis()
    windows = np.lib.stride_tricks.sliding_window_view(chunks, TONE_FFT, axis=-1)
    power = np.square(np.abs(rfft(windows[:, ::TONE_HOP] * window, axis=-1))).reshape(-1, len(window) // 2 + 1)
    db = 10 * np.log10(np.maximum(power, TONE_FLOOR))
    tone_median, tone_spread = summarize(db, axis=0)
    contrast = tone_median - median_filter(tone_median, size=17, mode="nearest")
    tones = np.concatenate((
        pool_spectrum(tone_median, bins, np.mean),
        pool_spectrum(contrast, bins, np.max),
        pool_spectrum(contrast, bins, np.mean),
        pool_spectrum(tone_spread, bins, np.mean),
    ))
    fine_window = hann(chunks.shape[-1], sym=False).astype(np.float32)
    fine_power = np.square(np.abs(rfft(chunks * fine_window, axis=-1)))
    max_bin = int(1600 * chunks.shape[-1] / TARGET_SAMPLE_RATE)
    fine_db = 10 * np.log10(np.maximum(fine_power[:, :max_bin], TONE_FLOOR))
    fine_median, fine_spread = summarize(fine_db, axis=0)
    contrast_3hz = fine_median - gaussian_filter1d(fine_median, sigma=3)
    contrast_20hz = fine_median - gaussian_filter1d(fine_median, sigma=20)
    fine_tones = np.concatenate((contrast_3hz, contrast_20hz, fine_spread))
    fine_low = np.concatenate((fine_median[:300], contrast_20hz[:300], fine_spread[:300]))
    return {
        "mel": spectral_shape.astype(np.float32), "dynamics": dynamics.astype(np.float32),
        "tones": tones.astype(np.float32), "fine_tones": fine_tones.astype(np.float32),
        "fine_low": fine_low.astype(np.float32),
    }
