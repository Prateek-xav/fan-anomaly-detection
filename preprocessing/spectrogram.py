"""Batched power spectra and fixed-reference log-Mel features."""
from functools import lru_cache

import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann

from config import HOP_LENGTH, N_FFT, N_MELS, TARGET_SAMPLE_RATE

POWER_FLOOR = 1e-10


@lru_cache(maxsize=1)
def window():
    return hann(N_FFT, sym=False).astype(np.float32)


@lru_cache(maxsize=1)
def mel_filters():
    # Slaney Mel spacing is linear below 1 kHz and logarithmic above it.
    linear_step = 200.0 / 3.0
    log_step = np.log(6.4) / 27.0
    transition = 1000.0 / linear_step
    nyquist = TARGET_SAMPLE_RATE / 2
    if nyquist < 1000:
        max_mel = nyquist / linear_step
    else:
        max_mel = transition + np.log(nyquist / 1000.0) / log_step
    mel_edges = np.linspace(0, max_mel, N_MELS + 2)
    hz_edges = np.where(
        mel_edges < transition, linear_step * mel_edges,
        1000.0 * np.exp(log_step * (mel_edges - transition)),
    )
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / TARGET_SAMPLE_RATE)
    offsets = frequencies[None, :] - hz_edges[:, None]
    widths = np.diff(hz_edges)[:, None]
    triangles = np.maximum(0, np.minimum(offsets[:-2] / widths[:-1], -offsets[2:] / widths[1:]))
    area_scale = 2.0 / (hz_edges[2:] - hz_edges[:-2])
    return (triangles * area_scale[:, None]).astype(np.float32)


def power_spectrogram(chunks):
    padded = np.pad(np.atleast_2d(chunks), ((0, 0), (N_FFT // 2, N_FFT // 2)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT, axis=-1)
    spectrum = rfft(frames[:, ::HOP_LENGTH] * window(), axis=-1)
    return np.square(np.abs(spectrum)).transpose(0, 2, 1)


def log_mel(chunks):
    mel_power = mel_filters() @ power_spectrogram(chunks)
    decibels = 10 * np.log10(np.maximum(mel_power, POWER_FLOOR))
    return decibels[:, None, :, :].astype(np.float32, copy=False)
