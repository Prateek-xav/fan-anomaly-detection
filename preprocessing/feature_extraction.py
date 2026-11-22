"""Shared recording-to-feature pipeline and optional waveform summaries."""
import numpy as np

from preprocessing.audio_loader import load_audio
from preprocessing.audio_segmentation import segment_audio
from preprocessing.spectrogram import log_mel


def recording_features(source, recording_id="", label=None):
    waveform = load_audio(source)
    chunks = segment_audio(waveform, recording_id, label)
    return log_mel(chunks.audio)


def waveform_summary(waveform):
    return {
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
        "peak": float(np.max(np.abs(waveform))),
        "zero_crossing_rate": float(np.mean(np.signbit(waveform[1:]) != np.signbit(waveform[:-1])))
        if waveform.size > 1 else 0.0,
    }
