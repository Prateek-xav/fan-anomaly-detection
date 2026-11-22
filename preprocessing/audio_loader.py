"""Decode WAV audio and convert it to mono at the shared sample rate."""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import TARGET_SAMPLE_RATE
from preprocessing.audio_cleaning import clean_audio


def read_wav(source):
    try:
        with sf.SoundFile(source) as file:
            if file.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("Expected a WAV recording.")
            audio = file.read(dtype="float32", always_2d=True)
            rate = file.samplerate
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Unable to decode WAV: {exc}") from exc
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and contain only finite samples.")
    return audio, rate


def load_audio(source, target_rate=TARGET_SAMPLE_RATE):
    audio, sample_rate = read_wav(source)
    mono = clean_audio(audio.mean(axis=1))
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    return mono.astype(np.float32, copy=False)
