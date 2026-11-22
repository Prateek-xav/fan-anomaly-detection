"""Fixed-length waveform chunks with recording-level metadata."""
from dataclasses import dataclass

import numpy as np

from config import CHUNK_DURATION, TARGET_SAMPLE_RATE

SAMPLES_PER_CHUNK = int(CHUNK_DURATION * TARGET_SAMPLE_RATE)


@dataclass(frozen=True)
class Chunks:
    audio: np.ndarray
    recording_id: str
    label: int | None


def segment_audio(waveform, recording_id, label=None):
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("Expected a nonempty mono waveform.")
    tail = -waveform.size % SAMPLES_PER_CHUNK
    if tail:
        waveform = np.pad(waveform, (0, tail))
    return Chunks(waveform.reshape(-1, SAMPLES_PER_CHUNK), recording_id, label)
