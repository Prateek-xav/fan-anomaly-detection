"""Audit the local WAV corpus one recording at a time."""
import csv
import hashlib
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from config import CLIP_LEVEL, DATASET_ROOT, INSPECTION_CSV, SILENCE_RMS, SPLITS

FIELDS = (
    "file_path", "recording_id", "dataset_split", "label", "sample_rate",
    "channels", "duration_seconds", "num_samples", "peak_amplitude",
    "rms_amplitude", "silent", "clipping", "invalid_values", "content_sha256",
    "status", "error",
)


def filename_label(path: Path) -> str:
    tokens = re.findall(r"(?<=_)(normal|anomaly)(?=_)", path.stem.lower())
    return tokens[0] if len(tokens) == 1 else "unknown"


def inspect_one(task):
    split, path = task
    row = {
        "file_path": path.relative_to(DATASET_ROOT).as_posix(),
        "recording_id": path.relative_to(DATASET_ROOT).as_posix(),
        "dataset_split": split,
        "label": filename_label(path),
    }
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        frames, channels = audio.shape
        row.update(sample_rate=rate, channels=channels, num_samples=frames,
                   duration_seconds=frames / rate)
        invalid = int(audio.size - np.count_nonzero(np.isfinite(audio)))
        row["invalid_values"] = invalid
        if frames == 0 or invalid:
            raise ValueError("Empty recording" if frames == 0 else "Non-finite audio samples")
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        digest = hashlib.sha256(f"{rate}:{channels}:".encode())
        digest.update(audio.astype("<f4", copy=False).tobytes())
        row.update(peak_amplitude=peak, rms_amplitude=rms, silent=rms <= SILENCE_RMS,
                   clipping=peak >= CLIP_LEVEL, content_sha256=digest.hexdigest(),
                   status="ok")
    except (OSError, RuntimeError, ValueError) as exc:
        row.update(status="error", error=str(exc))
    return row


def main():
    tasks = []
    for split, folder in SPLITS.items():
        if not folder.is_dir():
            raise SystemExit(f"Missing split directory: {folder}")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
        if not files:
            raise SystemExit(f"No WAV recordings in {folder}")
        tasks.extend((split, path) for path in files)

    INSPECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    issues = Counter()
    invalid_labels = 0
    temporary = INSPECTION_CSV.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(inspect_one, tasks):
                split, label = row["dataset_split"], row["label"]
                counts[split, label] += 1
                invalid_labels += label == "unknown" or (split == "train" and label != "normal")
                for issue in ("silent", "clipping"):
                    issues[split, issue] += bool(row.get(issue))
                issues[split, "read_errors"] += row["status"] != "ok"
                writer.writerow(row)
    temporary.replace(INSPECTION_CSV)
    for split in SPLITS:
        print(f"{split}: counts {dict((k, counts[split, k]) for k in ('normal', 'anomaly', 'unknown'))}")
        print(f"{split}: quality {dict((k, issues[split, k]) for k in ('silent', 'clipping', 'read_errors'))}")
    print(f"Report: {INSPECTION_CSV}")
    if invalid_labels:
        raise SystemExit(f"{invalid_labels} unexpected labels. Review the CSV before training.")


if __name__ == "__main__":
    main()
