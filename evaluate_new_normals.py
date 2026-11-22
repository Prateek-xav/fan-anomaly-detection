"""Evaluate WAV-only false alarms on separate, normal-only fan sections."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from config import INSPECTION_CSV
from proxy_cnn import REPORT, batch_features, load, probabilities


def main(folder):
    model, mean, std, settings = load()
    threshold = settings["wav_only"]["threshold"]
    with INSPECTION_CSV.open(encoding="utf-8") as stream:
        seen = {r["content_sha256"] for r in csv.DictReader(stream)}
    rows = []
    skipped = 0
    for path in sorted(folder.glob("*_train_normal_*.wav")):
        audio, rate = sf.read(path, always_2d=True, dtype="float32")
        digest = hashlib.sha256(f"{rate}:{audio.shape[1]}:".encode())
        digest.update(audio.astype("<f4", copy=False).tobytes())
        key = digest.hexdigest()
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        rows.append(path)
    counts, alerts = Counter(), Counter()
    for start in range(0, len(rows), 64):
        paths = rows[start:start+64]
        scores = 1 - probabilities(model, (batch_features(paths) - mean) / std).max(axis=1)
        for path, score in zip(paths, scores):
            group = "_".join(path.stem.split("_")[1:3])
            counts[group] += 1
            alerts[group] += int(score >= threshold)
    result = {"unique_normals": len(rows), "duplicate_waveforms_skipped": skipped,
              "false_alarms": sum(alerts.values()), "sections": dict(counts),
              "false_alarms_by_section_domain": dict(alerts)}
    (REPORT / "fresh_normals.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)
