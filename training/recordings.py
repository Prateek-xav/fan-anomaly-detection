"""Build recording-level partitions from the inspected WAV manifest."""
import csv
import re
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from config import DATASET_ROOT, INSPECTION_CSV, LABELS, RANDOM_SEED, VALIDATION_FRACTION


@dataclass(frozen=True)
class Recording:
    id: str
    label: int
    digest: str
    section: str

    @property
    def path(self):
        return DATASET_ROOT / self.id


def read_recordings(split):
    with INSPECTION_CSV.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset_split"] == split]
    recordings = []
    for row in rows:
        recording_id = row["recording_id"]
        match = re.match(r"section_(\d+)_", recording_id.split("/")[-1])
        if row["status"] != "ok" or row["label"] not in LABELS or match is None:
            raise ValueError(f"Review the inspection row for {recording_id}.")
        recordings.append(Recording(recording_id, LABELS.index(row["label"]),
                                    row["content_sha256"], match[1]))
    if not recordings:
        raise ValueError(f"No inspected recordings found for {split}.")
    return recordings


def verify_disjoint(*partitions):
    seen_ids = set()
    seen_digests = set()
    for rows in partitions:
        ids = {row.id for row in rows}
        digests = {row.digest for row in rows}
        if len(ids) != len(rows) or seen_ids & ids or seen_digests & digests:
            raise ValueError("A recording ID or identical waveform overlaps partitions.")
        seen_ids.update(ids)
        seen_digests.update(digests)


def development_partitions():
    normal = read_recordings("train")
    if any(row.label for row in normal):
        raise ValueError("Expected only normal recordings in train.")
    source = read_recordings("source_test")
    source_train, validation = train_test_split(
        source, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED,
        stratify=[(row.section, row.label) for row in source],
    )
    verify_disjoint(normal, source_train, validation)
    return normal, source_train, validation
