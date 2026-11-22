"""Shared paths and experiment settings."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get(
    "FAN_DATASET_ROOT",
    str(PROJECT_ROOT / "dataset" / "fan"),
)).expanduser().resolve()
TRAIN_DIR = DATASET_ROOT / "train"
SOURCE_TEST_DIR = DATASET_ROOT / "source_test"
TARGET_TEST_DIR = DATASET_ROOT / "target_test"
SPLITS = {"train": TRAIN_DIR, "source_test": SOURCE_TEST_DIR, "target_test": TARGET_TEST_DIR}
LABELS = ("normal", "anomaly")
RESULTS_DIR = PROJECT_ROOT / "results"
INSPECTION_DIR = RESULTS_DIR / "inspection"
INSPECTION_CSV = INSPECTION_DIR / "audio_inspection.csv"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TARGET_SAMPLE_RATE = 16_000
CHUNK_DURATION = 1
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2
SILENCE_RMS = 1e-4
CLIP_LEVEL = 0.9999
