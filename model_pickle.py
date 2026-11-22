"""Export and load the final fan classifier as a Python pickle."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from proxy_cnn import feature, tf
from config import PROJECT_ROOT

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "final"
MODEL_FILE = PROJECT_ROOT / "fan_model.pkl"


class PicklePredictor:
    """Load a trusted model export once per Python process."""

    def __init__(self, path=MODEL_FILE):
        with Path(path).open("rb") as stream:
            state = pickle.load(stream)
        if state["format"] != "fan-cnn-v1":
            raise ValueError("Unsupported fan model export.")
        self.model = tf.keras.models.model_from_json(state["architecture"])
        self.model.set_weights(state["weights"])
        self.mean = state["mean"]
        self.std = state["std"]
        self.threshold = state["threshold"]

    def predict(self, wav):
        inputs = (feature(wav)[None] - self.mean) / self.std
        score = float(self.model(inputs, training=False).numpy()[0, 0])
        return {"prediction": "Anomaly" if score >= self.threshold else "Normal",
                "score": score, "threshold": self.threshold}


def export(path):
    model = tf.keras.models.load_model(CHECKPOINT / "fan.keras", compile=False)
    settings = json.loads((CHECKPOINT / "settings.json").read_text())
    with np.load(CHECKPOINT / "normalization.npz") as stats:
        state = {"format": "fan-cnn-v1", "architecture": model.to_json(),
                 "weights": model.get_weights(), "mean": stats["mean"],
                 "std": stats["std"], "threshold": settings["threshold"]}
    with Path(path).open("wb") as stream:
        pickle.dump(state, stream, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MODEL_FILE)
    export(parser.parse_args().out)
