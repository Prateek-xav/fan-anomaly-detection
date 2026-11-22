import json

import numpy as np
from proxy_cnn import ART, PROJECT_ROOT, feature, tf

FINAL = PROJECT_ROOT / "checkpoints" / "final"
RESULTS = PROJECT_ROOT / "results" / "final"


class Predictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(FINAL / "fan.keras", compile=False)
        with np.load(FINAL / "normalization.npz") as stats:
            self.mean, self.std = stats["mean"], stats["std"]
        self.settings = json.loads((FINAL / "settings.json").read_text())
        self.threshold = self.settings["threshold"]

    def scores(self, raw):
        return self.model((raw-self.mean)/self.std, training=False).numpy().ravel()

    def predict(self, wav):
        score = float(self.scores(feature(wav)[None])[0])
        return {"prediction": "Anomaly" if score >= self.threshold else "Normal",
                "score": score, "threshold": self.threshold}
