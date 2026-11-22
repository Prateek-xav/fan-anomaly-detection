"""Local WAV upload interface. Run python app.py and open the printed URL."""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from final_model import RESULTS
from model_pickle import PicklePredictor

PAGE = Path(__file__).with_name("index.html")
MODEL_LOCK = threading.Lock()


def report(truth, scores, threshold):
    y = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.shape != y.shape or not y.size or not np.isin(y,[0,1]).all()
            or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
        raise ValueError("Provide one valid score and label per recording.")
    decisions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, decisions, labels=[0,1]).ravel().tolist()
    both = len(set(y)) == 2
    return {"accuracy": float(np.mean(y==decisions)),
            "classes": classification_report(y, decisions, labels=[0,1], target_names=["Normal","Anomaly"],
                                             output_dict=True, zero_division=0),
            "matrix": [[tn,fp],[fn,tp]], "auc": float(roc_auc_score(y,scores)) if both else None,
            "ap": float(average_precision_score(y,scores)) if both else None,
            "fpr": fp/(fp+tn) if fp+tn else None, "fnr": fn/(fn+tp) if fn+tp else None}


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, value, mime="application/json"):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_data(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/benchmark":
            self.send_data(200, json.loads((RESULTS/"target_replay.json").read_text()))
        else:
            self.send_error(404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.send_error(403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 32*1024*1024:
                raise ValueError("Choose a nonempty file under 32 MiB.")
            data = self.rfile.read(size)
            if self.path == "/predict":
                stream = io.BytesIO(data)
                meta = sf.info(stream)
                if meta.duration > 120 or meta.frames*meta.channels > 12_000_000:
                    raise ValueError("Use a WAV of up to two minutes and 12 million channel samples.")
                stream.seek(0)
                with MODEL_LOCK:
                    result = self.server.predictor.predict(stream)
            elif self.path == "/report":
                values = json.loads(data)
                result = report(values["truth"], values["scores"], self.server.predictor.threshold)
            else:
                return self.send_error(404)
            self.send_data(200,result)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            self.send_data(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    predictor = PicklePredictor()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.predictor = predictor
        print(f"Fan monitor ready at http://127.0.0.1:{args.port}", flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
        server.serve_forever()


if __name__ == "__main__":
    main()
