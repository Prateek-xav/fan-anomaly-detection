# Fan sound monitor

Double-click **Start Fan Monitor.cmd** on the training computer. A local browser window opens after the model loads. Keep the terminal open while using the app. Stop with Ctrl+C. Alternatively, run `python app.py` and open the printed address. Use `--port 8766` if the default port is busy.

On another Windows machine use Python 3.10 and the included requirements:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --open-browser
```

Upload WAV files and read the recording-level prediction and score. Use the Known condition menu when a verified label is available. The report updates for the labeled subset. Refresh the page to clear the session. The benchmark panel refers to the saved evaluation.

## Reproduction

The supplied model runs without the original recordings. Training and evaluation require the local dataset. Set FAN_DATASET_ROOT to the directory containing train, source_test and target_test. The default is dataset/fan. The inspected fan manifest is included. To rebuild it, run `python -m inspection.inspect_dataset` first. `python train_final.py` runs section-held-out model comparison, refitting, threshold selection and target replay using the bundled proxy checkpoint. `python evaluate_final_normals.py 'C:\path\to\eval_data_fan_train\fan\train'` runs the separate normal-only check. To retrain the auxiliary encoder first, run `python proxy_cnn.py train 'C:\path\to\raw\dev'` with the four other machine types present.

See VIVA.txt for the model explanation. The deployed model, its normalization arrays and threshold are in checkpoints/final. Model-selection results are in settings.json, and the evaluation metrics and predictions are in results/final.

## Use the model in another website

The included `fan_model.pkl` bundles the final Keras model architecture and weights, normalization arrays, and decision threshold. The browser sends audio to a Python backend that loads the export. Use the project files and dependencies for the same WAV preprocessing and inference.

```python
from model_pickle import PicklePredictor

model = PicklePredictor("fan_model.pkl")
result = model.predict("recording.wav")
# {"prediction": "Normal", "score": 0.12, "threshold": 0.7463675737380981}
```

For an HTTP integration, send raw WAV bytes to `POST /predict` with `Content-Type: audio/wav` and read the JSON response. Serve the frontend and endpoint on the same origin for the supplied local server. Load the model once when your backend starts and load only trusted pickle files. The bundled launcher binds to 127.0.0.1 for local use.

```powershell
curl.exe --data-binary '@recording.wav' -H 'Content-Type: audio/wav' http://127.0.0.1:8765/predict
```

## Measured performance

The final model reached 62.0% accuracy and 0.658 ROC-AUC on the 600-recording target benchmark replay. It detected 158 of 300 anomalies and flagged 86 of 300 healthy recordings. This is an experimental classifier. A Normal result cannot reliably rule out a fault.
