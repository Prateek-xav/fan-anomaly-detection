"""Train, evaluate or run a section-classification CNN with proxy outlier exposure."""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from config import PROJECT_ROOT, RANDOM_SEED
from evaluation.metrics import binary_metrics
from preprocessing.audio_loader import load_audio
from preprocessing.audio_segmentation import segment_audio
from preprocessing.spectrogram import log_mel
from training.recordings import development_partitions, read_recordings, verify_disjoint

ART = PROJECT_ROOT / "checkpoints" / "proxy"
REPORT = PROJECT_ROOT / "results" / "proxy"
MACHINES = ("gearbox", "pump", "slider", "valve")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)


def feature(path):
    chunks = segment_audio(load_audio(path), "").audio
    mel = log_mel(chunks)[:, 0].transpose(1, 0, 2).reshape(64, -1)
    # Time-pool into 40 bins to retain order on recordings of variable length.
    edges = np.linspace(0, mel.shape[1], 41).astype(int)
    bands = [mel[:, left:max(left + 1, right)] for left, right in zip(edges, edges[1:])]
    return np.stack([np.stack([b.mean(1), b.std(1)], -1) for b in bands], axis=1)


def batch_features(paths):
    with threadpool_limits(limits=1), ThreadPoolExecutor(4) as pool:
        return np.stack(list(pool.map(feature, paths)))


def network():
    layers = tf.keras.layers
    inputs = layers.Input((64, 40, 2))
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(inputs)
    residual = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    residual = layers.Conv2D(16, 3, padding="same")(residual)
    x = layers.ReLU()(layers.Add()([x, residual]))
    x = layers.AveragePooling2D(2)(x)
    x = layers.Conv2D(24, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D((1, 5))(x)
    embedding = layers.Dense(24, activation="tanh")(layers.Flatten()(x))
    logits = layers.Dense(3)(embedding)
    return tf.keras.Model(inputs, [logits, embedding])


def probabilities(model, x):
    return np.vstack([tf.nn.softmax(model(batch, training=False)[0]).numpy()
                      for batch in np.array_split(x, max(1, (len(x) + 63) // 64))])


def choose(labels, scores):
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # Include the all-normal decision at a finite, reproducible threshold.
    thresholds[0] = np.nextafter(scores.max(), np.float32(np.inf))
    candidates = np.flatnonzero(fpr <= 0.1 + 1e-12)
    best = max(candidates, key=lambda i: (recall[i], -fpr[i]))
    return float(thresholds[best])


def scores_for(probs, sections):
    return {"wav_only": 1 - probs.max(axis=1),
            "known_section": 1 - probs[np.arange(len(probs)), sections]}


def train(root):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    normal, _, val = development_partitions()
    sections = np.array([int(r.section) for r in normal])
    source = np.array(["_source_train_" in r.id for r in normal])
    proxy_paths = []
    for machine in MACHINES:
        paths = sorted((root / f"dev_data_{machine}" / machine / "train").glob("*_source_train_normal_*.wav"))
        if len(paths) < 250:
            raise ValueError(f"Need 250 normal source clips for {machine}.")
        proxy_paths.extend(rng.choice(paths, size=250, replace=False))
    print("Extracting recording features.", flush=True)
    x = batch_features([r.path for r in normal])
    proxies = batch_features(proxy_paths)
    v = batch_features([r.path for r in val])
    mean = x[source].mean(axis=(0, 2), keepdims=True)
    std = x[source].std(axis=(0, 2), keepdims=True).clip(1e-4)
    x, proxies, v = [(arr - mean) / std for arr in (x, proxies, v)]
    model = network()
    optimizer = tf.keras.optimizers.Adam(0.001)
    @tf.function
    def step(fan, labels, others, targets, target_labels, adaptation):
        with tf.GradientTape() as tape:
            logits, embeds = model(fan, training=True)
            proxy_logits, _ = model(others, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            loss -= 0.5 * tf.reduce_mean(tf.nn.log_softmax(proxy_logits))
            if adaptation:
                target_logits, target_embeds = model(targets, training=True)
                loss += 0.25 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_labels, logits=target_logits))
                distance = tf.norm(embeds[:, None, :] - target_embeds[None, :, :] + 1e-8, axis=-1)
                same = labels[:, None] == target_labels[None, :]
                align = tf.reduce_mean(tf.square(tf.boolean_mask(distance, same)))
                repel = tf.reduce_mean(tf.square(tf.nn.relu(2.0 - tf.boolean_mask(distance, ~same))))
                loss += 0.1 * (align + repel)
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
    target = np.flatnonzero(~source)
    for epoch in range(15):
        adapt = epoch >= 12
        if epoch == 12:
            optimizer.learning_rate.assign(0.00015)
        losses = []
        for ids in np.array_split(rng.permutation(np.flatnonzero(source)), 64):
            other = rng.integers(len(proxies), size=len(ids))
            losses.append(float(step(x[ids], sections[ids], proxies[other], x[target], sections[target], adapt)))
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    labels = np.array([r.label for r in val])
    val_scores = scores_for(probabilities(model, v), [int(r.section) for r in val])
    settings = {}
    for mode, scores in val_scores.items():
        threshold = choose(labels, scores)
        settings[mode] = {"threshold": threshold, "validation": binary_metrics(labels, scores, threshold)}
    model.save(ART / "fan.keras")
    np.savez(ART / "normalization.npz", mean=mean, std=std)
    (ART / "settings.json").write_text(json.dumps(settings, indent=2))
    manifest = {"fan_train": [r.id for r in normal], "proxy_train": [p.relative_to(root).as_posix() for p in proxy_paths],
                "validation": [r.id for r in val]}
    (REPORT / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Model and thresholds frozen.", flush=True)


def load():
    model = tf.keras.models.load_model(ART / "fan.keras", compile=False)
    with np.load(ART / "normalization.npz") as data:
        mean, std = data["mean"], data["std"]
    return model, mean, std, json.loads((ART / "settings.json").read_text())


def evaluate():
    model, mean, std, settings = load()
    target = read_recordings("target_test")
    normal, source_train, val = development_partitions()
    verify_disjoint(normal + source_train + val, target)
    start = perf_counter()
    probs = probabilities(model, (batch_features([r.path for r in target]) - mean) / std)
    elapsed = perf_counter() - start
    sections = np.array([int(r.section) for r in target])
    labels = np.array([r.label for r in target])
    report = {"batch_seconds": elapsed}
    series = scores_for(probs, sections)
    for mode, scores in series.items():
        threshold = settings[mode]["threshold"]
        report[mode] = {"overall": binary_metrics(labels, scores, threshold),
                        "by_section": {str(i): binary_metrics(labels[sections == i], scores[sections == i], threshold)
                                       for i in np.unique(sections)}}
    (REPORT / "target_replay.json").write_text(json.dumps(report, indent=2))
    with (REPORT / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recording", "label", *series])
        writer.writerows(zip([r.id for r in target], labels, *series.values()))
    print({mode: report[mode]["overall"] for mode in series})


def predict(path, section):
    model, mean, std, settings = load()
    probs = probabilities(model, (feature(path)[None] - mean) / std)[0]
    mode = "wav_only" if section is None else "known_section"
    score = float(1 - (probs.max() if section is None else probs[section]))
    threshold = settings[mode]["threshold"]
    print(json.dumps({"prediction": "Anomaly" if score >= threshold else "Normal",
                      "score": score, "threshold": threshold, "mode": mode}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train").add_argument("machine_data_root", type=Path)
    sub.add_parser("evaluate")
    pred = sub.add_parser("predict")
    pred.add_argument("wav")
    pred.add_argument("--section", choices=(0, 1, 2), type=int)
    args = parser.parse_args()
    if args.command == "train":
        train(args.machine_data_root)
    elif args.command == "evaluate":
        evaluate()
    else:
        predict(args.wav, args.section)
