"""Compare heads using held-out fault sections, then freeze and evaluate."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from final_model import FINAL, RESULTS, Predictor
from proxy_cnn import ART, RANDOM_SEED, batch_features, choose, tf
from training.recordings import development_partitions, read_recordings, verify_disjoint
from evaluation.metrics import binary_metrics


def score(model, x):
    return np.concatenate([model(b, training=False).numpy().ravel() for b in np.array_split(x, max(1, len(x)//48))])


def fit(x, y, target_x, fine):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    base = tf.keras.models.load_model(ART/"fan.keras", compile=False)
    emb = tf.keras.Model(base.inputs[0], base.outputs[1])
    values = np.concatenate([emb(b, training=False).numpy() for b in np.array_split(x, 8)])
    scaler = StandardScaler().fit(values)
    clf = LogisticRegression(C=1, max_iter=1500).fit(scaler.transform(values), y)
    head = tf.keras.layers.Dense(1, activation="sigmoid", name="fault_score")
    model = tf.keras.Model(emb.inputs[0], head(emb.output))
    w = clf.coef_[0] / scaler.scale_
    head.set_weights([w.astype("float32")[:, None], (clf.intercept_ - w @ scaler.mean_).astype("float32")])
    if fine:
        for layer in emb.layers:
            layer.trainable = False
        [layer for layer in emb.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1].trainable = True
        emb.layers[-1].trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss="binary_crossentropy")
        rng = np.random.default_rng(RANDOM_SEED)
        train_x = np.concatenate((x, target_x))
        train_y = np.r_[y, np.zeros(len(target_x))].astype("float32")
        for _ in range(10):
            for ids in np.array_split(rng.permutation(len(train_y)), int(np.ceil(len(train_y)/32))):
                model.train_on_batch(train_x[ids], train_y[ids, None])
    return model


def main():
    normal, train, val = development_partitions()
    with np.load(ART/"normalization.npz") as stats:
        mean, std = stats["mean"], stats["std"]
    def inputs(rows):
        return (batch_features([r.path for r in rows])-mean)/std
    x, y = inputs(train), np.array([r.label for r in train])
    target_x = inputs([r for r in normal if "_target_train_" in r.id])
    groups = np.array([r.section for r in train])
    cv = {}
    for method in ("frozen", "fine_tuned"):
        folds = []
        for section in sorted(set(groups)):
            held = groups == section
            model = fit(x[~held], y[~held], target_x, method=="fine_tuned")
            scores = score(model, x[held])
            fpr, recall, _ = roc_curve(y[held], scores, drop_intermediate=False)
            folds.append(float(max(recall[fpr <= .1])))
        cv[method] = folds
        print(method, folds, flush=True)
    winner = max(cv, key=lambda k: (np.mean(cv[k]), min(cv[k])))
    model = fit(x, y, target_x, winner=="fine_tuned")
    val_y = np.array([r.label for r in val])
    val_scores = score(model, inputs(val))
    threshold = choose(val_y, val_scores)
    FINAL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    model.save(FINAL/"fan.keras")
    shutil.copyfile(ART/"normalization.npz", FINAL/"normalization.npz")
    settings = {"method": winner, "threshold": threshold, "section_cv_recall_at_10pct_fpr": cv,
                "validation": binary_metrics(val_y, val_scores, threshold)}
    (FINAL/"settings.json").write_text(json.dumps(settings, indent=2))
    print("Frozen", winner, threshold, flush=True)
    deployed = Predictor()
    test = read_recordings("target_test")
    verify_disjoint(normal, train, val, test)
    scores = np.concatenate([deployed.scores(batch_features([r.path for r in test[i:i+48]])) for i in range(0,len(test),48)])
    labels = np.array([r.label for r in test])
    report = binary_metrics(labels, scores, threshold)
    (RESULTS/"target_replay.json").write_text(json.dumps(report, indent=2))
    with (RESULTS/"target_predictions.csv").open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recording","label","anomaly_score","prediction"])
        writer.writerows((r.id,r.label,float(s),int(s>=threshold)) for r,s in zip(test,scores))
    print(report, flush=True)


if __name__ == "__main__":
    main()
