"""Binary recording-level metrics and source-validation threshold selection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)


def choose_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    best = max(finite, key=lambda i: (round(tpr[i] - fpr[i], 12), tpr[i]))
    lower = scores[scores < thresholds[best]]
    return float((thresholds[best] + lower.max()) / 2 if lower.size else thresholds[best])


def binary_metrics(labels, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "recordings": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(fp / (tn + fp)),
        "false_negative_rate": float(fn / (tp + fn)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(("normal", "anomaly"))
        },
    }
