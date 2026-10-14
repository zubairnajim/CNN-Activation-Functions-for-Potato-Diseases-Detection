from typing import Any, Dict, Optional

import numpy as np

from potato_rev.config import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = NUM_CLASSES) -> np.ndarray:
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def binary_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC through the rank-sum (Mann-Whitney U) identity, with average ranks for ties."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = np.asarray(score)[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                           n_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    """
    Accuracy, error rate and macro-averaged precision / recall / specificity / F1 from the
    confusion matrix, plus one-vs-rest ROC-AUC when class probabilities are given.
    Macro recall equals balanced accuracy.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    safe = lambda num, den: np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)
    precision, recall, specificity = safe(tp, tp + fp), safe(tp, tp + fn), safe(tn, tn + fp)
    f1 = safe(2 * precision * recall, precision + recall)
    support = cm.sum(1)

    auc = [float("nan")] * n_classes
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        auc = [binary_auc(y_true == c, y_prob[:, c]) for c in range(n_classes)]

    accuracy = float(tp.sum() / cm.sum())
    return {
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "specificity_macro": float(specificity.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * support).sum() / support.sum()),
        "auc_macro": float(np.nanmean(auc)) if y_prob is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": [{
            "class": CLASS_NAMES[c] if n_classes == NUM_CLASSES else str(c),
            "support": int(support[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "specificity": float(specificity[c]),
            "f1": float(f1[c]),
            "auc": auc[c],
        } for c in range(n_classes)],
    }
