# src/training/metrics.py
# 이진: AUC/Acc/Balanced Acc/F1/Sensitivity/Specificity/PPV/NPV (threshold 0.5)
# 다중: Acc/Balanced Acc/Macro F1/Macro AUC(OvR)
# 순서형(Sydney 항목별 Grade): QWK/MAE/±1 정확도/Acc/Balanced Acc/Macro F1/Macro AUC(OvR)

import warnings
from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    labels: (N,) int class ids
    probs:  (N, C) softmax probabilities
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n_classes = probs.shape[1]

    if n_classes == 2:
        preds = (probs[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics = {
            "auc": roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) == 2 else float("nan"),
            "accuracy": accuracy_score(labels, preds),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "sensitivity": _safe_div(tp, tp + fn),
            "specificity": _safe_div(tn, tn + fp),
            "ppv": _safe_div(tp, tp + fp),
            "npv": _safe_div(tn, tn + fn),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    else:
        preds = probs.argmax(axis=1)
        present = np.unique(labels)
        if len(present) == n_classes:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        else:
            auc = float("nan")  # OvR macro AUC needs every class present
        metrics = {
            "auc": auc,
            "accuracy": accuracy_score(labels, preds),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
        }
    return {k: float(v) if not isinstance(v, int) else v for k, v in metrics.items()}


LENIENT_MAP = {0: {0}, 1: {1, 2}, 2: {1, 2, 3}, 3: {2, 3}}


def lenient_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    SydneyMTL (Kim et al. 2026) lenient accuracy: grade 0 must match exactly, grades 1-3 accept the adjacent
    grade but never cross the absent (0) / present (1-3) boundary. Rows with label < 0 are skipped.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    valid = labels >= 0
    if not valid.any():
        return float("nan")
    return float(np.mean([p in LENIENT_MAP[int(y)] for y, p in zip(labels[valid], preds[valid])]))


def compute_ordinal_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    One ordinal task (e.g. one Sydney item, grades 0..K-1). Rows with label < 0 (missing) are skipped.
    labels, preds: (N,) grades;  probs: (N, K) grade probabilities
    qwk = quadratic weighted Cohen's kappa, mae = mean |pred - label|, within1 = share with |pred - label| <= 1,
    auc = macro OvR AUC over the grades present in labels (absent grades are skipped instead of giving NaN)
    """
    labels, probs, preds = np.asarray(labels), np.asarray(probs), np.asarray(preds)
    valid = labels >= 0
    y, p, pred = labels[valid], probs[valid], preds[valid]
    n_grades = p.shape[1]
    if len(y) == 0:
        return {k: float("nan") for k in ("qwk", "mae", "within1", "accuracy", "balanced_accuracy", "f1", "auc")} | {"n": 0}

    aucs = [roc_auc_score(y == g, p[:, g]) for g in range(n_grades) if 0 < (y == g).sum() < len(y)]
    with warnings.catch_warnings():  # 한 Grade만 있는 작은 fold에서 나오는 0/0, "y_pred contains classes not in y_true"
        warnings.simplefilter("ignore")
        qwk = cohen_kappa_score(y, pred, labels=list(range(n_grades)), weights="quadratic")
        bacc = balanced_accuracy_score(y, pred)
    err = np.abs(pred - y)
    return {
        "qwk": float(qwk),
        "mae": float(err.mean()),
        "within1": float((err <= 1).mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(bacc),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n": int(len(y)),
    }
