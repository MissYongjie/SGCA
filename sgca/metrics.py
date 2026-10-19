"""Binary change detection metrics (OA, Kappa, F1) and error-map rendering."""
import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.ravel() > 0, gt.ravel() > 0
    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    return tp, tn, fp, fn


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp, tn, fp, fn = confusion(pred, gt)
    n = tp + tn + fp + fn
    oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {"OA": oa, "Kappa": kappa, "F1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB error map: TP white, TN black, FP red, FN green (same legend as the paper)."""
    pred, gt = pred > 0, gt > 0
    rgb = np.zeros((*gt.shape, 3), np.uint8)
    rgb[pred & gt] = (255, 255, 255)
    rgb[pred & ~gt] = (255, 0, 0)
    rgb[~pred & gt] = (0, 255, 0)
    return rgb
